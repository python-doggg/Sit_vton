# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder, DatasetFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models_vton import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args
import wandb_utils

from create_data import create_dataloader # add
from einops import repeat # add
import open_clip # add
#from diffusion import IDDPM # add
from accelerate import Accelerator, InitProcessGroupKwargs # add
from accelerate.utils import DistributedType # add

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    import torch.distributed as dist # add
    import os # add
    os.environ['MASTER_ADDR'] = 'localhost' # add
    os.environ['MASTER_PORT'] = '5678' # add
    #local_rank = os.environ.get('LOCAL_RANK', '0') # add
    #os.environ['LOCAL_RANK'] = local_rank # add
    #torch.cuda.set_device(local_rank) # add
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl") #dist.init_process_group("nccl") # ���ڳ�ʼ����GPUѵ�������еĽ�����
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    find_unused_parameters = True # add
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.") # world_size=2
    local_batch_size = int(args.global_batch_size // dist.get_world_size())
    print("local_batch_size:", local_batch_size)

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:02d}-{model_string_name}-" \
                          f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        entity = os.environ["ENTITY"]
        project = os.environ["PROJECT"]
        if args.wandb:
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        logger = create_logger(None)

    # Create model:
    #assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = (args.image_size[0] // 8, args.image_size[1] // 8)
    #print("latent_size: ", latent_size, type(latent_size)) # (128, 64) <class 'tuple'>

    model = SiT_models[args.model]( # SiT_models is a dictionary that maps model names to model classes
        input_size=latent_size,
        num_classes=args.num_classes
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training


    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        print("state_dict", state_dict)
        model.load_state_dict(state_dict) # state_dict["model"]
        ema.load_state_dict(state_dict) # state_dict["ema"]
        #opt.load_state_dict(state_dict) # state_dict["opt"]
        #args = state_dict["args"] # state_dict["args"]

    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=find_unused_parameters) # DDP �����㽫һ��ģ�͸��Ƶ���� GPU �ϣ�ÿ�� GPU ����ģ�͵�һ����
    #train_diffusion = IDDPM(str(1000), learn_sigma=True, pred_sigma=True, snr=False)
    transport = create_transport(
        args.path_type, # 路径类型可以是"velocity"、"score"或"noise"
        args.prediction, # True使用预测模型的输出, False使用输入数据
        args.loss_weight, # 用于指定损失函数的权重
        args.train_eps, # 用于指定训练过程中的噪声水平
        args.sample_eps # 用于指定采样过程中的噪声水平
    )  # default: velocity;
    transport_sampler = Sampler(transport) # 设置一个用于生成概率流量样本的采样器
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load the clip model, add
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
    state_dict = torch.load('weights/finetuned_clip.pt', map_location='cuda')
    clip_model.load_state_dict(state_dict['CLIP'])
    clip_model = clip_model.eval().requires_grad_(False).cuda()
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0) # 0表示不使用权重衰减

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    
    #dataset = ImageFolder(args.data_path, transform=transform) ###
    #sshq_dataloader = create_dataloader(args.num_workers, batch_size=int(local_batch_size), is_inference=False, dataset="SHHQ", clip_transformer=preprocess) # add , clip_transformer=preprocess
    #vitonhd_dataloader = create_dataloader(args.num_workers, batch_size=int(local_batch_size), is_inference=False, dataset="VITON_HD", clip_tokenizer=clip_tokenizer) # add , clip_tokenizer=clip_tokenizer
    dresscode_dataloader = create_dataloader(args.num_workers, batch_size=int(local_batch_size), is_inference=False, dataset="DressCode", clip_tokenizer=clip_tokenizer)
    #frequencies = [1, 5]  # 根据需要设置

    # 创建 MultiDataLoader 实例
    #multi_dataloader = MultiDataLoader(vitonhd_dataloader, sshq_dataloader, frequencies=frequencies)
    
    sampler = DistributedSampler(
        dresscode_dataloader.dataset, #dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    '''
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    '''
    logger.info(f"Dataset contains {len(dresscode_dataloader.dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Labels to condition the model with (feel free to change):
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    # Create sampling noise:
    n = ys.size(0)
    #zs = torch.randn(n, 4, latent_size[0], latent_size[1], device=device) ##############################
    '''
    # Setup classifier-free guidance:
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward
    '''

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # for x, y in sshq_dataloader: # loader
        for data in dresscode_dataloader: # vitonhd_dataloader:
            #x = x.to(device)
            #y = y.to(device)
            #x = data["person"].to(device)
            person_z = data["person"].to(device)
            agnostic_z = data["agnostic"].to(device)
            densepose_z = data["densepose"].to(device)
            parse_agnostic_z = data["parse_agnostic"].to(device)

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                person_z = vae.encode(person_z).latent_dist.sample().mul_(0.18215) # 缩放因子0.18215
                agnostic_z = vae.encode(agnostic_z).latent_dist.sample().mul_(0.18215)
                densepose_z = vae.encode(densepose_z).latent_dist.sample().mul_(0.18215)
                parse_agnostic_z = vae.encode(parse_agnostic_z).latent_dist.sample().mul_(0.18215)

            if "prompt" in data:
                # print(data["prompt"].shape)
                image_text_emb = clip_model.encode_text(data["prompt"].cuda(), normalize=True) # add .cuda()
            else:
                image_text_emb = clip_model.encode_image(data["cloth"].cuda(), normalize=True) # add .cuda()

            image_text_emb = image_text_emb.unsqueeze(1).unsqueeze(1)
            #print("image_text_emb.shape", image_text_emb.shape) # torch.Size([2, 512])
            image_text_emb = repeat(image_text_emb, 'b 1 1 d -> b 1 k d', k=120)
            #print("image_text_emb: ", image_text_emb.shape) # torch.Size([2, 1, 120, 512])

            data_info = {
                "agnostic": agnostic_z,
                "densepose": densepose_z,
                "parse_agnostic": parse_agnostic_z,
                "img_hw": torch.tensor([[person_z.shape[2], person_z.shape[3]]], device=person_z.device),
                "aspect_ratio": torch.tensor([[person_z.shape[2] / person_z.shape[3]]], device=person_z.device),
                "class_tag": torch.tensor(data["tag"], device=person_z.device),
            }

            bs = person_z.shape[0]
            #print("bs: ", bs) # 2
            timesteps = torch.randint(0, 10000, (bs,), device=person_z.device).long()

            #model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model=model, x1=person_z, model_kwargs=dict(y=image_text_emb, data_info=data_info)) # , model_kwargs=dict(y=image_text_emb, data_info=data_info)
            loss = loss_dict["loss"].mean()


            #loss_term = train_diffusion.training_losses(model=model, x_start=person_z, t=timesteps, model_kwargs=dict(y=image_text_emb, data_info=data_info))
            #loss = loss_term['loss'].mean()
            #loss_mse = loss_term['mse'].mean()
            #loss_vb = loss_term['vb'].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        {"train loss": avg_loss, "train steps/sec": steps_per_sec},
                        step=train_steps
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0: # ckpt_every=50_000
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:0007d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            '''
            if train_steps % args.sample_every == 0 and train_steps > 0: # sample_every=10_000
                logger.info("Generating EMA samples...")
                sample_fn = transport_sampler.sample_ode()  # default to ode sampling
                samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                dist.barrier()

                if use_cfg:  # remove null samples
                    samples, _ = samples.chunk(2, dim=0)
                samples = vae.decode(samples / 0.18215).sample
                out_samples = torch.zeros((args.global_batch_size, 3, args.image_size, args.image_size), device=device)
                dist.all_gather_into_tensor(out_samples, samples)
                if args.wandb:
                    wandb_utils.log_image(out_samples, train_steps)
                logging.info("Generating EMA samples done.")
            '''

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=(1024, 768)) # default=256
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256) # 256
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000) # 50_000
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")
    #parser.add_argument("--local_rank", type=int) # add
    #parser.add_argument('--resume-from', help='the dir to save logs and models') # add

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)