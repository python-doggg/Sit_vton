# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from download import find_model
from models_vton import SiT_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
import sys
from time import time

from create_data import create_dataloader # add
from einops import repeat # add
import open_clip # add
from PIL import Image, ImageDraw, ImageFont # add
from torchvision import transforms


def draw_text(prompt):
    # 创建一个与图像同宽但高度任意的白色画布

    canvas = torch.ones(3, 1024, 768) # (batch_size, height, width)，1是白色

    # 使用PIL将文本绘制到画布上
    # 首先将PyTorch张量转换为PIL图像
    canvas_pil = transforms.ToPILImage()(canvas).convert("RGB") # 将张量canvas转换为PIL图像
    draw = ImageDraw.Draw(canvas_pil)

    # 设置字体和大小
    # 如果没有以下字体文件，请替换为系统上可用的字体文件路径
    font = ImageFont.truetype("/usr/share/fonts/smc/Meera.ttf", 72)
    # font = ImageFont.truetype(size=36)

    # 写入文本
    # text_width, text_height = draw.textlength(prompt, font=font), 36
    # text_x = (canvas_pil.width - text_width) // 2
    # text_y = (canvas_pil.height - text_height) // 2
    print("prompt:",prompt)
    t = len(prompt) // 25
    for i in range(t):
        draw.text((0, 400+i*60), prompt[25*i:25*(i+1)], font=font, fill="black")
    draw.text((0, 400+t*60), prompt[25*t:], font=font, fill="black")

    #prompt = list(prompt.split(","))
    # print("change prompt:",prompt)
    '''
    draw.text((0, 467), prompt[0], font=font, fill="black") ####已改
    draw.text((0, 497), prompt[1], font=font, fill="black")
    draw.text((0, 527), prompt[2], font=font, fill="black")
    '''

    # 将PIL图像转换回PyTorch张量
    canvas = transforms.ToTensor()(canvas_pil)

    return canvas


def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = True #learn_sigma = False i don't know why it's set to True

    # Load model:
    latent_size = (args.image_size[0] // 8, args.image_size[1] // 8)#latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device)
    print("model", model)
    #print("model.shape", model.shape)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )
            
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Load the clip model, add
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32')
    state_dict = torch.load('weights/finetuned_clip.pt', map_location='cuda')
    clip_model.load_state_dict(state_dict['CLIP'])
    clip_model = clip_model.eval().requires_grad_(False).cuda()
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # add begin
    #sshq_dataloader = create_dataloader(args.num_workers, batch_size=1, is_inference=True, dataset="SHHQ", clip_transformer=preprocess) #
    vitonhd_dataloader = create_dataloader(args.num_workers, batch_size=1, is_inference=True, dataset="VITON_HD", clip_tokenizer=clip_tokenizer)
    #dresscode_dataloader = create_dataloader(args.num_workers, batch_size=1, is_inference=True, dataset="DressCode", clip_tokenizer=clip_tokenizer)

    for data in vitonhd_dataloader: # vitonhd_dataloader
        person_z = data["person"].to(device)
        agnostic_z = data["agnostic"].to(device)
        densepose_z = data["densepose"].to(device)
        parse_agnostic_z = data["parse_agnostic"].to(device)
        print("person_z", person_z.shape)
        print("agnostic_z", agnostic_z.shape)
        print("densepose_z", densepose_z.shape)
        print("parse_agnostic_z", parse_agnostic_z.shape)
        print((densepose_z[:, 0, :, :]==densepose_z[:, 1, :, :]).all())
        print((densepose_z[:, 2, :, :] == densepose_z[:, 1, :, :]).all())
        with torch.no_grad():
            person_z = vae.encode(person_z).latent_dist.sample().mul_(0.18215)  # 缩放因子0.18215
            agnostic_z = vae.encode(agnostic_z).latent_dist.sample().mul_(0.18215)
            densepose_z = vae.encode(densepose_z).latent_dist.sample().mul_(0.18215)
            parse_agnostic_z = vae.encode(parse_agnostic_z).latent_dist.sample().mul_(0.18215)

            if "prompt" in data:
                # print(data["prompt"].shape)
                print("prompt guided")
                print(data["prompt"])
                image_text_emb = clip_model.encode_text(data["prompt"].cuda(), normalize=True)
            else:
                print("cloth guided")
                image_text_emb =  clip_model.encode_image(data["cloth"].cuda(), normalize=True)

            image_text_emb = image_text_emb.unsqueeze(1).unsqueeze(1)
            image_text_emb = repeat(image_text_emb, 'b 1 1 d -> b 1 k d', k=120)

            print("image_text_emb:", image_text_emb.shape)


            data_info = {
                "agnostic": agnostic_z,
                "densepose": densepose_z,
                "parse_agnostic": parse_agnostic_z,
                "img_hw": torch.tensor([[person_z.shape[2], person_z.shape[3]]], device=person_z.device),
                "aspect_ratio": torch.tensor([[person_z.shape[2]/person_z.shape[3]]], device=person_z.device),
                "class_tag": torch.tensor(data["tag"], device=person_z.device),
            }


            # # Sample a random timestep for each image
            bs = person_z.shape[0]
            #timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=person_z.device).long()
            timesteps = torch.randint(0, 10000, (bs,), device=person_z.device).long()
            #null_y = model.y_embedder.y_embedding.repeat(bs,1, 1).unsqeenze(1)# .repeat(bs, 1, 1)
            null_y = model.y_embedder.y_embedding[None].repeat(bs, 1, 1)[:, None]
            print("null_y:", null_y.shape)

            z = torch.randn(bs, 4, person_z.shape[2], person_z.shape[3], device=person_z.device).repeat(2, 1, 1, 1)
            #print("image_text_emb:", image_text_emb.shape)
            #print("null_y:", null_y.shape) # torch.Size([1, 1, 120, 512])
            model_kwargs = dict(y=torch.cat([image_text_emb, null_y]),
                                cfg_scale=4.5, data_info=data_info, mask=None)

            # Sample images:
            start_time = time()
            samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            samples = vae.decode(samples / 0.18215).sample
            print(f"Sampling took {time() - start_time:.2f} seconds.")
            person_img = data["person"].to(device)

            if "prompt" in data: # add all
                text_prompts = data["text_prompt"]
                cloth_torchs = [draw_text(text_prompt) for text_prompt in text_prompts]
                cloth_torchs = torch.stack(cloth_torchs, dim=0).to(device=samples.device)
            else:
                cloth_torchs = data["cloth_torch"]

            for i, sample in enumerate(samples):
                # text_img = draw_text(prompts[i]).to(device=device)
                combined_image = torch.cat((person_img[i], cloth_torchs[i], sample), dim=2)
                #save_image(sample, "./sample_results/dc3_20000d/" + data['seed'][0], nrow=1, normalize=True, value_range=(-1, 1))
                save_image(combined_image, "./sample_results/ziti_vtonhd/" + data['seed'][0], nrow=1, normalize=True, value_range=(-1, 1))

            # Save and display images:
            #save_image(samples, "./sample_results/sample05_130000/"+data['seed'][0], nrow=4, normalize=True, value_range=(-1, 1))


    print("finish")
'''
    # Labels to condition the model with (feel free to change):
    #class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    
    # Create sampling noise:
    #n = len(class_labels)
    # z = torch.randn(n, 4, latent_size, latent_size, device=device)
    z = torch.randn(n, 4, latent_size[0], latent_size[1], device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    start_time = time()
    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--sampler-type", type=str, default="ODE", choices=["ODE", "SDE"])
    
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema") # default="mse"
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=(1024, 768)) # type=int default=256
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    parser.add_argument("--num-workers", type=int, default=4) #


    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    
    args = parser.parse_known_args()[0]
    main(mode, args)
