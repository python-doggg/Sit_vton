conda activate SiT
torchrun --nnodes=1 --nproc_per_node=2 python train.py --model SiT-XL/2 --data-path /home/pengjie/data/SHHQ/SHHQ-1.0/no_segment/ --wandb
# -m torch.distributed.launch 