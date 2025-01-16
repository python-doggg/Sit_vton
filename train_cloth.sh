#!/usr/bin/bash
#SBATCH -o t_c.%j.out
#SBATCH --partition=ai_training
#SBATCH --qos=medium
#SBATCH -J sit_dc
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:7

#nvidia-smi
eval "$(conda shell.bash hook)"
conda activate sitt
export WANDB_KEY="key"
export ENTITY="entity name"
export PROJECT="t_c"
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG=INFO


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6" python -m torch.distributed.launch  --nproc_per_node=7 --master_port=5678 --use_env train_vton.py --model SiT-XL/2 --data-path /home/pengjie/data/VITON-HD --global-batch-size 28 #--ckpt /home/pengjie/SiT_0/results/good-05-SiT-XL-2-Linear-velocity-None/checkpoints/0050000.pt
