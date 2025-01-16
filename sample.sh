#!/usr/bin/bash
#SBATCH -o t_c.%j.out
#SBATCH --partition=ai_training
#SBATCH --qos=medium
#SBATCH -J sample
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=dx-ai-node9

eval "$(conda shell.bash hook)"
conda activate sitt
python sample_vton.py ODE --model SiT-XL/2 --ckpt /home/pengjie/SiT_0/results/good-05-SiT-XL-2-Linear-velocity-None/checkpoints/0020000.pt