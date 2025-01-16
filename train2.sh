eval "$(conda shell.bash hook)"
conda activate SiT
export WANDB_KEY="key"
export ENTITY="entity name"
export PROJECT="t_c"
export NCCL_DEBUG_SUBSYS=COLL
export NCCL_DEBUG=INFO


CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch  --nproc_per_node=4 --master_port=5678 --use_env train_vton.py --model SiT-XL/2 --data-path /lustre/niuyunfang/data/SHHQ/ --global-batch-size 16