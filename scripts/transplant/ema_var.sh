#!/bin/bash
#SBATCH --job-name=ema_var
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_l40s_b4
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:l40s:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Transplant/ImageNet/ema_var.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Transplant/ImageNet/ema_var.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12262 train_VAR_transplant.py --VQ=ema_vq --dataset_name=ImageNet --global_batch_size=128 --codebook_size 4096  --codebook_dim=16 --use_multiscale --stage=transplant --alpha=1.0 --beta=1.0 --gamma=0.0
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12262 train_VAR_transplant.py --VQ=ema_vq --dataset_name=ImageNet --global_batch_size=128 --codebook_size 8192  --codebook_dim=16 --use_multiscale --stage=transplant --alpha=1.0 --beta=1.0 --gamma=0.0