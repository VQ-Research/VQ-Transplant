#!/bin/bash
#SBATCH --job-name=wasserstein_vq_multiscale
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --nodelist=g[003-009]
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant/VAR/slurm/Substitution/ImageNet/wasserstein_vq_multiscale.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant/VAR/slurm/Substitution/ImageNet/wasserstein_vq_multiscale.err

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/share_VAR
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=12221 train_substitution.py --VQ=wasserstein_vq --dataset_name=ImageNet --global_batch_size=64 --factor=16 --resolution=256 --codebook_size 4096  --codebook_dim=32 --stage=substitution --use_multiscale --gamma_1=0.2
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=12221 train_substitution.py --VQ=wasserstein_vq --dataset_name=ImageNet --global_batch_size=64 --factor=16 --resolution=256 --codebook_size 8192  --codebook_dim=32 --stage=substitution --use_multiscale --gamma_1=0.2
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=12221 train_substitution.py --VQ=wasserstein_vq --dataset_name=ImageNet --global_batch_size=64 --factor=16 --resolution=256 --codebook_size 16384  --codebook_dim=32 --stage=substitution --use_multiscale --gamma_1=0.2
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=12221 train_substitution.py --VQ=wasserstein_vq --dataset_name=ImageNet --global_batch_size=64 --factor=16 --resolution=256 --codebook_size 32768  --codebook_dim=32 --stage=substitution --use_multiscale --gamma_1=0.2