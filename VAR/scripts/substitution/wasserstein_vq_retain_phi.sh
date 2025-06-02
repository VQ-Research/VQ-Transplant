#!/bin/bash
#SBATCH --job-name=wasserstein_vq_retain_phi
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=30gb
#SBATCH --cpus-per-task 10
#SBATCH --nodelist=g[003-009]
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant/VAR/slurm/Substitution/ImageNet/wasserstein_vq_retain_phi.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant/VAR/slurm/Substitution/ImageNet/wasserstein_vq_retain_phi.err

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/share_VAR
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 train_substitution.py --VQ=wasserstein_vq --dataset_name=ImageNet --global_batch_size 64 --factor 16 --resolution 256 --codebook_size 16384 --codebook_dim=8 --stage=substitution --use_multiscale --add_projection --gamma_1=0.2 --beta=0.1