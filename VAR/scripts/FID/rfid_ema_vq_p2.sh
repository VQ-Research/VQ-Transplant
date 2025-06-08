#!/bin/bash
#SBATCH --job-name=ema_vq_p1
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 8
#SBATCH --nodelist=g[003-009]
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant/VAR/metrics/Substitution/ImageNet/ema_vq_p1.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant/VAR/metrics/Substitution/ImageNet/ema_vq_p1.err

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/FID
CUDA_VISIBLE_DEVICES="0" python /home/fangxian/VQ-Projects/VQ-Transplant/VAR/Substitution/evaluator.py --sample_name ema_vq_8192_32_True.npz