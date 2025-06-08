#!/bin/bash
#SBATCH --job-name=var_no_vq
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 8
#SBATCH --nodelist=g[003-009]
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant/VAR/metrics/Substitution/ImageNet/var_no_vq.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant/VAR/metrics/Substitution/ImageNet/var_no_vq.err

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/FID
CUDA_VISIBLE_DEVICES="0" python /home/fangxian/VQ-Projects/VQ-Transplant/VAR/Substitution/evaluator.py --sample_name var_no_vq.npz