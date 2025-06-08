#!/bin/bash
#SBATCH --job-name=wasserstein_vq_p2
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 8
#SBATCH --nodelist=g[003-009]
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant/VAR/metrics/Substitution/ImageNet/wasserstein_vq_p2.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant/VAR/metrics/Substitution/ImageNet/wasserstein_vq_p2.err

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/FID
CUDA_VISIBLE_DEVICES="0" python /home/fangxian/VQ-Projects/VQ-Transplant/VAR/Substitution/evaluator.py --sample_name wasserstein_vq_8192_32_True.npz