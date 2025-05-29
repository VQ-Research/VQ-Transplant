#!/bin/bash
#SBATCH --job-name=var_no_vq
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --cpus-per-task 8
#SBATCH --nodelist=g[002-009]
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant/VAR/slurm/Substitution/ImageNet/var_no_vq.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant/VAR/slurm/Substitution/ImageNet/var_no_vq.err

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/share_VAR
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=12223 train_substitution.py --VQ=var_no_vq --dataset_name=ImageNet --global_batch_size 64 --factor 16 --resolution 256 --codebook_size 4096