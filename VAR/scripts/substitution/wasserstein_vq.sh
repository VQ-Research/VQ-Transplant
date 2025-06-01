#!/bin/bash
#SBATCH --job-name=wasserstein_vq
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=30gb
#SBATCH --cpus-per-task 10
#SBATCH --nodelist=g[003-009]
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant/VAR/slurm/Substitution/ImageNet/wasserstein_vq.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant/VAR/slurm/Substitution/ImageNet/wasserstein_vq.err

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/share_VAR
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 train_substitution.py --VQ=wasserstein-vq --dataset_name=ImageNet --global_batch_size 32 --factor 16 --resolution 256 --codebook_size 4096  --stage=substitution --use_trick=False --use_multiscale=True --fold_token=False --add_projection=False