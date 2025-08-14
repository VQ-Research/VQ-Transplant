#!/bin/bash
#SBATCH --job-name=wasserstein_vq_p1
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --nodelist=g[003-009]
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant/VAR/slurm/Substitution/ImageNet/ema_vq_multiscale.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant/VAR/slurm/Substitution/ImageNet/ema_vq_multiscale.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate

CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=12251 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=ImageNet --global_batch_size=32 --codebook_size 16384  --codebook_dim=32 --stage=transplant