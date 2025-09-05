#!/bin/bash
#SBATCH --job-name=wasserstein_other_p2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=g[007-009]
#SBATCH --gpus-per-node=2
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant3/slurm/Transplant/ImageNet/wasserstein_other_p2.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant3/slurm/Transplant/ImageNet/wasserstein_other_p2.err

source ~/.bashrc
conda activate /projects/yuanai/fangxian/packages/anaconda/envs/VQ-Tokenizer
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=18256 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=FFHQ --path=bc --global_batch_size=64 --codebook_size 16384  --codebook_dim=16 --stage=transplant --alpha=1.0 --beta=0.2 --gamma=1.0
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=18256 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=FFHQ --path=bc --global_batch_size=64 --codebook_size 32768  --codebook_dim=16 --stage=transplant --alpha=1.0 --beta=0.2 --gamma=1.0