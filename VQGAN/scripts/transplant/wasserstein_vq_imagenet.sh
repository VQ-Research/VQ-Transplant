#!/bin/bash
#SBATCH --job-name=wasserstein_vq_imagenet
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=gb001
#SBATCH --gpus-per-node=2
#SBATCH --time=1-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant3/slurm/Transplant/ImageNet/wasserstein_vq_imagenet.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant3/slurm/Transplant/ImageNet/wasserstein_vq_imagenet.err

source ~/.bashrc
conda activate /projects/yuanai/fangxian/packages/anaconda/envs/VQ-Tokenizer
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12251 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=ImageNet --path=bc --global_batch_size=64 --codebook_size 16384  --codebook_dim=16 --stage=transplant --alpha=1.0 --beta=0.2 --gamma=1.0
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12251 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=ImageNet --path=bc --global_batch_size=64 --codebook_size 65536  --codebook_dim=16 --stage=transplant --alpha=1.0 --beta=0.2 --gamma=1.0
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12251 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=ImageNet --path=bc --global_batch_size=64 --codebook_size 32768  --codebook_dim=16 --stage=transplant --alpha=1.0 --beta=0.2 --gamma=1.0