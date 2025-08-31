#!/bin/bash
#SBATCH --job-name=vanilla_vq_imagenet_p2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=g[007-009]
#SBATCH --gpus-per-node=2
#SBATCH --time=1-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant2/slurm/Transplant/ImageNet/vanilla_vq_imagenet_p2.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant2/slurm/Transplant/ImageNet/vanilla_vq_imagenet_p2.err

source ~/.bashrc
conda activate /projects/yuanai/fangxian/packages/anaconda/envs/VQ-Tokenizer
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12873 train_VQ_transplant.py --VQ=vanilla_vq --dataset_name=ImageNet --global_batch_size=64 --codebook_size 65536  --codebook_dim=16 --stage=transplant --alpha=1.0 --beta=1.0 --gamma=0.0