#!/bin/bash
#SBATCH --job-name=wasserstein_vq_refinement
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=gb001
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant3/slurm/Refinement/ImageNet/wasserstein_vq_refinement.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant3/slurm/Refinement/ImageNet/wasserstein_vq_refinement.err

source ~/.bashrc
conda activate /projects/yuanai/fangxian/packages/anaconda/envs/VQ-Tokenizer
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12652 train_refinement.py --VQ=wasserstein_vq --dataset_name=ImageNet --path=bc --global_batch_size=64 --codebook_size 16384  --codebook_dim=8 --stage=refinement --alpha=1.0 --beta=0.2 --gamma=1.0 --checkpoint_name checkpoint-wasserstein_vq_transplant_False_ImageNet_model_16384_8_1_loss_1.0_0.2_1.0_0.4.pth.tar
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12652 train_refinement.py --VQ=wasserstein_vq --dataset_name=ImageNet --path=bc --global_batch_size=64 --codebook_size 32768  --codebook_dim=8 --stage=refinement --alpha=1.0 --beta=0.2 --gamma=1.0 --checkpoint_name checkpoint-wasserstein_vq_transplant_False_ImageNet_model_32768_8_1_loss_1.0_0.2_1.0_0.4.pth.tar
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12652 train_refinement.py --VQ=wasserstein_vq --dataset_name=ImageNet --path=bc --global_batch_size=64 --codebook_size 65536  --codebook_dim=8 --stage=refinement --alpha=1.0 --beta=0.2 --gamma=1.0 --checkpoint_name checkpoint-wasserstein_vq_transplant_False_ImageNet_model_65536_8_1_loss_1.0_0.2_1.0_0.4.pth.tar