#!/bin/bash
#SBATCH --job-name=mmd_vq_refinement_other_p2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=g[007-009]
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant2/slurm/Refinement/ImageNet/mmd_vq_refinement_other_p2.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant2/slurm/Refinement/ImageNet/mmd_vq_refinement_other_p2.err

source ~/.bashrc
conda activate /projects/yuanai/fangxian/packages/anaconda/envs/VQ-Tokenizer
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=16652 train_refinement.py --VQ=mmd_vq --dataset_name=CelebAHQ --path=bc --global_batch_size=64 --codebook_size 16384  --codebook_dim=8 --stage=refinement --alpha=1.0 --beta=1.0 --gamma=1.0 --checkpoint_name checkpoint-mmd_vq_transplant_False_CelebAHQ_model_16384_8_1_loss_1.0_1.0_1.0_0.4.pth.tar
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=16652 train_refinement.py --VQ=mmd_vq --dataset_name=Churches --path=bc --global_batch_size=64 --codebook_size 16384  --codebook_dim=8 --stage=refinement --alpha=1.0 --beta=1.0 --gamma=1.0 --checkpoint_name checkpoint-mmd_vq_transplant_False_Churches_model_16384_8_1_loss_1.0_1.0_1.0_0.4.pth.tar