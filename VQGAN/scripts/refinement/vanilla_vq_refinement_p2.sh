#!/bin/bash
#SBATCH --job-name=vanilla_vq_refinement_p2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=g[007-009]
#SBATCH --gpus-per-node=2
#SBATCH --time=4-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant3/slurm/Refinement/ImageNet/vanilla_vq_refinement_p2.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant3/slurm/Refinement/ImageNet/vanilla_vq_refinement_p2.err

source ~/.bashrc
conda activate /projects/yuanai/fangxian/packages/anaconda/envs/VQ-Tokenizer
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12656 train_refinement.py --VQ=vanilla_vq --dataset_name=ImageNet --path=bc --global_batch_size=32 --codebook_size 32768  --codebook_dim=16 --stage=refinement --alpha=1.0 --beta=1.0 --gamma=0.0 --checkpoint_name checkpoint-vanilla_vq_transplant_False_ImageNet_model_32768_8_1_loss_1.0_1.0_0.0_0.4.pth.tar