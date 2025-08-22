#!/bin/bash
#SBATCH --job-name=online_var_refinement
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=10
#SBATCH --time=5-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Refinement/ImageNet/online_var_refinement.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Refinement/ImageNet/online_var_refinement.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12283 train_refinement.py --VQ=online_vq --dataset_name=ImageNet --global_batch_size=64 --codebook_size 4096  --codebook_dim=16 --use_multiscale --stage=refinement --alpha=1.0 --beta=1.0 --gamma=0.0 --checkpoint_name checkpoint-online_vq_transplant_True_ImageNet_model_4096_16_1_loss_1.0_1.0_0.0_0.2.pth.tar
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12283 train_refinement.py --VQ=online_vq --dataset_name=ImageNet --global_batch_size=64 --codebook_size 8192  --codebook_dim=16 --use_multiscale --stage=refinement --alpha=1.0 --beta=1.0 --gamma=0.0 --checkpoint_name checkpoint-online_vq_transplant_True_ImageNet_model_8192_16_1_loss_1.0_1.0_0.0_0.2.pth.tar