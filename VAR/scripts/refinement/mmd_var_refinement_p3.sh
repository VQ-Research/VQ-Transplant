#!/bin/bash
#SBATCH --job-name=mmd_var_refinement_p3
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b4,gpubase_h100_b5
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant3/slurm/Refinement/ImageNet/mmd_var_refinement_p3.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant3/slurm/Refinement/ImageNet/mmd_var_refinement_p3.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12381 train_refinement.py --VQ=mmd_vq --dataset_name=ImageNet --global_batch_size=64 --codebook_size 4096  --codebook_dim=32 --use_multiscale --stage=refinement --alpha=1.0 --beta=1.0 --gamma=1.0 --refinement_epochs=10 --checkpoint_name checkpoint-mmd_vq_transplant_True_ImageNet_model_4096_32_1_loss_1.0_1.0_0.5_0.2-1.pth.tar