#!/bin/bash
#SBATCH --job-name=wasserstein_vq_p1
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b4
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Transplant/ImageNet/wasserstein_vq_p1.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Transplant/ImageNet/wasserstein_vq_p1.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=12251 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=ImageNet --global_batch_size=64 --codebook_size 16384  --codebook_dim=32 --stage=transplant --alpha=1.0 --beta=1.0 --gamma=0.5