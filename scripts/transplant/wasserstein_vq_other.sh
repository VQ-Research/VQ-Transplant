#!/bin/bash
#SBATCH --job-name=wasserstein_vq_other
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b4
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:h100:2
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Transplant/ImageNet/wasserstein_vq_other.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant/slurm/Transplant/ImageNet/wasserstein_vq_other.err

module load gcc opencv/4.8.1
source /home/sunset/environment/VQ-Tokenizer/bin/activate
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12256 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=FFHQ --global_batch_size=128 --codebook_size 16384  --codebook_dim=8 --stage=transplant --alpha=1.0 --beta=1.0 --gamma=1.0
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12256 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=CelebAHQ --global_batch_size=128 --codebook_size 16384  --codebook_dim=8 --stage=transplant --alpha=1.0 --beta=1.0 --gamma=1.0
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12256 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=Churches --global_batch_size=128 --codebook_size 16384  --codebook_dim=8 --stage=transplant --alpha=1.0 --beta=1.0 --gamma=1.0
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=2 --master_port=12256 train_VQ_transplant.py --VQ=wasserstein_vq --dataset_name=Bedrooms --global_batch_size=128 --codebook_size 16384  --codebook_dim=8 --stage=transplant --alpha=1.0 --beta=1.0 --gamma=1.0