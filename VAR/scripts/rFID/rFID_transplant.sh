#!/bin/bash
#SBATCH --job-name=rfid_var_transplant
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant3/metrics/Transplant/ImageNet/rfid_var_transplant.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant3/metrics/Transplant/ImageNet/rfid_var_transplant.err

source /home/sunset/environment/FID/bin/activate
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction/Transplant/ImageNet --sample_name ema_vq_transplant_4096_True.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction/Transplant/ImageNet --sample_name ema_vq_transplant_8192_True.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction/Transplant/ImageNet --sample_name mmd_vq_transplant_4096_True.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction/Transplant/ImageNet --sample_name mmd_vq_transplant_8192_True.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction/Transplant/ImageNet --sample_name online_vq_transplant_4096_True.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction/Transplant/ImageNet --sample_name online_vq_transplant_8192_True.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction/Transplant/ImageNet --sample_name vanilla_vq_transplant_4096_True.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction/Transplant/ImageNet --sample_name vanilla_vq_transplant_8192_True.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction/Transplant/ImageNet --sample_name wasserstein_vq_transplant_4096_True.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction/Transplant/ImageNet --sample_name wasserstein_vq_transplant_8192_True.npz