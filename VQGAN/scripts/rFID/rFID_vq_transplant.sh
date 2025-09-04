#!/bin/bash
#SBATCH --job-name=rfid_vq_transplant
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=g[010-019]
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant2/metrics/Transplant/ImageNet/rfid_vq_transplant.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant2/metrics/Transplant/ImageNet/rfid_vq_transplant.err

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/FID
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name ema_vq_transplant_16384_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name ema_vq_transplant_32768_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name ema_vq_transplant_65536_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name mmd_vq_transplant_16384_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name mmd_vq_transplant_32768_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name mmd_vq_transplant_65536_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name online_vq_transplant_16384_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name online_vq_transplant_32768_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name online_vq_transplant_65536_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name vanilla_vq_transplant_16384_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name vanilla_vq_transplant_32768_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name vanilla_vq_transplant_65536_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name wasserstein_vq_transplant_16384_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name wasserstein_vq_transplant_32768_False.npz
CUDA_VISIBLE_DEVICES="0" python /projects/yuanai/projects/VQ-Transplant2/code/evaluator.py --sample_path /projects/yuanai/projects/VQ-Transplant2/reconstruction/Transplant/ImageNet --sample_name wasserstein_vq_transplant_65536_False.npz