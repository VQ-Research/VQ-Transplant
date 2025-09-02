#!/bin/bash
#SBATCH --job-name=rfid_var_refinement_p1
#SBATCH --account=aip-rudner
#SBATCH --partition=gpubase_h100_b2
#SBATCH --nodes=1
#SBATCH --mem=50gb
#SBATCH --cpus-per-task 10
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output /project/6105494/sunset/VQ-Projects/VQ-Transplant3/metrics/Refinement/ImageNet/rfid_var_refinement_p1.out
#SBATCH --error /project/6105494/sunset/VQ-Projects/VQ-Transplant3/metrics/Refinement/ImageNet/rfid_var_refinement_p1.err

source /home/sunset/environment/FID/bin/activate
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction2/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_True_1.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction2/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_True_2.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction2/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_True_3.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction2/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_True_4.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction2/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_True_5.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction2/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_True_6.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction2/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_True_7.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction2/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_True_8.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction2/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_True_9.npz
CUDA_VISIBLE_DEVICES="0" python /project/6105494/sunset/VQ-Projects/VQ-Transplant3/code/evaluator.py --sample_path /project/6105494/sunset/VQ-Projects/VQ-Transplant3/reconstruction2/Refinement/ImageNet --sample_name mmd_vq_refinement_4096_True_10.npz
