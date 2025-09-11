#!/bin/bash
#SBATCH --job-name=rfid_Churches
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --mem=100gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=gb001
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output /projects/yuanai/projects/VQ-Transplant3/metrics/rfid_Churches.out
#SBATCH --error /projects/yuanai/projects/VQ-Transplant3/metrics/rfid_Churches.err

source ~/.bashrc
conda activate /projects/yuanai/fangxian/packages/anaconda/envs/VQ-Tokenizer
CUDA_VISIBLE_DEVICES="0" python evaluator_church.py