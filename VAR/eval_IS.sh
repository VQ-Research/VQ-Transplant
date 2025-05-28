#!/bin/bash
#SBATCH --job-name=eval_var_IS
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --cpus-per-task 12
#SBATCH --nodelist=g[003-006]
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output /projects/yuanai/processed_data/rFID/baselines/var_eval_IS.out
#SBATCH --error /projects/yuanai/processed_data/rFID/baselines/var_eval_IS_error.out

source ~/.bashrc
conda activate /home/fangxian/packages/anaconda/envs/share_VAR
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=35531 eval_IS.py