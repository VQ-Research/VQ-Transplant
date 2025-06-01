#!/bin/bash
#SBATCH --account=def-kdhkdh
#SBATCH --mem=10GB
#SBATCH --gpus-per-task=1
#SBATCH --time=0-04:00:00
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=text.err
#SBATCH --mail-user=bbingqing.li@mail.utoronto.ca
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate /home/libingq9/projects/def-kdhkdh/libingq9/packages/anaconda/envs/share_VAR
CUDA_VISIBLE_DEVICES="0" python -m torch.distributed.launch --nproc_per_node=1 --master_port=12223 test.py