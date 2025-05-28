import torch
import os
from cleanfid import fid
from pytorch_image_generation_metrics import (
    get_inception_score_from_directory,
    get_fid_from_directory,
    get_inception_score_and_fid_from_directory)

def main_worker():
    rec_image_path = "/projects/yuanai/processed_data/rFID/baselines/VAR"
    IS, IS_std = get_inception_score_from_directory(rec_image_path)
    print("IS:"+str(IS)+"  IS_std:"+str(IS_std))

if __name__ == '__main__':
    main_worker()  
