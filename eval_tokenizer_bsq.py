import gc
import os
import shutil
import sys
import time
import warnings
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch import nn, optim
import math
import json
import random
import scipy.io as sio
from torch.nn import functional as F
from scipy.io import savemat
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from data.dataloader import build_dataloader
import torchvision.models as torchvision_models
from torchvision import models, datasets, transforms
from torch import distributed as dist
import itertools
from copy import deepcopy

import config
from utils.util import Logger, LossManager, Pack, adjust_learning_rate
from data import dataloader
from metric.metric import PSNR, LPIPS, SSIM
import warnings
warnings.filterwarnings('ignore')

def eval_one_epoch(args, model, epoch, val_dataloader, len_val_set):
    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    ssim, psnr, lpips, rec_loss, prob_commit_loss, entropy_loss, avg_entropy_loss, total_num =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0  
    if args.stage == "transplant":
        vq_loss = 0.0

    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(int(os.environ['LOCAL_RANK']), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size
        with torch.no_grad():
            if args.stage == "transplant":
                x_rec, rec_loss_eval, vq_loss_eval, prob_commit_loss_eval, entropy_loss_eval, avg_entropy_loss_eval = model.module.collect_eval_info_transplant(x)
                info_pack = Pack(rec_loss=rec_loss_eval, vq_loss=vq_loss_eval, prob_commit_loss=prob_commit_loss_eval, entropy_loss=entropy_loss_eval, avg_entropy_loss=avg_entropy_loss_eval)
            else:
                x_rec, rec_loss_eval, prob_commit_loss_eval, entropy_loss_eval, avg_entropy_loss_eval = model.module.collect_eval_info_refinement(x)
                info_pack = Pack(rec_loss=rec_loss_eval, prob_commit_loss=prob_commit_loss_eval, entropy_loss=entropy_loss_eval, avg_entropy_loss=avg_entropy_loss_eval)
            
            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0
            batch_lpips = lpips_metric(x_norm, x_rec_norm).sum()
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

        handler1 = dist.all_reduce(batch_lpips, async_op=True)
        handler2 = dist.all_reduce(batch_psnr, async_op=True)
        handler3 = dist.all_reduce(batch_ssim, async_op=True)
        handler1.wait()
        handler2.wait()
        handler3.wait()

        if int(os.environ['LOCAL_RANK']) == 0:
            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()
            rec_loss += info_pack.rec_loss.item() * batch_size
            if args.stage == "transplant":
                vq_loss += info_pack.vq_loss.item() * batch_size
            prob_commit_loss += info_pack.prob_commit_loss.item() * batch_size
            entropy_loss += info_pack.entropy_loss.item() * batch_size
            avg_entropy_loss += info_pack.avg_entropy_loss.item() * batch_size

    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    eval_rec_loss = rec_loss/total_num
    if args.stage == "transplant":
        eval_vq_loss = vq_loss/total_num
    eval_prob_commit_loss = prob_commit_loss/total_num
    eval_entropy_loss = entropy_loss/total_num
    eval_avg_entropy_loss = avg_entropy_loss/total_num

    model.train()
    if args.stage == "transplant":
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss, vq_loss=eval_vq_loss, prob_commit_loss=eval_prob_commit_loss, entropy_loss=eval_entropy_loss, avg_entropy_loss=eval_avg_entropy_loss)
    else:
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss, prob_commit_loss=eval_prob_commit_loss, entropy_loss=eval_entropy_loss, avg_entropy_loss=eval_avg_entropy_loss)