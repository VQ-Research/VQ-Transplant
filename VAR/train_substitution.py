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
from torch import distributed as tdist
import itertools

import config
from utils.util import Logger, LossManager, Pack, adjust_learning_rate
from data import dataloader
from model.var_substitution import VAR_Substitution
from metric.metric import PSNR, LPIPS, SSIM
from eval_reconstruction import eval_reconstruction

import warnings
warnings.filterwarnings('ignore')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def eval_one_epoch(args, model, epoch, val_dataloader, len_val_set):
    model.eval()
    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    if args.VQ == "var_no_vq":
        ssim, psnr, lpips, rec_loss_scalar, total_num = 0.0, 0.0, 0.0, 0.0, 0
    else:
        ssim, psnr, lpips, rec_loss_scalar, quantization_error, utilization, perplexity, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0 
        histogram_all: torch.Tensor = 0.0

    if self.args.VQ == "wasserstein-vq":
        wasserstein_loss_scalar = 0.0 

    for step, (x, _) in enumerate(val_dataloader):
        x = x.cuda(int(os.environ['LOCAL_RANK']), non_blocking=True)
        batch_size = x.size(0)
        total_num += batch_size
        with torch.no_grad():
            if args.VQ == "var_no_vq":
                x_rec, rec_loss = model.module.collect_eval_info(x)
            elif self.args.VQ == "wasserstein-vq":
                x_rec, rec_loss, wasserstein_loss, quant_error, histogram = model.module.collect_eval_info(x)
                histogram_all += histogram
            else:
                x_rec, rec_loss, quant_error, histogram = model.module.collect_eval_info(x)
                histogram_all += histogram

            batch_lpips = lpips_metric(x, x_rec).sum()
            x_norm = (x + 1.0)/2.0
            x_rec_norm = (x_rec + 1.0)/2.0
            batch_psnr = psnr_metric(x_norm, x_rec_norm).sum()
            batch_ssim = ssim_metric(x_norm, x_rec_norm).sum()

        handler1 = tdist.all_reduce(batch_lpips, async_op=True)
        handler2 = tdist.all_reduce(batch_psnr, async_op=True)
        handler3 = tdist.all_reduce(batch_ssim, async_op=True)
        handler1.wait()
        handler2.wait()
        handler3.wait()

        if int(os.environ['LOCAL_RANK']) == 0:
            ssim += batch_ssim.item()
            psnr += batch_psnr.item()
            lpips += batch_lpips.item()
            rec_loss_scalar += rec_loss.item() * batch_size
            if args.VQ != "var_no_vq":
                quantization_error += quant_error.item() * batch_size
            if self.args.VQ == "wasserstein-vq":
                wasserstein_loss_scalar += wasserstein_loss.item() * batch_size

    if args.VQ != "var_no_vq":
        codebook_usage_counts = (histogram_all > 0).float().sum()
        utilization  = codebook_usage_counts.item() / args.codebook_size

        avg_probs = histogram_all/histogram_all.sum(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    eval_psnr = psnr/len_val_set
    eval_ssim = ssim/len_val_set
    eval_lpips = lpips/len_val_set
    eval_rec_loss = rec_loss_scalar/total_num
    if args.VQ != "var_no_vq":
        eval_utilization = utilization
        eval_perplexity = perplexity.item()
        eval_quantization_error = quantization_error/total_num
    if self.args.VQ == "wasserstein-vq":
        eval_wasserstein_loss = wasserstein_loss_scalar/total_num

    model.train()
    if args.VQ == "var_no_vq": 
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, rec_loss=eval_rec_loss)
    elif self.args.VQ == "wasserstein-vq":
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, quant_error=eval_quantization_error, utilization=eval_utilization, perplexity=eval_perplexity, rec_loss=eval_rec_loss, wasserstein_loss=eval_wasserstein_loss)
    else:
        return Pack(psnr=eval_psnr, ssim=eval_ssim, lpips=eval_lpips, quant_error=eval_quantization_error, utilization=eval_utilization, perplexity=eval_perplexity, rec_loss=eval_rec_loss)

def calc_pretrain_var_metrics(args, model, epoch, val_dataloader, len_val_set):
    if args.VQ == "var_no_vq":
        results_eval = {'epoch':[], 'psnr':[], 'ssim':[], 'rec_loss': [], 'lpips':[]}
    elif args.VQ == 'original_var':
        results_eval = {'epoch':[], 'psnr':[], 'ssim':[], 'rec_loss': [], 'lpips':[], 'quant_error':[], 'utilization':[], 'perplexity':[]}

    with torch.no_grad():
        results_pack = eval_one_epoch(args, model, epoch, val_dataloader, len_val_set)
        if int(os.environ['LOCAL_RANK']) == 0:
            results_eval['epoch'].append(epoch)
            results_eval['psnr'].append(results_pack.psnr)
            results_eval['ssim'].append(results_pack.ssim)
            results_eval['rec_loss'].append(results_pack.rec_loss)
            results_eval['lpips'].append(results_pack.lpips)
            if args.VQ != "var_no_vq": 
                results_eval['quant_error'].append(results_pack.quant_error)
                results_eval['utilization'].append(results_pack.utilization)
                results_eval['perplexity'].append(results_pack.perplexity)
            
            results_val_len = len(results_eval['epoch'])
            data_frame = pd.DataFrame(data=results_eval, index=range(1, results_val_len+1))
            data_frame.to_csv('{}/eval_{}_rec_results.csv'.format(args.results_dir, args.saver_name_pre), index_label='index')

def main_worker(args):
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.distributed.init_process_group(backend='nccl')

    model = VAR_Substitution(args)
    model = model.cuda(int(os.environ['LOCAL_RANK']))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=False, broadcast_buffers=True)
    train_dataloader, val_dataloader, train_sampler, len_train_set, len_val_set = build_dataloader(args)

    if args.VQ == "var_no_vq" or args.VQ == 'original_var':
        epoch = 0
        #eval_reconstruction(args, model)
        calc_pretrain_var_metrics(args, model, epoch, val_dataloader, len_val_set)
        return
        
if __name__ == '__main__':
    args = config.parse_arg()
    dict_args = vars(args)
    sys.stdout = Logger(args.saver_dir, args.saver_name_pre)
    if int(os.environ['LOCAL_RANK']) == 0:
        for k, v in zip(dict_args.keys(), dict_args.values()):
            print("{0}: {1}".format(k, v))
    main_worker(args)


