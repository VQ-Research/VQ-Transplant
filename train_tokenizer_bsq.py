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
from torch.nn.parallel import DistributedDataParallel as DDP
import ruamel.yaml as yaml

import config
from utils.util import Logger, LossManager, Pack, adjust_learning_rate, save_checkpoint
from data import dataloader
from models.tokenizer import VQModel
from models.vq_loss import VQLoss
from metric.metric import PSNR, LPIPS, SSIM
from eval_tokenizer import eval_one_epoch

from timm.scheduler import create_scheduler_v2 as create_scheduler
from utils.distributed import init_distributed_mode

os.environ["TORCHDYNAMO_LOGLEVEL"] = "INFO"
os.environ["TORCHDYNAMO_VERBOSE"] = "1" 

import warnings
warnings.filterwarnings('ignore')

def main_worker(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    vq_model = VQModel(args)
    total_para = 0
    for p in vq_model.encoder.parameters():
        total_para += p.numel()
    for p in vq_model.decoder.parameters():
        total_para += p.numel()
    print("VQ Model Parameters:", total_para)
    
    vq_model = vq_model.to(device)    
    vq_model = nn.SyncBatchNorm.convert_sync_batchnorm(vq_model)

    model_para = list(vq_model.encoder.parameters()) 
    code_para = list(vq_model.quantizer.projector_in.parameters()) + list(vq_model.quantizer.projector_out.parameters())
    optimizer = torch.optim.AdamW([{'params': model_para}, {'params': code_para, 'lr': 0.0001}], lr=args.lr_transplant, betas=(0.9, 0.95), weight_decay=0.0001)
    train_dataloader, val_dataloader, train_sampler, len_train_set, len_val_set = build_dataloader(args)

    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=False)
    vq_model.train()

    results_eval = {'epoch':[], 'psnr':[], 'ssim':[], 'lpips':[], 'rec_loss': [], 'vq_loss': [], 'prob_commit_loss':[], 'entropy_loss':[], 'avg_entropy_loss':[]}
    train_loss = LossManager()
    print("Start training...")
    start_epoch = 1 
    for epoch in range(start_epoch, args.transplant_epochs+1):
        train_sampler.set_epoch(epoch)
        print("epoch:%d, cur_lr:%4f"%(epoch, optimizer.param_groups[0]["lr"]))
    
        start_time = time.time()
        for step, (x, _) in enumerate(train_dataloader):
            cur_iter = len(train_dataloader) * (epoch-1) + step
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                x = x.to(device, non_blocking=True)

                optimizer.zero_grad()
                transplant_loss, rec_loss, vq_loss, categorical_loss, prob_commit_loss, entropy_loss, avg_entropy_loss = vq_model.module.transplant(x, cur_iter)
                info_pack = Pack(transplant_loss=transplant_loss, rec_loss=rec_loss, vq_loss=vq_loss, categorical_loss=categorical_loss, prob_commit_loss=prob_commit_loss, entropy_loss=entropy_loss, avg_entropy_loss=avg_entropy_loss)
                
                transplant_loss.backward()
                torch.nn.utils.clip_grad_norm_(model_para, 1.0)
                optimizer.step()

            train_loss.add_loss(info_pack)
            if int(os.environ['LOCAL_RANK']) == 0 and (step+1) %10 ==0:
                print(train_loss.pprint(window=50, prefix='Train Epoch: [{}/{}] Iters:[{}/{}]'.format(epoch, args.transplant_epochs, step+1, len(train_dataloader))))

        train_loss.clear()
        if epoch % args.eval_epochs == 0 and int(os.environ['LOCAL_RANK']) == 0:
            vq_model.train()
            checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-'+args.saver_name_pre+'-'+str(epoch)+'.pth.tar')
            save_checkpoint({'epoch': epoch, 'model': vq_model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'args': args}, is_best=False, filename=checkpoint_path) 
        if epoch % args.eval_epochs == 0:
            with torch.no_grad():
                results_pack = eval_one_epoch(args, vq_model, epoch, val_dataloader, len_val_set)

            if int(os.environ['LOCAL_RANK']) == 0:
                results_eval['epoch'].append(epoch)
                results_eval['psnr'].append(results_pack.psnr)
                results_eval['ssim'].append(results_pack.ssim)
                results_eval['lpips'].append(results_pack.lpips)
                results_eval['rec_loss'].append(results_pack.rec_loss)
                results_eval['vq_loss'].append(results_pack.vq_loss)
                results_eval['prob_commit_loss'].append(results_pack.prob_commit_loss)
                results_eval['entropy_loss'].append(results_pack.entropy_loss)
                results_eval['avg_entropy_loss'].append(results_pack.avg_entropy_loss)
                
                results_val_len = len(results_eval['epoch'])
                data_frame = pd.DataFrame(data=results_eval, index=range(1, results_val_len+1))
                data_frame.to_csv('{}/eval_{}_rec_results.csv'.format(args.results_dir, args.saver_name_pre), index_label='index')

    print("######### saving checkpoint #########")
    vq_model.train() 
    if int(os.environ['LOCAL_RANK']) == 0:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint-'+args.saver_name_pre+'.pth.tar')
        save_checkpoint({'epoch': epoch, 'model': vq_model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'args': args}, is_best=False, filename=checkpoint_path) 

    vq_model.eval() 
    dist.destroy_process_group()

if __name__ == '__main__':
    args = config.parse_arg()
    dict_args = vars(args)
    sys.stdout = Logger(args.saver_dir, args.saver_name_pre)
    if int(os.environ['LOCAL_RANK']) == 0:
        for k, v in zip(dict_args.keys(), dict_args.values()):
            print("{0}: {1}".format(k, v))
    main_worker(args)