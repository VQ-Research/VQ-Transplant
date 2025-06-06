import json
import os
import random
import re
import sys
import time
import numpy as np
import torch
from collections import OrderedDict
from typing import Optional, Union
import argparse
import torch.distributed as dist

def parse_arg():
    parser = argparse.ArgumentParser(description='VQ-Transplant based on Pretrained VAR visual tokenizer.') 

    ###Dataset and Dataloader Configuration
    parser.add_argument('--dataset_dir', default="/projects/yuanai/data/", type=str, help='the directory of dataset') 
    parser.add_argument('--dataset_name', default='ImageNet', help='the name of dataset', choices=['ImageNet', 'FFHQ', 'CelebAHQ', 'Churches', 'Bedrooms'])
    parser.add_argument('--global_batch_size', type=int, default=128, help="the size of batch samples")
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
    parser.add_argument('--resolution', type=int, choices=[256], default=256, help='resolution of train and test')
    parser.add_argument('--channels', default=3, type=int, metavar='N', help='the channels of images')
    
    ###Model Configuration
    parser.add_argument('--ms_patch_size', default="1_2_3_4_5_6_8_10_13_16", type=str, help='multi-scale patch size.')
    parser.add_argument('--importance', default="1_1_1_2_2_2_3_3_5_5", type=str, help='importance of multi-scale multi-scale VQ.')
    parser.add_argument('--max_patch_size', default=16, type=int, help='the maximum patch size.')
    parser.add_argument('--codebook_size', default=4096, type=int, help='the size of codebook.')
    parser.add_argument('--codebook_dim', default=32, type=int, help='the dimension of codebook vectors.')
    parser.add_argument('--z_channels', default=256, type=int, help='the resolution of latent variables.')
    parser.add_argument('--factor', default=16, type=int, help='the downscale factor of vanilla image to the latent variable', choices=[16])
    parser.add_argument('--sigma', type=float, default=0.8, help="sigma correction in distributional method (very important in wasserstein vq and adversarile vq).")

    ###Loss Configuration
    parser.add_argument('--gamma_1', type=float, default=0.2, help="substitution stage: the hyperparameter of wasserstein_loss in wasserstein_vq.")
    parser.add_argument('--gamma_2', type=float, default=0.2, help="substitution stage: the hyperparameter of codebook d_loss and g_loss in adversarial_vq.")
    parser.add_argument('--lambd', type=float, default=0.2, help="adaptation stage: dino discriminator loss weight for gan training")

    ###Training Configuration
    parser.add_argument('--VQ', default='wasserstein_vq', help='various vq approaches.', choices=['wasserstein_vq', 'vanilla_vq', 'ema_vq', 'adversarial_vq', 'original_var', 'var_no_vq'])
    parser.add_argument('--resume', action='store_true', help='reloading model from specified checkpoint.')
    parser.add_argument('--use_multiscale', action='store_true', help='False: employ single VQ; True: use multiscale-VQ as original VAR.')
    parser.add_argument('--use_pq', action='store_true', help='False: do not use product quantization; True: use product quantization.')
    parser.add_argument('--epochs', type=int, default=1, help="training epochs, 1 epochs for ImageNet, 50 epochs for other datasets")
    parser.add_argument('--eval_epochs', type=int, default=1, help="epochs for each eval, 1 epochs for ImageNet, 5 epochs for FFHQ datasets.")
    parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate for encoder-decoder architecture.')
    parser.add_argument('--dropout', help='dropout for the model', type=float, default=0.0)
    parser.add_argument('--seed', help='random seed', type=int, default=3407)
    parser.add_argument('--iterations', default=1000, type=int, help='the iteration to teriminal the program (for check quantization-decoder mismatch).')
    parser.add_argument('--warmup_iters', help='Number of steps for warmup of lr', type=int, default=2000)
    parser.add_argument('--decay_iters', help='Number of steps for cosine decay of lr', type=int, default=10000)
    parser.add_argument('--stage', default='substitution', help='there are two stages:vq-substitution and decoder adaptation.', choices=['substitution', 'adaptation'])
    parser.add_argument('--pretrained_tokenizer', default="/projects/yuanai/projects/VQ-Transplant/VAR/pretrained_tokenizer/vae_ch160v4096z32.pth", type=str, help='the path to pretrained visual tokenizer.')
    parser.add_argument('--checkpoint_dir', default="/projects/yuanai/projects/VQ-Transplant/VAR/checkpoint/", type=str, help='the directory of checkpoint.')
    parser.add_argument('--results_dir', default="/projects/yuanai/projects/VQ-Transplant/VAR/results/", type=str, help='the directory of results.')
    parser.add_argument('--saver_dir', default="/projects/yuanai/projects/VQ-Transplant/VAR/saver/", type=str, help='the directory of saver.')
    parser.add_argument('--reconstruction_dir', default="/projects/yuanai/projects/VQ-Transplant/VAR/reconstruction/", type=str, help='the directory of saver.')
    parser.add_argument('--nnodes', default=-1, type=int, help='node rank for distributed training.')
    parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training.')
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training.')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend.')
    args = parser.parse_args()

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.batch_size = round(args.global_batch_size/args.world_size)
    args.workers = min(max(0, args.workers), args.batch_size)
    args.ms_token_size = tuple(map(int, args.ms_patch_size.replace('-', '_').split('_')))
    args.importance = tuple(map(int, args.importance.replace('-', '_').split('_')))
    if args.stage == "substitution":
        args.checkpoint_dir = os.path.join(os.path.join(args.checkpoint_dir, "Substitution"), args.dataset_name)
        args.results_dir = os.path.join(os.path.join(args.results_dir, "Substitution"), args.dataset_name)
        args.saver_dir = os.path.join(os.path.join(args.saver_dir, "Substitution"), args.dataset_name)
        args.reconstruction_dir = os.path.join(os.path.join(args.reconstruction_dir, "Substitution"), args.dataset_name)
    elif args.stage == "adaptation":
        args.checkpoint_dir = os.path.join(os.path.join(args.checkpoint_dir, "Adaptation"), args.dataset_name)
        args.results_dir = os.path.join(os.path.join(args.results_dir, "Adaptation"), args.dataset_name)
        args.saver_dir = os.path.join(os.path.join(args.saver_dir, "Adaptation"), args.dataset_name)
        args.reconstruction_dir = os.path.join(os.path.join(args.reconstruction_dir, "Adaptation"), args.dataset_name)

    ################### very important
    if args.codebook_dim == 8:
        args.sigma = 1.0
    elif args.codebook_dim == 16:
        args.sigma = 0.8
    elif args.codebook_dim == 32:
        args.sigma = 0.5

    if args.stage == "substitution":
        if args.dataset_name == "ImageNet":
            args.epochs = 1
            args.eval_epochs = 1
        elif args.dataset_name == "Bedrooms":
            args.epochs = 5
            args.eval_epochs = 1
        else:
            args.epochs = 10
            args.eval_epochs = 5
    else:
        if args.dataset_name == "ImageNet":
            args.epochs = 4
            args.eval_epochs = 1
        elif args.dataset_name == "Bedrooms":
            args.epochs = 20
            args.eval_epochs = 5
        else:
            args.epochs = 40
            args.eval_epochs = 5
        
    args.data_pre = '{}_{}'.format(args.dataset_name, args.resolution)
    args.model_pre = 'model_{}_{}_{}'.format(args.codebook_size, args.codebook_dim, args.factor)
    if args.VQ == 'original_var' or args.VQ == 'var_no_vq':
        args.loss_pre = 'loss_empty'
    if args.stage == "substitution": 
        if args.VQ =="ema_vq" or args.VQ == "vanilla_vq":
            args.loss_pre = 'loss_empty'
        elif args.VQ == "wasserstein_vq":
            args.loss_pre = 'loss_{}'.format(args.gamma_1)
        elif args.VQ == "adversarial_vq":
            args.loss_pre = 'loss_{}'.format(args.gamma_2)

    if args.stage == "adaptation":
        args.loss_pre = 'loss_{}'.format(args.lambd)
    
    if args.VQ == "wasserstein_vq" or args.VQ == "vanilla_vq" or args.VQ == "ema_vq" or args.VQ == "adversarial_vq":
        if args.use_pq == False:
            args.training_pre = '{}_{}_{}_{}'.format(args.VQ, args.stage, args.epochs, args.use_multiscale)
        else:
            args.training_pre = '{}_{}_{}_{}_{}'.format(args.VQ, args.stage, args.epochs, args.use_multiscale, args.iterations)

    elif args.VQ == 'original_var' or args.VQ == 'var_no_vq':
        args.training_pre = '{}'.format(args.VQ)
    args.saver_name_pre = args.training_pre + '_' + args.data_pre + '_' + args.model_pre + '_' + args.loss_pre
    
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return args