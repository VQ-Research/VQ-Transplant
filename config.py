import json
import os
import random
import re
import subprocess
import sys
import time
import numpy as np
import torch
from collections import OrderedDict
from typing import Optional, Union
import argparse
import torch.distributed as dist
from utils.misc import str2bool
import ruamel.yaml as yaml

def parse_arg():
    parser = argparse.ArgumentParser(description='VQ-Transplant based on SANA Tokenizer.') 

    ### Dataset and Dataloader Configuration
    parser.add_argument('--dataset_dir', default="/projects/yuanai/data/", type=str, help='the directory of dataset') 
    parser.add_argument('--dataset_name', default='ImageNet', help='the name of dataset', choices=['ImageNet'])
    parser.add_argument('--global_batch_size', type=int, default=128, help="the size of batch samples")
    parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loader workers')
    parser.add_argument('--resolution', type=int, choices=[256], default=256, help='resolution of train and test')
    parser.add_argument('--channels', default=3, type=int, metavar='N', help='the channels of images')
    
    ### Model Configuration
    parser.add_argument('--ms_patch_size', default="1_2_3_4_5_6_8", type=str, help='multi-scale patch size.')
    parser.add_argument('--codebook_size', default=256, type=int, help='the size of codebook.', choices=[128, 256, 512, 1024])
    parser.add_argument('--codebook_dim', default=32, type=int, help='the dimension of codebook vectors.')
    parser.add_argument('--L', default=64, type=int, help='Quantize probability distributions into fractions with a denominator of L.', choices=[8, 16, 32, 64])

    ### Loss Configuration
    parser.add_argument('--gamma', type=float, default=0.05, help="transplant stage: the hyperparameter of entropy_loss.")
    parser.add_argument('--lambd', type=float, default=0.05, help="transplant stage: the hyperparameter of avg_entropy_loss.")
    parser.add_argument("--temperature", type=float, default=1.0, help="transplant stage: temperature for temperature.")
    parser.add_argument('--disc_weight', type=float, default=0.2, help="refinement stage: discriminator loss weight for gan training")
    parser.add_argument('--lecam_loss_weight', type=float, default=0.001, help='refinement stage: lecam_loss_weight')
    parser.add_argument('--disc_cr_loss_weight', type=float, default=4.0, help='refinement stage: disc_cr_loss_weight')

    ### Training Configuration
    parser.add_argument('--model_name', default='CategoricalTokenizer', help='the name of models.', choices=['CategoricalTokenizer'])
    parser.add_argument('--frozen_encoder', action='store_true', help='during the transplant phase, whether frozen the encoder.')
    parser.add_argument('--transplant_epochs', type=int, default=10, help="training epochs, 10 epochs for transplant stage.")
    parser.add_argument('--refinement_epochs', type=int, default=10, help="training epochs, 10 epochs for refinement stage.")
    parser.add_argument('--eval_epochs', type=int, default=1, help="epochs for each eval, 1 epochs for ImageNet.")
    parser.add_argument('--lr_transplant', default=1e-5, type=float, metavar='LR', help='initial learning rate for transplant stage.')
    parser.add_argument('--lr_refinement', default=1e-6, type=float, metavar='LR', help='initial learning rate for refinement stage.')
    parser.add_argument('--dropout', help='dropout for the model', type=float, default=0.0)
    parser.add_argument('--seed', help='random seed', type=int, default=3407)
    parser.add_argument('--weight_decay', help='weight decay for optimizer', type=float, default=0.0001)
    parser.add_argument('--stage', default='transplant', help='there are two stages: transplant and refinement.', choices=['transplant', 'refinement'])

    ##BC:/projects/yuanai/projects/Discrete-Categorical-Tokenizer
    parser.add_argument('--checkpoint_dir', default="/projects/yuanai/projects/Discrete-Categorical-Tokenizer/checkpoint/", type=str, help='the directory of checkpoint.')
    parser.add_argument('--results_dir', default="/projects/yuanai/projects/Discrete-Categorical-Tokenizer/results/", type=str, help='the directory of results.')
    parser.add_argument('--saver_dir', default="/projects/yuanai/projects/Discrete-Categorical-Tokenizer/saver/", type=str, help='the directory of saver.')
    parser.add_argument('--reconstruction_dir', default="/projects/yuanai/projects/Discrete-Categorical-Tokenizer/reconstruction/", type=str, help='the directory of saver.')
    parser.add_argument('--yaml_dir', default="/projects/yuanai/projects/Discrete-Categorical-Tokenizer/yaml/", type=str, help='the directory of saver.')
    parser.add_argument('--pretrained_tokenizer', default="/projects/yuanai/projects/Discrete-Categorical-Tokenizer/pretrained_tokenizer/dc-ae-f32c32-sana-1.0.safetensors", type=str, help='the directory of sana checkpoint.')
    parser.add_argument('--checkpoint_name', default="", type=str, help='the directory of saved checkpoint name for the refinement stage.')
    parser.add_argument('--nnodes', default=-1, type=int, help='node rank for distributed training.')
    parser.add_argument('--node_rank', default=-1, type=int, help='node rank for distributed training.')
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training.')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend.')
    args = parser.parse_args()

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.batch_size = round(args.global_batch_size/args.world_size)
    args.workers = min(max(0, args.workers), args.batch_size)

    if args.stage == "transplant":
        args.checkpoint_dir = os.path.join(os.path.join(args.checkpoint_dir, "Transplant"), args.dataset_name)
        args.results_dir = os.path.join(os.path.join(args.results_dir, "Transplant"), args.dataset_name)
        args.saver_dir = os.path.join(os.path.join(args.saver_dir, "Transplant"), args.dataset_name)
        args.reconstruction_dir = os.path.join(os.path.join(args.reconstruction_dir, "Transplant"), args.dataset_name)
        args.yaml_dir = os.path.join(os.path.join(args.yaml_dir, "Transplant"), args.dataset_name) 
    elif args.stage == "refinement":
        args.checkpoint_dir = os.path.join(os.path.join(args.checkpoint_dir, "Refinement"), args.dataset_name)
        args.results_dir = os.path.join(os.path.join(args.results_dir, "Refinement"), args.dataset_name)
        args.saver_dir = os.path.join(os.path.join(args.saver_dir, "Refinement"), args.dataset_name)
        args.reconstruction_dir = os.path.join(os.path.join(args.reconstruction_dir, "Refinement"), args.dataset_name)
        args.yaml_dir = os.path.join(os.path.join(args.yaml_dir, "Refinement"), args.dataset_name) 
        
    args.data_pre = '{}'.format(args.dataset_name)
    args.model_pre = 'model_{}_{}'.format(args.codebook_size, args.L)
    args.loss_pre = 'loss_{}_{}_{}'.format(args.gamma, args.lambd, args.disc_weight)
    args.training_pre = '{}_{}'.format(args.model_name, args.frozen_encoder)
    args.saver_name_pre = args.training_pre + '_' + args.data_pre + '_' + args.model_pre + '_' + args.loss_pre

    dict_args = vars(args)
    config_name = args.saver_name_pre+'.yaml'
    with open(os.path.join(args.yaml_dir, config_name), 'w', encoding='utf-8') as f:
        file_yaml = yaml.YAML()
        file_yaml.dump(dict_args, f)
    
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