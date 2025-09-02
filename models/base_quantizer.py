### Adopted code from the https://github.com/FoundationVision/VAR/blob/main/models/quant.py
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from torch import distributed as tdist

class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)

class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

class EmbeddingEMA(nn.Module):
    def __init__(self, codebook_size, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps

        weight = torch.randn(codebook_size, codebook_dim)/self.codebook_size
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(codebook_size), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized) 

class Queue(nn.Module):
    def __init__(self, args):
        super(Queue, self).__init__()
        self.args = args
        self.codebook_dim = args.codebook_dim
        self.codebook_size = args.codebook_size
        if args.use_multiscale == False:
            self.queue_size = 32768
        else:
            self.queue_size = 87040
        self.register_buffer("queue", torch.randn(self.queue_size, self.codebook_dim)/self.args.codebook_size)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, key):
        batch_size = key.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
        self.queue[ptr:ptr + batch_size, :] = key
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr  

    @torch.no_grad()
    def obtain_feature_from_queue(self):
        return self.queue.detach().clone()

    @torch.no_grad()
    def obatin_latest_feature_from_queue(self):
        assert self.codebook_size < self.queue_size
        ptr = int(self.queue_ptr)
        if ptr >= self.codebook_size:
            return self.queue[(ptr-self.codebook_size):ptr, :].detach().clone()
        else:
            part1 = self.queue[0:ptr, :].detach().clone()
            part2 = self.queue[(self.queue_size-(self.codebook_size-ptr)):self.queue_size,:].detach().clone()
            return torch.cat((part1, part2), 0)

class VectorQuantizer(nn.Module):
    def __init__(self, args):
        super(VectorQuantizer, self).__init__()
        self.args = args
        self.codebook_size = args.codebook_size
        self.codebook_dim = args.codebook_dim
        self.alpha = args.alpha
        self.beta = args.beta
        self.decay = 0.8
        if args.VQ == "wasserstein_vq" or args.VQ == "vanilla_vq" or args.VQ == "mmd_vq":
            self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
            self.embedding.weight.data.uniform_(-1.0 /self.codebook_size, 1.0/self.codebook_size)
            self.embedding.weight.requires_grad = True
        elif args.VQ == "online_vq":
            self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
            self.embedding.weight.data.uniform_(-1.0 /self.codebook_size, 1.0/self.codebook_size)
            self.embedding.weight.requires_grad = True
            self.register_buffer("embed_prob", torch.zeros(self.codebook_size))
        elif args.VQ == "ema_vq":
            self.embedding = EmbeddingEMA(self.codebook_size, self.codebook_dim, self.decay, eps=1e-5)
            
        if args.VQ == "wasserstein_vq":
            self.queue = Queue(args)

class ProductQuantizer(nn.Module):
    def __init__(self, args):
        super(ProductQuantizer, self).__init__()
        self.args = args
        self.codebook_size = args.codebook_size
        self.codebook_dim = args.codebook_dim
        self.alpha = args.alpha
        self.beta = args.beta
        self.decay = 0.8
        if args.VQ == "wasserstein_vq" or args.VQ == "vanilla_vq" or args.VQ == "mmd_vq":
            self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
            self.embedding.weight.data.uniform_(-1.0 /self.codebook_size, 1.0/self.codebook_size)
            self.embedding.weight.requires_grad = True
        elif args.VQ == "online_vq":
            self.embedding = nn.Embedding(self.codebook_size, self.codebook_dim)
            self.embedding.weight.data.uniform_(-1.0 /self.codebook_size, 1.0/self.codebook_size)
            self.embedding.weight.requires_grad = True
            self.register_buffer("embed_prob", torch.zeros(self.codebook_size))
        elif args.VQ == "ema_vq":
            self.embedding = EmbeddingEMA(self.codebook_size, self.codebook_dim, self.decay, eps=1e-5)
            
        if args.VQ == "wasserstein_vq":
            self.queue = Queue(args)

        