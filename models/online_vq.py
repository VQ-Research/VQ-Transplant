import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from torch import distributed as tdist
from models.base_quantizer import BaseQuantizer, MultiscaleBaseQuantizer

class OnlineQuantizer(BaseQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.decay = 0.99

    def forward(self, z_enc):
        B, C, H, W = z_enc.shape
        # reshape z_enc -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        z = rearrange(z_enc, 'b c h w -> b h w c')
        z_flat = z.reshape(-1, C)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flat.detach().pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.data.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flat.detach(), self.embedding.weight.data) # 'n d -> d n'

        token = torch.argmin(d, dim=1)
        z_q = self.embedding(token).view(z.shape)
        commit_loss = F.mse_loss(z_q, z.detach())

        # adjuest the shape back to match original input shape
        z_dec = z_q.permute(0, 3, 1, 2).contiguous()

        if self.args.residual:
            z_dec = z_dec.detach() + self.residual(z_dec.detach())
            residual_loss = F.mse_loss(z_dec, z_enc.detach())

        histogram = token.bincount(minlength=self.codebook_size).float()
        avg_probs = histogram/histogram.sum(0)
        self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)

        ## random feature
        sort_distance, indices = d.sort(dim=0)
        random_feat = z_flat.detach()[indices[-1,:]]
    
        decay = torch.exp(-(self.embed_prob * self.codebook_size * 10)/(1-self.decay) - 1e-3).unsqueeze(1).repeat(1, self.codebook_dim)
        self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay

        ## Criterion Triple defined in the paper
        quant_error = F.mse_loss(z_dec.detach(), z_enc.detach())
        codebook_usage_counts = (histogram > 0).float().sum()
        codebook_utilization = codebook_usage_counts.item() / self.args.codebook_size
        codebook_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.args.residual:
            loss = commit_loss + residual_loss
        else:
            loss = commit_loss
        return z_dec, loss, quant_error, codebook_utilization, codebook_perplexity

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z = z_enc.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, C)

        # distances from z to embeddings
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flat, self.embedding.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_q = self.embedding(token).view(z.shape)

        # adjuest the shape back to match original input shape
        z_dec = z_q.permute(0, 3, 1, 2).contiguous()
        if self.args.residual:
            z_dec = z_dec.detach() + self.residual(z_dec.detach())

        quant_error = F.mse_loss(z_dec.detach(), z_enc.detach())
        histogram = token.bincount(minlength=self.args.codebook_size).float()
        return z_dec, quant_error, histogram