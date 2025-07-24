import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from torch import distributed as tdist
from models.probability_quantization import quantize_batch_probs

class Categorical_Quantizer(nn.Module):
    def __init__(self, args):
        super(Categorical_Quantizer, self).__init__()
        self.args = args
        self.codebook_size = args.codebook_size
        self.codebook_dim = args.codebook_dim
        self.projector_in =  nn.Sequential(
                nn.Linear(self.codebook_dim, 8*self.codebook_size),
                nn.BatchNorm1d(8*self.codebook_size),
                nn.SiLU(),
                nn.Linear(8*self.codebook_size, 8*self.codebook_size),
                nn.BatchNorm1d(8*self.codebook_size),
                nn.SiLU(),
                nn.Linear(8*self.codebook_size, 8*self.codebook_size),
                nn.BatchNorm1d(8*self.codebook_size),
                nn.SiLU(),
                nn.Linear(8*self.codebook_size, self.codebook_size),
            )

        self.projector_out =  nn.Sequential(
                nn.Linear(self.codebook_size, 8*self.codebook_size),
                nn.BatchNorm1d(8*self.codebook_size),
                nn.SiLU(),
                nn.Linear(8*self.codebook_size, 8*self.codebook_size),
                nn.BatchNorm1d(8*self.codebook_size),
                nn.SiLU(),
                nn.Linear(8*self.codebook_size, 8*self.codebook_size),
                nn.BatchNorm1d(8*self.codebook_size),
                nn.SiLU(),
                nn.Linear(8*self.codebook_size, self.codebook_dim),
            )

    def forward(self, z_enc, cur_iter):
        B, C, H, W = z_enc.shape
        if self.args.frozen_encoder:
            z_enc = z_enc.detach()

        z = z_enc.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        z = self.projector_in(z)
        probs = F.softmax(z, dim=1)
        entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), 1).mean()

        avg_probs = probs.mean(0)
        avg_entropy_loss = math.log(self.args.codebook_size) + torch.sum(avg_probs * torch.log(avg_probs + 1e-8))
        
        ### prob quantization
        quant_probs = quantize_batch_probs(probs.detach(), self.args.L)
        prob_commit_loss = (probs - quant_probs.detach()).pow(2).sum(1).mean()

        ### STE operator
        quant_probs = probs + (quant_probs - probs).detach()
        z_quant = self.projector_out(quant_probs)
        z_dec = z_quant.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        tau_0 = (self.args.L/64) * 10
        
        ### overall loss
        categorical_loss = tau_0 * prob_commit_loss + self.args.gamma * entropy_loss + self.args.lambd * avg_entropy_loss
        return z_dec, categorical_loss, prob_commit_loss, entropy_loss, avg_entropy_loss

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z = z_enc.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        z = self.projector_in(z)
        
        ### calc entropy loss and average entropy loss
        probs = F.softmax(z, dim=1)
        entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), 1).mean()

        avg_probs = probs.mean(0)
        avg_entropy_loss = math.log(self.args.codebook_size) + torch.sum(avg_probs * torch.log(avg_probs + 1e-8))

        ### prob quantization
        quant_probs = quantize_batch_probs(probs.detach(), self.args.L)
        prob_commit_loss = (probs - quant_probs.detach()).pow(2).sum(1).mean()

        z_quant = self.projector_out(quant_probs)
        z_dec = z_quant.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return z_dec, prob_commit_loss, entropy_loss, avg_entropy_loss

    '''
    def latent_tokenization(self, z_enc):
        B, C, H, W = z_enc.shape
        z = z_enc.permute(0, 2, 3, 1).reshape(-1, C)

        distance = torch.sum(z.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
        distance.addmm_(z.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)
        distance = distance.div(math.sqrt(self.args.codebook_dim) * self.args.temperature)

        probs = F.softmax(-1.0 * distance, dim=1)
        quant_probs = quantize_batch_probs(probs.detach(), self.args.L)
        token = quant_probs.reshape(B, H, W, -1).contiguous()

        embeddings = torch.mm(quant_probs.detach(), self.embedding.weight.data).reshape(B, H, W, C).contiguous()
        return tokens, embeddings
    '''
