import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import einsum
from einops import rearrange
from torch import distributed as tdist
from models.base_quantizer import VectorQuantizer, MultiscaleVectorQuantizer

#### not the multi-scale quantizer and no residual quantization
class WassersteinVectorQuantizer(VectorQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def calc_wasserstein_loss(self, z=None):
        if z == None:
            z = self.queue.obtain_feature_from_queue()

        N = z.size(0)
        D = z.size(1)

        std = z.std(dim=0).max().detach()
        z = z / (std + 1e-8)
        z_mean = z.mean(0).detach()
        z_covariance = torch.cov(z.t()) + 1e-8 * torch.eye(D, device=z.device) 
        z_covariance = z_covariance.detach()

        ### compute the mean and covariance of codebook vectors
        c = self.embedding.weight /  (std + 1e-8)
        c_mean = c.mean(0)
        c_covariance = torch.cov(c.t()) + 1e-8 * torch.eye(D, device=z.device)
        
        ### calculation of part1
        part_mean =  torch.sum(torch.multiply(z_mean - c_mean, z_mean - c_mean))

        ### 1/2 covariance
        S1, Q1 = torch.linalg.eigh(z_covariance)
        sqrt_S1 = torch.diag(torch.sqrt(F.relu(S1)+ 1e-8))
        temp = torch.mm(Q1, sqrt_S1)
        temp = torch.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)
        z_sqrt_covariance = torch.mm(temp, Q1.T)
        z_sqrt_covariance = torch.nan_to_num(z_sqrt_covariance, nan=0.0, posinf=0.0, neginf=0.0)
        z_sqrt_covariance = z_sqrt_covariance.detach()
        
        ### 1/2 covariance
        temp = torch.mm(z_sqrt_covariance, c_covariance)
        temp = torch.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)

        covariance = torch.mm(temp, z_sqrt_covariance)
        covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)

        S2, Q2 = torch.linalg.eigh(covariance)
        sqrt_S2 = torch.sqrt(F.relu(S2)+ 1e-8)

        #############calculation of part2
        part_covariance = F.relu(torch.trace(z_covariance.detach() + c_covariance) - 2.0 * sqrt_S2.sum())
        wasserstein_loss = torch.sqrt(part_mean + part_covariance + 1e-10)
        return wasserstein_loss

    def forward(self, z_enc):
        # reshape z_enc -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  

        with torch.no_grad():
            self.queue.dequeue_and_enqueue(z_flat.detach())
        wasserstein_loss = self.calc_wasserstein_loss()

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flat.detach().pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.data.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flat.detach(), self.embedding.weight.data) # 'n d -> d n'
                
        token = torch.argmin(d, dim=1)
        z_dec = self.embedding(token).view(z.shape).permute(0, 3, 1, 2).contiguous()
        commit_loss = self.beta * F.mse_loss(z_dec.detach(), z_enc) +  self.alpha * F.mse_loss(z_dec, z_enc.detach())

        histogram = token.bincount(minlength=self.args.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()

        codebook_usage_counts = (histogram > 0).float().sum()
        utilization = codebook_usage_counts.item() / self.args.codebook_size
            
        avg_probs = histogram/histogram.sum(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        z_dec = z_enc + (z_dec - z_enc).detach()
        loss = commit_loss + self.args.gamma * wasserstein_loss
        return z_dec, loss, utilization, perplexity

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  

        # distances from z to embeddings
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flat, self.embedding.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_dec = self.embedding(token).view(z.shape).permute(0, 3, 1, 2).contiguous()
        histogram = token.bincount(minlength=self.args.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()
        return z_dec, histogram
    
##### multi-scale quantizer
class WassersteinVARQuantizer(MultiscaleVectorQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def calc_wasserstein_loss(self, z=None):
        if z == None:
            z = self.queue.obtain_feature_from_queue()

        N = z.size(0)
        D = z.size(1)

        std = z.std(dim=0).max().detach()
        z = z / (std + 1e-8)
        z_mean = z.mean(0).detach()
        z_covariance = torch.cov(z.t()) + 1e-8 * torch.eye(D, device=z.device) 
        z_covariance = z_covariance.detach()

        ### compute the mean and covariance of codebook vectors
        c = self.embedding.weight /  (std + 1e-8)
        c_mean = c.mean(0)
        c_covariance = torch.cov(c.t()) + 1e-8 * torch.eye(D, device=z.device)
        
        ### calculation of part1
        part_mean =  torch.sum(torch.multiply(z_mean - c_mean, z_mean - c_mean))

        ### 1/2 covariance
        S1, Q1 = torch.linalg.eigh(z_covariance)
        sqrt_S1 = torch.diag(torch.sqrt(F.relu(S1)+ 1e-8))
        temp = torch.mm(Q1, sqrt_S1)
        temp = torch.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)
        z_sqrt_covariance = torch.mm(temp, Q1.T)
        z_sqrt_covariance = torch.nan_to_num(z_sqrt_covariance, nan=0.0, posinf=0.0, neginf=0.0)
        z_sqrt_covariance = z_sqrt_covariance.detach()
        
        ### 1/2 covariance
        temp = torch.mm(z_sqrt_covariance, c_covariance)
        temp = torch.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)

        covariance = torch.mm(temp, z_sqrt_covariance)
        covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)

        S2, Q2 = torch.linalg.eigh(covariance)
        sqrt_S2 = torch.sqrt(F.relu(S2)+ 1e-8)

        #############calculation of part2
        part_covariance = F.relu(torch.trace(z_covariance.detach() + c_covariance) - 2.0 * sqrt_S2.sum())
        wasserstein_loss = torch.sqrt(part_mean + part_covariance + 1e-10)
        return wasserstein_loss

    def forward(self, z_enc):
        ### projector in layer
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  
        z_flat = self.projector_in(z_flat)
        z_pre = z_flat.view(z.shape).permute(0, 3, 1, 2).contiguous()

        z_no_grad = z_pre.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        token_cat: List[torch.Tensor] = []
        z_cat: List[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            multi_vq_loss: torch.Tensor = 0.0
            wasserstein_loss: torch.Tensor = 0.0
            vq_loss: torch.Tensor = 0.0
            levels = len(self.args.ms_token_size)
            ms_token_size =  self.args.ms_token_size

            for level, pn in enumerate(ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != levels -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)
                z_cat.append(z_downscale.detach())
                
                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)
                
                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                embed = self.embedding(token)

                token_cat.append(token)                  
                token_Bhw = token.view(B, pn, pn)

                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != levels -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
                z_upscale = self.phi[level/(levels-1)](z_upscale)

                z_dec = z_dec + z_upscale
                z_rest = z_rest - z_upscale

                multi_vq_loss += self.alpha * F.mse_loss(z_dec, z_no_grad) + self.beta * F.mse_loss(z_dec.detach(), z_pre)

            multi_vq_loss *= 1. / len(ms_token_size)
            token_cat = torch.cat(token_cat, 0)
            z_cat = torch.cat(z_cat, 0)
            with torch.no_grad():
                self.queue.dequeue_and_enqueue(z_cat.detach())

            ### projector out layer
            z_dec = z_pre + (z_dec-z_pre).detach()
            zq = rearrange(z_dec, 'b c h w -> b h w c') 
            zq_flat = zq.reshape(-1, C).contiguous()  
            zq_flat = self.projector_out(zq_flat)
            z_dec = zq_flat.view(zq.shape).permute(0, 3, 1, 2).contiguous()
            vq_loss = F.mse_loss(z_dec, z_enc.detach()) 

            ## Criterion Triple defined in the paper
            quant_error = F.mse_loss(z_dec.detach(), z_enc.detach())

            histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            handler = tdist.all_reduce(histogram, async_op=True)
            handler.wait()
                
            codebook_usage_counts = (histogram > 0).float().sum()
            codebook_utilization = codebook_usage_counts.item() / self.args.codebook_size
            
            avg_probs = histogram/histogram.sum(0)
            codebook_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            ### compute wasserstein distance
            wasserstein_loss = self.calc_wasserstein_loss()
            loss =  multi_vq_loss + self.args.gamma * wasserstein_loss + vq_loss
            
        return z_dec, loss, quant_error, codebook_utilization, codebook_perplexity

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  
        z_flat = self.projector_in(z_flat)
        z_pre = z_flat.view(z.shape).permute(0, 3, 1, 2).contiguous()

        z_no_grad = z_pre.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        token_cat: List[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            levels = len(self.args.ms_token_size)
            ms_token_size =  self.args.ms_token_size

            for level, pn in enumerate(ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != levels -1) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)

                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)

                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                token_cat.append(token)

                token_Bhw = token.view(B, pn, pn)
                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != levels -1) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
                z_upscale = self.phi[level/(levels-1)](z_upscale)

                z_dec.add_(z_upscale)
                z_rest.sub_(z_upscale)

            ### projector out layer
            zq = rearrange(z_dec, 'b c h w -> b h w c') 
            zq_flat = zq.reshape(-1, C).contiguous()  
            zq_flat = self.projector_out(zq_flat)
            z_dec = zq_flat.view(zq.shape).permute(0, 3, 1, 2).contiguous()

            token_cat = torch.cat(token_cat, 0)
            quant_error = F.mse_loss(z_dec.detach(), z_enc.detach())
            histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            handler = tdist.all_reduce(histogram, async_op=True)
            handler.wait()
        return z_dec, quant_error, histogram
