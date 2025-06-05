import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import einsum
from einops import rearrange
from torch import distributed as tdist
from model.base_quantizer import BaseQuantizer, MultiscaleBaseQuantizer

class VanillaQuantizer(BaseQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

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
        loss = F.mse_loss(z_q, z.detach())

        # preserve gradients
        z_q = z + (z_q - z).detach()

        ## Criterion Triple defined in the paper
        quant_error = F.mse_loss(z_q.detach(), z.detach())

        histogram = token.bincount(minlength=self.args.codebook_size).float()
        codebook_usage_counts = (histogram > 0).float().sum()
        codebook_utilization = codebook_usage_counts.item() / self.args.codebook_size
            
        avg_probs = histogram/histogram.sum(0)
        codebook_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # adjuest the shape back to match original input shape
        z_dec = z_q.permute(0, 3, 1, 2).contiguous()
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

        quant_error = F.mse_loss(z_q.detach(), z.detach())
        histogram = token.bincount(minlength=self.args.codebook_size).float()

        # adjuest the shape back to match original input shape
        z_dec = z_q.permute(0, 3, 1, 2).contiguous()
        return z_dec, quant_error, histogram
    
##### multi-scale quantizer
class MultiscaleVanillaQuantizer(MultiscaleBaseQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

    def forward(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        token_cat: List[torch.Tensor] = []
        z_cat: List[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            vq_loss: torch.Tensor = 0.0
            commit_loss: torch.Tensor = 0.0
            multi_vq_loss: torch.Tensor = 0.0
            
            if self.args.fold_token == False:
                levels = len(self.args.ms_token_size)
                ms_token_size =  self.args.ms_token_size
            else:
                levels = len(self.args.fold_token_size)
                ms_token_size = self.args.fold_token_size 

            for level, pn in enumerate(ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != levels -1 or self.args.fold_token == True) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)
                z_cat.append(z_downscale.detach())
                
                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)
                
                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                embed = self.embedding(token)
                
                ## the multi-scale vector quantization loss
                commit_loss += F.mse_loss(embed, z_downscale.detach())

                token_cat.append(token)                  
                token_Bhw = token.view(B, pn, pn)

                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != levels -1 or self.args.fold_token == True) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
                z_upscale = self.phi[level/(levels-1)](z_upscale)

                z_dec = z_dec + z_upscale
                z_rest = z_rest - z_upscale

                multi_vq_loss += F.mse_loss(z_dec, z_no_grad)
            
            ## residual quantization loss
            vq_loss = F.mse_loss(z_dec, z_enc.data) 
            commit_loss *= 1. / levels
            multi_vq_loss *= 1. / levels

            token_cat = torch.cat(token_cat, 0)
            z_cat = torch.cat(z_cat, 0)

            ## Criterion Triple defined in the paper
            z_dec = (z_dec - z_enc).detach().add_(z_enc)
            quant_error = F.mse_loss(z_dec.detach(), z_enc.detach())

            histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            handler = tdist.all_reduce(histogram, async_op=True)
            handler.wait()
                
            codebook_usage_counts = (histogram > 0).float().sum()
            codebook_utilization = codebook_usage_counts.item() / self.args.codebook_size
            
            avg_probs = histogram/histogram.sum(0)
            codebook_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            #loss = vq_loss + multi_vq_loss + commit_loss 
            loss = multi_vq_loss 
        return z_dec, loss, quant_error, codebook_utilization, codebook_perplexity

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z_no_grad = z_enc.detach()
        z_rest = z_no_grad.clone()
        z_dec = torch.zeros_like(z_rest)

        token_cat: List[torch.Tensor] = []
        z_cat: List[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            if self.args.fold_token == False:
                levels = len(self.args.ms_token_size)
                ms_token_size =  self.args.ms_token_size
            else:
                levels = len(self.args.fold_token_size)
                ms_token_size = self.args.fold_token_size

            for level, pn in enumerate(ms_token_size):
                z_downscale = F.interpolate(z_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (level != levels -1 or self.args.fold_token == True) else z_rest.permute(0, 2, 3, 1).reshape(-1, C)
                z_cat.append(z_downscale)

                ## distance [B*ph*pw, vocab_size]
                distance = torch.sum(z_downscale.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                distance.addmm_(z_downscale.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)

                ## token [B*ph*pw]
                token = torch.argmin(distance, dim=1)
                token_cat.append(token)

                token_Bhw = token.view(B, pn, pn)
                z_upscale = F.interpolate(self.embedding(token_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (level != levels -1 or self.args.fold_token == True) else self.embedding(token_Bhw).permute(0, 3, 1, 2).contiguous()
                z_upscale = self.phi[level/(levels-1)](z_upscale)

                z_dec.add_(z_upscale)
                z_rest.sub_(z_upscale)

            token_cat = torch.cat(token_cat, 0)
            z_cat = torch.cat(z_cat, 0)

            quant_error = F.mse_loss(z_dec.detach(), z_enc.detach())
            histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            handler = tdist.all_reduce(histogram, async_op=True)
            handler.wait()
            
        return z_dec, quant_error, histogram
