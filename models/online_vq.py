import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from torch import distributed as tdist
from models.base_quantizer import VectorQuantizer, MultiscaleVectorQuantizer

class OnlineVectorQuantizer(VectorQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.decay = 0.99

    def forward(self, z_enc):
        B, C, H, W = z_enc.shape
        # reshape z_enc -> (batch, height, width, channel) and flatten
        #z, 'b c h w -> b h w c'
        z = rearrange(z_enc, 'b c h w -> b h w c') 
        z_flat = z.reshape(-1, C).contiguous()  
        z_flat = self.projector_in(z_flat) 
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = z_flat.detach().pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.data.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flat.detach(), self.embedding.weight.data) # 'n d -> d n'

        token = torch.argmin(d, dim=1)
        z_q = self.embedding(token)
        commit_loss = self.beta * F.mse_loss(z_q.detach(), z_flat) + self.alpha * F.mse_loss(z_q, z_flat.detach())

        ## adjuest the shape back to match original input shape
        z_q = z_flat + (z_q - z_flat).detach()
        z_q = self.projector_out(z_q)
        z_dec = z_q.view(z.shape).permute(0, 3, 1, 2).contiguous()
        vq_loss = F.mse_loss(z_dec, z_enc.detach())

        histogram = token.bincount(minlength=self.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()

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

        loss = vq_loss + commit_loss
        return z_dec, loss, quant_error, codebook_utilization, codebook_perplexity

    def collect_eval_info(self, z_enc):
        B, C, H, W = z_enc.shape
        z = z_enc.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, C).contiguous()
        z_flat = self.projector_in(z_flat) 

        # distances from z to embeddings
        d = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.data**2, dim=1) - 2 * \
            torch.matmul(z_flat, self.embedding.weight.data.t())

        token = torch.argmin(d, dim=1)
        z_q = self.embedding(token)
        z_q = self.projector_out(z_q)

        # adjuest the shape back to match original input shape
        z_dec = z_q.view(z.shape).permute(0, 3, 1, 2).contiguous()

        quant_error = F.mse_loss(z_dec.detach(), z_enc.detach())
        histogram = token.bincount(minlength=self.args.codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()
        return z_dec, quant_error, histogram

##### multi-scale quantizer
class OnlineVARQuantizer(MultiscaleVectorQuantizer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.decay = 0.99

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
            
            ## residual quantization loss
            multi_vq_loss *= 1. / len(ms_token_size)
            token_cat = torch.cat(token_cat, 0)
            z_cat = torch.cat(z_cat, 0)

            ### projector out layer
            z_dec = z_pre + (z_dec-z_pre).detach()
            zq = rearrange(z_dec, 'b c h w -> b h w c') 
            zq_flat = zq.reshape(-1, C).contiguous()  
            zq_flat = self.projector_out(zq_flat)
            z_dec = zq_flat.view(zq.shape).permute(0, 3, 1, 2).contiguous()
            vq_loss = F.mse_loss(z_dec, z_enc.detach()) 

            histogram = token_cat.bincount(minlength=self.args.codebook_size).float()
            handler = tdist.all_reduce(histogram, async_op=True)
            handler.wait()
            avg_probs = histogram/histogram.sum(0)
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)

            ## random feature
            d = torch.sum(z_cat.detach().square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
            d.addmm_(z_cat.detach(), self.embedding.weight.data.T, alpha=-2, beta=1)
            sort_distance, indices = d.sort(dim=0)
            random_feat = z_cat.detach()[indices[-1,:]]
    
            decay = torch.exp(-(self.embed_prob * self.codebook_size * 10)/(1-self.decay) - 1e-3).unsqueeze(1).repeat(1, self.codebook_dim)
            self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay

            ## Criterion Triple defined in the paper
            quant_error = F.mse_loss(z_dec.detach(), z_enc.detach())
            codebook_usage_counts = (histogram > 0).float().sum()
            codebook_utilization = codebook_usage_counts.item() / self.args.codebook_size
            codebook_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

            loss = multi_vq_loss + vq_loss
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