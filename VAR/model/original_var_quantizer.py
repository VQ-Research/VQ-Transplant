from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

class Original_VAR(nn.Module):
    def __init__(self, vocab_size, Cvae, using_znorm, beta: float = 0.25, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4):
        super(Original_VAR).__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
        
        self.quant_resi_ratio = quant_resi
        self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
    
    def forward(self, f_BChw: torch.Tensor):
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
        return f_hat, mean_vq_loss

    ##(criterion triple analysis)
    def collect_eval_info(self, f_BChw: torch.Tensor):
        dtype = f_BChw.dtype
        if dtype != torch.float32: f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        token_cat: List[torch.Tensor] = []
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(pn, pn), mode='area').permute(0, 2, 3, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)
                
                token_cat.append(idx_N)
                # calc loss
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode='bicubic').contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
                h_BChw = self.quant_resi[si/(SN-1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw
               
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)
            quant_error = F.mse_loss(f_hat.detach()-f_BChw.detach())
            histogram = token_cat.bincount(minlength=self.vocab_size).float()
            handler = tdist.all_reduce(histogram, async_op=True)
            handler.wait()
        return f_hat, quant_error, histogram

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
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'