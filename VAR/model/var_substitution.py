import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from model.encoder_decoder import Encoder, Decoder
from original_var_quantizer import Original_VAR
from utils.util import Pack

class VAR_Substitution(nn.Module):
    def __init__(self, args):
        super(VAR_Substitution, self).__init__()
        self.args = args

        ddconfig = dict(
            dropout=0, ch=160, z_channels=32,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   
            using_sa=True, using_mid_sa=True,                          
        )
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.quant_conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.post_quant_conv = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)

        if self.args.VQ == "original_var":
            self.quantizer = Original_VAR(vocab_size=4096, Cvae=32, using_znorm=False, beta=0.25, v_patch_nums=self.args.ms_token_size, quant_resi=0.5, share_quant_resi=4)

    def forward(self, x):
        pass

    def collect_eval_info(self, x):
        ## encoder
        z = self.encoder(x)
        z = self.pre_quant_proj(z)

        if self.args.VQ == "var_no_vq":
            z_q = z
        else:
            z_q, quant_error, histogram = self.quantizer.collect_eval_info(z)

        z = self.post_quant_proj(z_q)
        x_rec = self.decoder(z).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        
        if self.args.VQ == "var_no_vq":
            return x_rec, rec_loss
        else:
            return x_rec, rec_loss, quant_error, histogram  






