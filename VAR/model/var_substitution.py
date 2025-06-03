import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from model.encoder_decoder import Encoder, Decoder
from model.original_var_quantizer import Original_VAR
from model.wasserstein_vq import MultiscaleWassersteinQuantizer, WassersteinQuantizer
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

        ####loading pretrained visual tokenizer
        pretrain_dict = torch.load(args.pretrained_tokenizer, map_location='cpu')
        encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
        decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
        quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quant_conv.')}
        post_quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('post_quant_conv.')}

        encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
        decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
        quant_conv_dict = {k.replace('quant_conv.', '', 1): v for k, v in quant_conv_dict.items()}
        post_quant_conv_dict = {k.replace('post_quant_conv.', '', 1): v for k, v in post_quant_conv_dict.items()}

        self.encoder.load_state_dict(encoder_dict, strict=True)
        self.decoder.load_state_dict(decoder_dict, strict=True)
        self.quant_conv.load_state_dict(quant_conv_dict, strict=True)
        self.post_quant_conv.load_state_dict(post_quant_conv_dict, strict=True)
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        for param in self.quant_conv.parameters():
            param.requires_grad = False
        for param in self.post_quant_conv.parameters():
            param.requires_grad = False  
        
        if self.args.VQ == "original_var":
            quantizer_dict = {k: v for k, v in pretrain_dict.items() if "quantize.embedding." in k or "quantize.quant_resi." in k}
            quantizer_dict = {k.replace('quantize.', '', 1): v for k, v in quantizer_dict.items()}
            self.quantizer = Original_VAR(vocab_size=4096, Cvae=32, using_znorm=False, beta=0.25, v_patch_nums=self.args.ms_token_size, quant_resi=0.5, share_quant_resi=4)
            self.quantizer.load_state_dict(quantizer_dict, strict=True)
        if self.args.VQ == "wasserstein_vq":
            if self.args.use_multiscale == True:
                self.quantizer = MultiscaleWassersteinQuantizer(args)
                if self.args.use_pq == True:
                    self.quantizer2 = MultiscaleWassersteinQuantizer(args)
            else:
                self.quantizer = WassersteinQuantizer(args)
                if self.args.use_pq == True: 
                    self.quantizer2 = WassersteinQuantizer(args)

    def forward(self, x):
        ## encoder
        with torch.no_grad():
            z = self.encoder(x)
            z = self.quant_conv(z)
            if self.args.use_pq == True:
                z_1, z_2 = torch.chunk(z, 2, dim=1) ##[B, 16, 16, 16]
            else:
                z_1 = z ##[B, 32, 16, 16]

        ## quantizer
        if self.args.VQ == "wasserstein_vq": 
            z_q_1, vq_loss, wasserstein_loss, quant_error, codebook_utilization, codebook_perplexity = self.quantizer(z_1)
            if self.args.use_pq == True:
                z_q_2, vq_loss_2, wasserstein_loss_2, quant_error_2, codebook_utilization_2, codebook_perplexity_2 = self.quantizer2(z_2)
                z_q = torch.cat((z_q_1, z_q_2), dim=1)
                vq_loss = (vq_loss+vq_loss_2) * 0.5
            else:
                z_q = z_q_1
                
        quant_error = F.mse_loss(z_q.detach(), z.detach())
        ## decoder
        with torch.no_grad():
            z = self.post_quant_conv(z_q)
            x_rec = self.decoder(z)
            rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())

        info_pack = Pack(vq_loss=vq_loss, rec_loss=rec_loss, wasserstein_loss=wasserstein_loss, quant_error=quant_error, codebook_utilization=codebook_utilization, codebook_perplexity=codebook_perplexity)
        return x_rec, vq_loss, info_pack

    def collect_eval_info(self, x):
        ## encoder
        z = self.encoder(x)
        z = self.quant_conv(z)
        if self.args.use_pq == True:
            z_1, z_2 = torch.chunk(z, 2, dim=1) ##[B, 16, 16, 16]
        else:
            z_1 = z ##[B, 32, 16, 16]

        if self.args.VQ == "var_no_vq":
            z_q = z
        elif self.args.VQ == "wasserstein_vq":
            z_q_1, wasserstein_loss, quant_error, histogram = self.quantizer.collect_eval_info(z_1)
            if self.args.use_pq == True:
                z_q_2, wasserstein_loss_2, quant_error_2, histogram_2 = self.quantizer.collect_eval_info(z_2)
                z_q = torch.cat((z_q_1, z_q_2), dim=1)
            else:
                z_q = z_q_1
        else:
            z_q_1, quant_error, histogram = self.quantizer.collect_eval_info(z_1)
            if self.args.use_pq == True:
                z_q_2, quant_error_2, histogram_2 = self.quantizer.collect_eval_info(z_2)
                z_q = torch.cat((z_q_1, z_q_2), dim=1)
            else:
                z_q = z_q_1

        quant_error = F.mse_loss(z_q.detach(), z.detach())
        z = self.post_quant_conv(z_q)
        x_rec = self.decoder(z).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        
        if self.args.VQ == "var_no_vq":
            return x_rec, rec_loss
        else:
            return x_rec, rec_loss, quant_error, histogram

