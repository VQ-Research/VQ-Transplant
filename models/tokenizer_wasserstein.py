import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from models.categorical_quantizer import Categorical_Quantizer
from models.encoder_decoder import Encoder, Decoder, EncoderConfig, DecoderConfig
from utils.util import Pack
from safetensors.torch import load_file

class VQModel(nn.Module):
    def __init__(self, args):
        super(VQModel, self).__init__()
        self.args = args
        enc_config = EncoderConfig
        dec_config = DecoderConfig
        self.encoder = Encoder(EncoderConfig)
        self.decoder = Decoder(DecoderConfig)
        self.quantizer = Categorical_Quantizer(args)
        if args.stage == "transplant":
            self.encoder2 = Encoder(EncoderConfig)
    
        if args.stage == "transplant":
            pretrain_dict = load_file(args.pretrained_tokenizer)
            encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
            decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.encoder2.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.quantizer.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.encoder2.parameters():
                param.requires_grad = False
            self.encoder2.eval()
            self.decoder.eval()

        if args.stage == "refinement":
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quantizer.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = True
            self.encoder.eval()
            self.quantizer.eval()

    def transplant(self, x, cur_iter):
        assert self.args.stage == "transplant"
        with torch.no_grad():
            z_obj = self.encoder2(x) 
        z = self.encoder(x)
        z_q, categorical_loss, prob_commit_loss, entropy_loss, avg_entropy_loss = self.quantizer(z, cur_iter)
        with torch.no_grad():
            x_rec = self.decoder(z_q)

        vq_loss = F.mse_loss(z_q, z_obj.detach())
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        transplant_loss  = vq_loss + categorical_loss
        return transplant_loss, rec_loss, vq_loss, categorical_loss, prob_commit_loss, entropy_loss, avg_entropy_loss
    
    def refinement(self, x, cur_iter):
        assert self.args.stage == "refinement"
        with torch.no_grad():
            z = self.encoder(x)
            z_q, _, _, _, _, _ = self.quantizer(z, cur_iter)
        x_rec = self.decoder(z_q)
        return x_rec

    def collect_eval_info_transplant(self, x):
        z_obj = self.encoder2(x) 
        z = self.encoder(x)
        z_q, prob_commit_loss, entropy_loss, avg_entropy_loss = self.quantizer.collect_eval_info(z)
        x_rec = self.decoder(z_q).clamp_(-1, 1)

        vq_loss = F.mse_loss(z_q.detach(), z_obj.detach())
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss, vq_loss, prob_commit_loss, entropy_loss, avg_entropy_loss

    def collect_eval_info_refinement(self, x):
        z = self.encoder(x)
        z_q, prob_commit_loss, commit_loss, entropy_loss, avg_entropy_loss = self.quantizer.collect_eval_info(z)
        x_rec = self.decoder(z_q).clamp_(-1, 1)

        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss, prob_commit_loss, commit_loss, entropy_loss, avg_entropy_loss
