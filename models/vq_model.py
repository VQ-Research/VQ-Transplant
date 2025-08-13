import os
import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from models.vanilla_vq import VanillaQuantizer, MultiscaleVanillaQuantizer
from models.ema_vq import EMAQuantizer, MultiscaleEMAQuantizer
from models.online_vq import OnlineQuantizer, MultiscaleOnlineQuantizer
from models.wasserstein_vq import WassersteinQuantizer, MultiscaleWassersteinQuantizer
from models.mmd_vq import MMDQuantizer, MultiscaleMMDQuantizer
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
        if args.use_multiscale:
            if args.VQ == "vanilla_vq":
                self.quantizer = MultiscaleVanillaQuantizer(args)
            elif args.VQ == "ema_vq":
                self.quantizer = MultiscaleEMAQuantizer(args)
            elif args.VQ == "online_vq":
                self.quantizer = MultiscaleOnlineQuantizer(args)
            elif args.VQ == "wasserstein_vq":
                self.quantizer = MultiscaleWassersteinQuantizer(args)
            elif args.VQ == "mmd_vq":
                self.quantizer = MultiscaleMMDQuantizer(args)
        else:
            if args.VQ == "vanilla_vq":
                self.quantizer = VanillaQuantizer(args)
            elif args.VQ == "ema_vq":
                self.quantizer = EMAQuantizer(args)
            elif args.VQ == "online_vq":
                self.quantizer = OnlineQuantizer(args)
            elif args.VQ == "wasserstein_vq":
                self.quantizer = WassersteinQuantizer(args)
            elif args.VQ == "mmd_vq":
                self.quantizer = MMDQuantizer(args)

        if args.stage == "transplant":
            pretrain_dict = load_file(args.pretrained_tokenizer)
            encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
            decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quantizer.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.decoder.eval()

        if args.stage == "refinement":
            checkpoint_dir = os.path.join(os.path.join(args.init_checkpoint_dir, "Transplant"), args.dataset_name)
            checkpoint_name = args.checkpoint_name
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

            pretrain_dict = torch.load(checkpoint_path, map_location='cpu')['model']
            encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
            decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
            quantizer_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer.')}

            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            quantizer_dict = {k.replace('quantizer.', '', 1): v for k, v in quantizer_dict.items()}

            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            self.quantizer.load_state_dict(quantizer_dict, strict=True)
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quantizer.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = True
            self.encoder.eval()
            self.quantizer.eval()

    def transplant(self, x):
        assert self.args.stage == "transplant"
        with torch.no_grad():
            z = self.encoder(x)

        if self.args.VQ == "ema_vq" and self.args.use_multiscale==False and self.args.residual==False:
            z_q, quant_error, utilization, perplexity = self.quantizer(z)
        else:
            z_q, transplant_loss, quant_error, utilization, perplexity = self.quantizer(z)

        with torch.no_grad():
            x_rec = self.decoder(z_q)
            
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        if self.args.VQ == "ema_vq" and self.args.use_multiscale==False and self.args.residual==False: 
            return  rec_loss, quant_error, utilization, perplexity
        else:
            return  transplant_loss, rec_loss, quant_error, utilization, perplexity
    
    def refinement(self, x):
        assert self.args.stage == "refinement"
        with torch.no_grad():
            z = self.encoder(x)
            if self.args.VQ == "ema_vq" and self.args.use_multiscale==False and self.args.residual==False:
                z_q, _, _, _ = self.quantizer(z)
            else:
                z_q, _, _, _, _ = self.quantizer(z)
        x_rec = self.decoder(z_q)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec

    def collect_eval_info_transplant(self, x):
        z = self.encoder(x)
        z_q, quant_error, histogram = self.quantizer.collect_eval_info(z)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss, quant_error, histogram 

    def collect_eval_info_refinement(self, x):
        z = self.encoder(x)
        z_q, _, _ = self.quantizer.collect_eval_info(z)
        x_rec = self.decoder(z_q).clamp_(-1, 1)

        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss
