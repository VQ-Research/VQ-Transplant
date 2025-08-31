import os
import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from models.vanilla_quantizer import VanillaVectorQuantizer
from models.ema_quantizer import EMAVectorQuantizer
from models.online_quantizer import OnlineVectorQuantizer
from models.wasserstein_quantizer import WassersteinVectorQuantizer
from models.mmd_quantizer import MMDVectorQuantizer
from models.encoder_decoder import Encoder, Decoder
from utils.util import Pack
from safetensors.torch import load_file
from models.lpips import LPIPS

class VQModel(nn.Module):
    def __init__(self, args):
        super(VQModel, self).__init__()
        self.args = args
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.quant_conv = torch.nn.Conv2d(256, 8, 1)
        self.post_quant_conv = torch.nn.Conv2d(8, 256, 1)

        if args.VQ == "vanilla_vq":
            self.quantizer = VanillaVectorQuantizer(args)
        elif args.VQ == "ema_vq":
            self.quantizer = EMAVectorQuantizer(args)
        elif args.VQ == "online_vq":
            self.quantizer = OnlineVectorQuantizer(args)
        elif args.VQ == "wasserstein_vq":
            self.quantizer = WassersteinVectorQuantizer(args)
        elif args.VQ == "mmd_vq":
            self.quantizer = MMDVectorQuantizer(args)

        self.projector_in = nn.Sequential(
                nn.Conv2d(8, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 8, kernel_size=3, padding=1),
            )
        self.projector_out = nn.Sequential(
                nn.Conv2d(8, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 8, kernel_size=3, padding=1),
            )

        if args.stage == "transplant":
            self.perceptual_loss = LPIPS().eval()
            pretrain_dict = torch.load(args.pretrained_tokenizer, map_location='cpu', weights_only=False)["model"]
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
            for param in self.quant_conv.parameters():
                param.requires_grad = False
            for param in self.quantizer.parameters():
                param.requires_grad = True
            for param in self.projector_out.parameters():
                param.requires_grad = True
            for param in self.projector_in.parameters():
                param.requires_grad = True
            for param in self.post_quant_conv.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.decoder.eval()
            self.quant_conv.eval()
            self.post_quant_conv.eval()

        if args.stage == "refinement":
            checkpoint_dir = os.path.join(os.path.join(args.init_checkpoint_dir, "Transplant"), args.dataset_name)
            checkpoint_name = args.checkpoint_name
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

            pretrain_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
            encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
            decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
            quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quant_conv.')}
            post_quant_conv_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('post_quant_conv.')}
            quantizer_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer.')}
            projector_out_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('projector_out.')}
            projector_in_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('projector_in.')}

            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            quant_conv_dict = {k.replace('quant_conv.', '', 1): v for k, v in quant_conv_dict.items()}
            post_quant_conv_dict = {k.replace('post_quant_conv.', '', 1): v for k, v in post_quant_conv_dict.items()}
            quantizer_dict = {k.replace('quantizer.', '', 1): v for k, v in quantizer_dict.items()}
            projector_out_dict = {k.replace('projector_out.', '', 1): v for k, v in projector_out_dict.items()}
            projector_in_dict = {k.replace('projector_in.', '', 1): v for k, v in projector_in_dict.items()}

            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            self.quant_conv.load_state_dict(quant_conv_dict, strict=True)
            self.post_quant_conv.load_state_dict(post_quant_conv_dict, strict=True)
            self.quantizer.load_state_dict(quantizer_dict, strict=True)
            self.projector_out.load_state_dict(projector_out_dict, strict=True)
            self.projector_in.load_state_dict(projector_in_dict, strict=True)
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters():
                param.requires_grad = False
            for param in self.quantizer.parameters():
                param.requires_grad = False
            for param in self.projector_in.parameters():
                param.requires_grad = False
            for param in self.projector_out.parameters():
                param.requires_grad = False
            for param in self.post_quant_conv.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            self.encoder.eval()
            self.quant_conv.eval()
            self.projector_in.eval()
            self.projector_out.eval()
            self.quantizer.eval()

    def transplant(self, x):
        assert self.args.stage == "transplant"
        with torch.no_grad():
            ze = self.encoder(x)
            z_pre = self.quant_conv(ze)
            z_obj = F.normalize(z_pre, p=2, dim=-1)

        z_p = F.normalize(z_pre, p=2, dim=-1)
        z_q, vq_loss, utilization, perplexity = self.quantizer(z_p)
        #z_q = F.normalize(z_q, p=2, dim=-1)

        loss = F.mse_loss(z_q, z_obj.detach())
        quant_error = F.mse_loss(z_q.detach(), z_obj.detach())
        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q)

        p_loss = self.perceptual_loss(x.contiguous(), x_rec.contiguous())
        p_loss = torch.mean(p_loss)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        transplant_loss = 5.0 * rec_loss + p_loss + 10.0 * loss + vq_loss
        return  transplant_loss, rec_loss, p_loss, quant_error, utilization, perplexity

    def refinement(self, x):
        assert self.args.stage == "refinement"
        with torch.no_grad():
            ze = self.encoder(x)
            z_pre = self.quant_conv(ze)

            z_p = F.normalize(z_pre + self.projector_in(z_pre), p=2, dim=-1)
            z_q, _ = self.quantizer.collect_eval_info(z_p)
            z_q = F.normalize(z_q + self.projector_out(z_q), p=2, dim=-1)

        z_q = self.post_quant_conv(z_q)  
        x_rec = self.decoder(z_q)
        return x_rec

    def collect_eval_info_transplant(self, x):
        ze = self.encoder(x)
        z_pre = self.quant_conv(ze)
        z_obj = F.normalize(z_pre, p=2, dim=-1)

        z_p = F.normalize(z_pre + self.projector_in(z_pre), p=2, dim=-1)
        z_q, histogram = self.quantizer.collect_eval_info(z_p)
        z_q = F.normalize(z_q + self.projector_out(z_q), p=2, dim=-1)

        quant_error = F.mse_loss(z_q.detach(), z_obj.detach())
        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss, quant_error, histogram

    def collect_eval_info_refinement(self, x):
        ze = self.encoder(x)
        z_pre = self.quant_conv(ze)

        z_p = F.normalize(z_pre + self.projector_in(z_pre), p=2, dim=-1)
        z_q, _ = self.quantizer.collect_eval_info(z_p)
        z_q = F.normalize(z_q + self.projector_out(z_q), p=2, dim=-1)

        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss
        
    def reconstruction(self, x):
        ze = self.encoder(x)
        z_pre = self.quant_conv(ze)

        z_p = F.normalize(z_pre + self.projector_in(z_pre), p=2, dim=-1)
        z_q = self.quantizer.collect_reconstruction(z_p)
        z_q = F.normalize(z_q + self.projector_out(z_q), p=2, dim=-1)
        
        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        return x_rec

