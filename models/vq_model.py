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
        if args.VQ == "vanilla_vq":
            self.quantizer1 = VanillaVectorQuantizer(args)
            self.quantizer2 = VanillaVectorQuantizer(args)
            self.quantizer3 = VanillaVectorQuantizer(args)
            self.quantizer4 = VanillaVectorQuantizer(args)
        elif args.VQ == "ema_vq":
            self.quantizer1 = EMAVectorQuantizer(args)
            self.quantizer2 = EMAVectorQuantizer(args)
            self.quantizer3 = EMAVectorQuantizer(args)
            self.quantizer4 = EMAVectorQuantizer(args)
        elif args.VQ == "online_vq":
            self.quantizer1 = OnlineVectorQuantizer(args)
            self.quantizer2 = OnlineVectorQuantizer(args)
            self.quantizer3 = OnlineVectorQuantizer(args)
            self.quantizer4 = OnlineVectorQuantizer(args)
        elif args.VQ == "wasserstein_vq":
            self.quantizer1 = WassersteinVectorQuantizer(args)
            self.quantizer2 = WassersteinVectorQuantizer(args)
            self.quantizer3 = WassersteinVectorQuantizer(args)
            self.quantizer4 = WassersteinVectorQuantizer(args)
        elif args.VQ == "mmd_vq":
            self.quantizer1 = MMDVectorQuantizer(args)
            self.quantizer2 = MMDVectorQuantizer(args)
            self.quantizer3 = MMDVectorQuantizer(args)
            self.quantizer4 = MMDVectorQuantizer(args)

        self.projector_in = nn.Sequential(
                nn.Conv2d(32, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024,  4*self.codebook_dim, kernel_size=3, padding=1),
            )
        self.projector_out = nn.Sequential(
                nn.Conv2d(4*self.codebook_dim, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 32, kernel_size=3, padding=1),
            )

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
            for param in self.quantizer1.parameters():
                param.requires_grad = True
            for param in self.quantizer2.parameters():
                param.requires_grad = True
            for param in self.quantizer3.parameters():
                param.requires_grad = True
            for param in self.quantizer4.parameters():
                param.requires_grad = True
            for param in self.projector_in.parameters():
                param.requires_grad = True
            for param in self.projector_out.parameters():
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
            quantizer1_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer1.')}
            quantizer2_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer2.')}
            quantizer3_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer3.')}
            quantizer4_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer4.')}
            projector_in_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('projector_in.')}
            projector_out_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('projector_out.')}

            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            quantizer1_dict = {k.replace('quantizer1.', '', 1): v for k, v in quantizer1_dict.items()}
            quantizer2_dict = {k.replace('quantizer2.', '', 1): v for k, v in quantizer2_dict.items()}
            quantizer3_dict = {k.replace('quantizer3.', '', 1): v for k, v in quantizer3_dict.items()}
            quantizer4_dict = {k.replace('quantizer4.', '', 1): v for k, v in quantizer4_dict.items()}
            projector_in_dict = {k.replace('projector_in.', '', 1): v for k, v in projector_in_dict.items()}
            projector_out_dict = {k.replace('projector_out.', '', 1): v for k, v in projector_out_dict.items()}

            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            self.quantizer1.load_state_dict(quantizer1_dict, strict=True)
            self.quantizer2.load_state_dict(quantizer2_dict, strict=True)
            self.quantizer3.load_state_dict(quantizer3_dict, strict=True)
            self.quantizer4.load_state_dict(quantizer4_dict, strict=True)
            self.projector_in.load_state_dict(projector_in_dict, strict=True)
            self.projector_out.load_state_dict(projector_out_dict, strict=True)
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.quantizer1.parameters():
                param.requires_grad = False
            for param in self.quantizer2.parameters():
                param.requires_grad = False
            for param in self.quantizer3.parameters():
                param.requires_grad = False
            for param in self.quantizer4.parameters():
                param.requires_grad = False
            for param in self.projector_in.parameters():
                param.requires_grad = False
            for param in self.projector_out.parameters():
                param.requires_grad = False 
            for param in self.decoder.parameters():
                param.requires_grad = True
            self.encoder.eval()
            self.quantizer1.eval()
            self.quantizer2.eval()
            self.quantizer3.eval()
            self.quantizer4.eval()
            self.projector_in.eval()
            self.projector_out.eval()

    def transplant(self, x):
        assert self.args.stage == "transplant"
        with torch.no_grad():
            z = self.encoder(x)

        z_p = self.projector_in(z)
        z_1, z_2, z_3, z_4 = torch.chunk(z_p, 4, dim=1)
        z_q_1, vq_loss_1 = self.quantizer1(z_1)
        z_q_2, vq_loss_2 = self.quantizer2(z_2)
        z_q_3, vq_loss_3 = self.quantizer3(z_3)
        z_q_4, vq_loss_4 = self.quantizer4(z_4)
        z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
        z_q = self.projector_out(z_q)

        loss = F.mse_loss(z_q, z.detach())
        quant_error = F.mse_loss(z_q.detach(), z.detach())
        with torch.no_grad():
            x_rec = self.decoder(z_q)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        transplant_loss = 5.0 * loss + vq_loss_1 + vq_loss_2 + vq_loss_3 + vq_loss_4
        return  transplant_loss, rec_loss, quant_error
    
    def refinement(self, x):
        assert self.args.stage == "refinement"
        with torch.no_grad():
            z = self.encoder(x)
            z_p = self.projector_in(z)
            z_1, z_2, z_3, z_4 = torch.chunk(z_p, 4, dim=1)
            z_q_1, _ = self.quantizer1(z_1)
            z_q_2, _ = self.quantizer2(z_2)
            z_q_3, _ = self.quantizer3(z_3)
            z_q_4, _ = self.quantizer4(z_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
            z_q = self.projector_out(z_q)

        x_rec = self.decoder(z_q)
        return x_rec

    def collect_eval_info_transplant(self, x):
        z = self.encoder(x)
        z_p = self.projector_in(z)
        z_1, z_2, z_3, z_4 = torch.chunk(z_p, 4, dim=1)
        z_q_1 = self.quantizer1.collect_eval_info(z_1)
        z_q_2 = self.quantizer2.collect_eval_info(z_2)
        z_q_3 = self.quantizer3.collect_eval_info(z_3)
        z_q_4 = self.quantizer4.collect_eval_info(z_4)
        z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
        z_q = self.projector_out(z_q)
        quant_error = F.mse_loss(z_q.detach(), z.detach())
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss, quant_error

    def collect_eval_info_refinement(self, x):
        z = self.encoder(x)
        z_p = self.projector_in(z)
        z_1, z_2, z_3, z_4 = torch.chunk(z_p, 4, dim=1)
        z_q_1 = self.quantizer1.collect_eval_info(z_1)
        z_q_2 = self.quantizer2.collect_eval_info(z_2)
        z_q_3 = self.quantizer3.collect_eval_info(z_3)
        z_q_4 = self.quantizer4.collect_eval_info(z_4)
        z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
        z_q = self.projector_out(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss
