import os
import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from models.vanilla_vq import VanillaQuantizer
from models.ema_vq import EMAQuantizer
from models.online_vq import OnlineQuantizer
from models.wasserstein_vq import WassersteinQuantizer
from models.mmd_vq import MMDQuantizer
from models.encoder_decoder import Encoder, Decoder, EncoderConfig, DecoderConfig
from utils.util import Pack
from safetensors.torch import load_file

class PQModel(nn.Module):
    def __init__(self, args):
        super(PQModel, self).__init__()
        self.args = args
        enc_config = EncoderConfig
        dec_config = DecoderConfig
        self.encoder = Encoder(EncoderConfig)
        self.decoder = Decoder(DecoderConfig)
        
        if args.VQ == "vanilla_vq":
            if args.pq == 2:
                self.quantizer1 = VanillaQuantizer(args)
                self.quantizer2 = VanillaQuantizer(args)
            elif args.pq == 4:
                self.quantizer1 = VanillaQuantizer(args)
                self.quantizer2 = VanillaQuantizer(args)
                self.quantizer3 = VanillaQuantizer(args)
                self.quantizer4 = VanillaQuantizer(args)

        elif args.VQ == "ema_vq":
            if args.pq == 2:
                self.quantizer1 = EMAQuantizer(args)
                self.quantizer2 = EMAQuantizer(args)
            elif args.pq == 4:
                self.quantizer1 = EMAQuantizer(args)
                self.quantizer2 = EMAQuantizer(args)
                self.quantizer3 = EMAQuantizer(args)
                self.quantizer4 = EMAQuantizer(args)

        elif args.VQ == "online_vq":
            if args.pq == 2:
                self.quantizer1 = OnlineQuantizer(args)
                self.quantizer2 = OnlineQuantizer(args)
            elif args.pq == 4:
                self.quantizer1 = OnlineQuantizer(args)
                self.quantizer2 = OnlineQuantizer(args)
                self.quantizer3 = OnlineQuantizer(args)
                self.quantizer4 = OnlineQuantizer(args)

        elif args.VQ == "wasserstein_vq":
            if args.pq == 2:
                self.quantizer1 = WassersteinQuantizer(args)
                self.quantizer2 = WassersteinQuantizer(args)
            elif args.pq == 4:
                self.quantizer1 = WassersteinQuantizer(args)
                self.quantizer2 = WassersteinQuantizer(args)
                self.quantizer3 = WassersteinQuantizer(args)
                self.quantizer4 = WassersteinQuantizer(args)

        elif args.VQ == "mmd_vq":
            if args.pq == 2:
                self.quantizer1 = MMDQuantizer(args)
                self.quantizer2 = MMDQuantizer(args)
            elif args.pq == 4:
                self.quantizer1 = MMDQuantizer(args)
                self.quantizer2 = MMDQuantizer(args)
                self.quantizer3 = MMDQuantizer(args)
                self.quantizer4 = MMDQuantizer(args)

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
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.decoder.eval()
            
            if args.pq == 2:
                for param in self.quantizer1.parameters():
                    param.requires_grad = True
                for param in self.quantizer2.parameters():
                    param.requires_grad = True
            elif args.pq == 4:
                for param in self.quantizer1.parameters():
                    param.requires_grad = True
                for param in self.quantizer2.parameters():
                    param.requires_grad = True
                for param in self.quantizer3.parameters():
                    param.requires_grad = True
                for param in self.quantizer4.parameters():
                    param.requires_grad = True

        if args.stage == "refinement":
            checkpoint_dir = os.path.join(os.path.join(args.init_checkpoint_dir, "Transplant"), args.dataset_name)
            checkpoint_name = args.checkpoint_name
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

            pretrain_dict = torch.load(checkpoint_path, map_location='cpu')['model']
            encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
            decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
            if args.pq ==2:
                quantizer_dict1 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer1.')}
                quantizer_dict2 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer2.')}
            elif args.pq == 4:
                quantizer_dict1 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer1.')}
                quantizer_dict2 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer2.')}
                quantizer_dict3 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer3.')}
                quantizer_dict4 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer4.')}

            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            if args.pq == 2:
                quantizer_dict1 = {k.replace('quantizer1.', '', 1): v for k, v in quantizer_dict1.items()}
                quantizer_dict2 = {k.replace('quantizer2.', '', 1): v for k, v in quantizer_dict2.items()}
            elif args.pq == 4:
                quantizer_dict1 = {k.replace('quantizer1.', '', 1): v for k, v in quantizer_dict1.items()}
                quantizer_dict2 = {k.replace('quantizer2.', '', 1): v for k, v in quantizer_dict2.items()}
                quantizer_dict3 = {k.replace('quantizer3.', '', 1): v for k, v in quantizer_dict3.items()}
                quantizer_dict4 = {k.replace('quantizer4.', '', 1): v for k, v in quantizer_dict4.items()}

            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            if args.pq == 2:
                self.quantizer1.load_state_dict(quantizer_dict1, strict=True)
                self.quantizer2.load_state_dict(quantizer_dict2, strict=True)
            elif args.pq == 4:
                self.quantizer1.load_state_dict(quantizer_dict1, strict=True)
                self.quantizer2.load_state_dict(quantizer_dict2, strict=True)
                self.quantizer3.load_state_dict(quantizer_dict3, strict=True)
                self.quantizer4.load_state_dict(quantizer_dict4, strict=True)

            for param in self.encoder.parameters():
                param.requires_grad = False
            if args.pq ==2:
                for param in self.quantizer1.parameters():
                    param.requires_grad = False
                for param in self.quantizer2.parameters():
                    param.requires_grad = False
            elif args.pq == 4:
                for param in self.quantizer1.parameters():
                    param.requires_grad = False
                for param in self.quantizer2.parameters():
                    param.requires_grad = False
                for param in self.quantizer3.parameters():
                    param.requires_grad = False
                for param in self.quantizer4.parameters():
                    param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = True
            self.encoder.eval()

            if args.pq ==2:
                self.quantizer1.eval()
                self.quantizer2.eval()
            elif args.pq == 4:
                self.quantizer1.eval()
                self.quantizer2.eval()
                self.quantizer3.eval()
                self.quantizer4.eval()

    def transplant(self, x):
        assert self.args.stage == "transplant"
        with torch.no_grad():
            z = self.encoder(x)
            if self.args.pq == 2:
                z_1, z_2 = torch.chunk(z, 2, dim=1)
            elif self.args.pq == 4:
                z_1, z_2, z_3, z_4 = torch.chunk(z, 4, dim=1)

        if self.args.pq == 2:
            z_q_1, transplant_loss_1, _, utilization_1, _ = self.quantizer1(z_1)
            z_q_2, transplant_loss_2, _, utilization_2, _ = self.quantizer2(z_2)
            transplant_loss = 0.5 * (transplant_loss_1 + transplant_loss_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
            utilization = 0.5*(utilization_1 + utilization_2)
        elif self.args.pq == 4:
            z_q_1, transplant_loss_1, _, utilization_1, _ = self.quantizer1(z_1)
            z_q_2, transplant_loss_2, _, utilization_2, _ = self.quantizer2(z_2)
            z_q_3, transplant_loss_3, _, utilization_3, _ = self.quantizer3(z_3)
            z_q_4, transplant_loss_4, _, utilization_4, _ = self.quantizer4(z_4)
            transplant_loss = 0.25 * (transplant_loss_1 + transplant_loss_2 + transplant_loss_3 + transplant_loss_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
            utilization = 0.25*(utilization_1 + utilization_2 + utilization_3 + utilization_4)

        quant_error = F.mse_loss(z_q.detach(), z.detach())
        with torch.no_grad():
            x_rec = self.decoder(z_q)
            
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return  transplant_loss, rec_loss, quant_error, utilization
    
    def refinement(self, x):
        assert self.args.stage == "refinement"
        with torch.no_grad():
            z = self.encoder(x)
            if self.args.pq == 2:
                z_1, z_2 = torch.chunk(z, 2, dim=1)
            elif self.args.pq == 4:
                z_1, z_2, z_3, z_4 = torch.chunk(z, 4, dim=1)

            if self.args.pq == 2:
                z_q_1, _, _, _, _ = self.quantizer1(z_1)
                z_q_2, _, _, _, _ = self.quantizer2(z_2)
                z_q = torch.cat((z_q_1, z_q_2), dim=1)
            elif self.args.pq == 4:
                z_q_1, _, _, _, _ = self.quantizer1(z_1)
                z_q_2, _, _, _, _ = self.quantizer2(z_2)
                z_q_3, _, _, _, _ = self.quantizer3(z_3)
                z_q_4, _, _, _, _ = self.quantizer4(z_4)
                z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)

        x_rec = self.decoder(z_q)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec

    def collect_eval_info_transplant(self, x):
        z = self.encoder(x)
        if self.args.pq == 2:
            z_1, z_2 = torch.chunk(z, 2, dim=1)
        elif self.args.pq == 4:
            z_1, z_2, z_3, z_4 = torch.chunk(z, 4, dim=1)

        if self.args.pq == 2:
            z_q_1, _, _ = self.quantizer1.collect_eval_info(z_1)
            z_q_2, _, _ = self.quantizer2.collect_eval_info(z_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
        elif self.args.pq == 4:
            z_q_1, _, _ = self.quantizer1.collect_eval_info(z_1)
            z_q_2, _, _ = self.quantizer2.collect_eval_info(z_2)
            z_q_3, _, _ = self.quantizer3.collect_eval_info(z_3)
            z_q_4, _, _ = self.quantizer4.collect_eval_info(z_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)

        quant_error = F.mse_loss(z_q.detach(), z.detach())
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss, quant_error

    def collect_eval_info_refinement(self, x):
        z = self.encoder(x)
        if self.args.pq == 2:
            z_1, z_2 = torch.chunk(z, 2, dim=1)
        elif self.args.pq == 4:
            z_1, z_2, z_3, z_4 = torch.chunk(z, 4, dim=1)

        if self.args.pq == 2:
            z_q_1, _, _ = self.quantizer1.collect_eval_info(z_1)
            z_q_2, _, _ = self.quantizer2.collect_eval_info(z_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
        elif self.args.pq == 4:
            z_q_1, _, _ = self.quantizer1.collect_eval_info(z_1)
            z_q_2, _, _ = self.quantizer2.collect_eval_info(z_2)
            z_q_3, _, _ = self.quantizer3.collect_eval_info(z_3)
            z_q_4, _, _ = self.quantizer4.collect_eval_info(z_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
            
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss
