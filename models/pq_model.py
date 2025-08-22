import os
import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from models.vanilla_quantizer import VanillaProductQuantizer
from models.ema_quantizer import EMAProductQuantizer
from models.online_quantizer import OnlineProductQuantizer
from models.wasserstein_quantizer import WassersteinProductQuantizer
from models.mmd_quantizer import MMDProductQuantizer
from models.encoder_decoder import Encoder, Decoder
from utils.util import Pack
from safetensors.torch import load_file
from torch import distributed as tdist

class PQModel(nn.Module):
    def __init__(self, args):
        super(PQModel, self).__init__()
        self.args = args
        enc_ddconfig = dict(double_z=True, z_channels=16, resolution=256, in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, out_ch=3, ch=128, attn_resolutions=[16], dropout=0.0,)
        dec_ddconfig = dict(z_channels=16, resolution=256, in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, out_ch=3, ch=128, attn_resolutions=[16], dropout=0.0,)
        self.encoder = Encoder(**enc_ddconfig)
        self.decoder = Decoder(**dec_ddconfig)
        self.quant_conv = torch.nn.Conv2d(32, 32, 1)
        self.post_quant_conv = torch.nn.Conv2d(16, 16, 1)
        if args.pq == 2:
            self.overall_codebook_size = args.codebook_size * args.codebook_size
        elif args.pq == 4:
            self.overall_codebook_size = args.codebook_size * args.codebook_size * args.codebook_size * args.codebook_size

        if args.VQ == "vanilla_vq":
            if args.pq == 2:
                self.quantizer1 = VanillaProductQuantizer(args)
                self.quantizer2 = VanillaProductQuantizer(args)
            elif args.pq == 4:
                self.quantizer1 = VanillaProductQuantizer(args)
                self.quantizer2 = VanillaProductQuantizer(args)
                self.quantizer3 = VanillaProductQuantizer(args)
                self.quantizer4 = VanillaProductQuantizer(args)

        elif args.VQ == "ema_vq":
            if args.pq == 2:
                self.quantizer1 = EMAProductQuantizer(args)
                self.quantizer2 = EMAProductQuantizer(args)
            elif args.pq == 4:
                self.quantizer1 = EMAProductQuantizer(args)
                self.quantizer2 = EMAProductQuantizer(args)
                self.quantizer3 = EMAProductQuantizer(args)
                self.quantizer4 = EMAProductQuantizer(args)

        elif args.VQ == "online_vq":
            if args.pq == 2:
                self.quantizer1 = OnlineProductQuantizer(args)
                self.quantizer2 = OnlineProductQuantizer(args)
            elif args.pq == 4:
                self.quantizer1 = OnlineProductQuantizer(args)
                self.quantizer2 = OnlineProductQuantizer(args)
                self.quantizer3 = OnlineProductQuantizer(args)
                self.quantizer4 = OnlineProductQuantizer(args)

        elif args.VQ == "wasserstein_vq":
            if args.pq == 2:
                self.quantizer1 = WassersteinProductQuantizer(args)
                self.quantizer2 = WassersteinProductQuantizer(args)
            elif args.pq == 4:
                self.quantizer1 = WassersteinProductQuantizer(args)
                self.quantizer2 = WassersteinProductQuantizer(args)
                self.quantizer3 = WassersteinProductQuantizer(args)
                self.quantizer4 = WassersteinProductQuantizer(args)

        elif args.VQ == "mmd_vq":
            if args.pq == 2:
                self.quantizer1 = MMDProductQuantizer(args)
                self.quantizer2 = MMDProductQuantizer(args)
            elif args.pq == 4:
                self.quantizer1 = MMDProductQuantizer(args)
                self.quantizer2 = MMDProductQuantizer(args)
                self.quantizer3 = MMDProductQuantizer(args)
                self.quantizer4 = MMDProductQuantizer(args)

        self.projector_in = nn.Sequential(
                nn.Conv2d(32, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, args.pq * args.codebook_dim, kernel_size=3, padding=1),
            )
        self.projector_out = nn.Sequential(
                nn.Conv2d(args.pq * args.codebook_dim, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.BatchNorm2d(1024),
                nn.SiLU(),
                nn.Conv2d(1024, 16, kernel_size=3, padding=1),
            )

        if args.stage == "transplant":
            pretrain_dict = torch.load(args.pretrained_tokenizer, map_location='cpu', weights_only=False)["state_dict"]
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

            self.encoder.eval()
            self.decoder.eval()
            self.quant_conv.eval()
            self.post_quant_conv.eval()

            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.quant_conv.parameters():
                param.requires_grad = False
            for param in self.post_quant_conv.parameters():
                param.requires_grad = False
            for param in self.projector_in.parameters():
                param.requires_grad = True
            for param in self.projector_out.parameters():
                param.requires_grad = True
            
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

            pretrain_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)['model']
            encoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('encoder.')}
            decoder_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('decoder.')}
            projector_in_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('projector_in.')}
            projector_out_dict = {k: v for k, v in pretrain_dict.items() if k.startswith('projector_out.')}

            encoder_dict = {k.replace('encoder.', '', 1): v for k, v in encoder_dict.items()}
            decoder_dict = {k.replace('decoder.', '', 1): v for k, v in decoder_dict.items()}
            projector_in_dict = {k.replace('projector_in.', '', 1): v for k, v in projector_in_dict.items()}
            projector_out_dict = {k.replace('projector_out.', '', 1): v for k, v in projector_out_dict.items()}

            self.encoder.load_state_dict(encoder_dict, strict=True)
            self.decoder.load_state_dict(decoder_dict, strict=True)
            self.projector_in.load_state_dict(projector_in_dict, strict=True)
            self.projector_out.load_state_dict(projector_out_dict, strict=True)

            if args.pq ==2:
                quantizer_dict1 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer1.')}
                quantizer_dict2 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer2.')}
                quantizer_dict1 = {k.replace('quantizer1.', '', 1): v for k, v in quantizer_dict1.items()}
                quantizer_dict2 = {k.replace('quantizer2.', '', 1): v for k, v in quantizer_dict2.items()}
                self.quantizer1.load_state_dict(quantizer_dict1, strict=True)
                self.quantizer2.load_state_dict(quantizer_dict2, strict=True)
            elif args.pq == 4:
                quantizer_dict1 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer1.')}
                quantizer_dict2 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer2.')}
                quantizer_dict3 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer3.')}
                quantizer_dict4 = {k: v for k, v in pretrain_dict.items() if k.startswith('quantizer4.')}

                quantizer_dict1 = {k.replace('quantizer1.', '', 1): v for k, v in quantizer_dict1.items()}
                quantizer_dict2 = {k.replace('quantizer2.', '', 1): v for k, v in quantizer_dict2.items()}
                quantizer_dict3 = {k.replace('quantizer3.', '', 1): v for k, v in quantizer_dict3.items()}
                quantizer_dict4 = {k.replace('quantizer4.', '', 1): v for k, v in quantizer_dict4.items()}

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
            for param in self.projector_in.parameters():
                param.requires_grad = False
            for param in self.projector_out.parameters():
                param.requires_grad = False      
            for param in self.decoder.parameters():
                param.requires_grad = True
                
            self.encoder.eval()
            self.projector_in.eval()
            self.projector_out.eval()
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
            ze = self.encoder(x)
            zt = self.quant_conv(ze)
            zm, _ = torch.chunk(zt, 2, dim=1)
            z_obj = self.post_quant_conv(zm)

        z_p = self.projector_in(ze)
        if self.args.pq == 2:
            z_1, z_2 = torch.chunk(z_p, 2, dim=1)
            z_q_1, vq_loss_1, token_1 = self.quantizer1(z_1)
            z_q_2, vq_loss_2, token_2 = self.quantizer2(z_2)
            vq_loss = 0.5 * (vq_loss_1 + vq_loss_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
            token = token_1 + token_2 * self.args.codebook_size
        elif self.args.pq == 4:
            z_1, z_2, z_3, z_4 = torch.chunk(z_p, 4, dim=1)
            z_q_1, vq_loss_1, token_1 = self.quantizer1(z_1)
            z_q_2, vq_loss_2, token_2 = self.quantizer2(z_2)
            z_q_3, vq_loss_3, token_3 = self.quantizer3(z_3)
            z_q_4, vq_loss_4, token_4 = self.quantizer4(z_4)
            vq_loss = 0.25 * (vq_loss_1 + vq_loss_2 + vq_loss_3 + vq_loss_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
            token = token_1 + token_2 * self.args.codebook_size + token_3 * self.args.codebook_size * self.args.codebook_size + token_4 * self.args.codebook_size * self.args.codebook_size * self.args.codebook_size
        z_q = z_q + self.projector_out(z_q)

        loss = F.mse_loss(z_q, z_obj.detach())
        quant_error = F.mse_loss(z_q.detach(), z_obj.detach())
        x_rec = self.decoder(z_q)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        transplant_loss = rec_loss + 2.0 * loss + vq_loss

        histogram = token.bincount(minlength=self.overall_codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()

        codebook_usage_counts = (histogram > 0).float().sum()
        utilization = codebook_usage_counts.item() / self.overall_codebook_size
            
        avg_probs = histogram/histogram.sum(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return  transplant_loss, rec_loss, quant_error, utilization, perplexity

    def refinement(self, x):
        assert self.args.stage == "refinement"
        with torch.no_grad():
            ze = self.encoder(x)
            z_p = self.projector_in(ze)
            if self.args.pq == 2:
                z_1, z_2 = torch.chunk(z_p, 2, dim=1)
                z_q_1, _ = self.quantizer1.collect_eval_info(z_1)
                z_q_2, _ = self.quantizer2.collect_eval_info(z_2)
                z_q = torch.cat((z_q_1, z_q_2), dim=1)
            elif self.args.pq == 4:
                z_1, z_2, z_3, z_4 = torch.chunk(z_p, 4, dim=1)
                z_q_1, _ = self.quantizer1.collect_eval_info(z_1)
                z_q_2, _ = self.quantizer2.collect_eval_info(z_2)
                z_q_3, _ = self.quantizer3.collect_eval_info(z_3)
                z_q_4, _ = self.quantizer4.collect_eval_info(z_4)
                z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
            z_q = z_q + self.projector_out(z_q)
            
        x_rec = self.decoder(z_q)
        return x_rec

    def reconstruction(self, x):
        ze = self.encoder(x)
        z_p = self.projector_in(ze)
        if self.args.pq == 2:
            z_1, z_2 = torch.chunk(z_p, 2, dim=1)
            z_q_1 = self.quantizer1.collect_reconstruction(z_1)
            z_q_2 = self.quantizer2.collect_reconstruction(z_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
        elif self.args.pq == 4:
            z_1, z_2, z_3, z_4 = torch.chunk(z_p, 4, dim=1)
            z_q_1 = self.quantizer1.collect_reconstruction(z_1)
            z_q_2 = self.quantizer2.collect_reconstruction(z_2)
            z_q_3 = self.quantizer3.collect_reconstruction(z_3)
            z_q_4 = self.quantizer4.collect_reconstruction(z_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
        z_q = z_q + self.projector_out(z_q)
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        return x_rec

    def collect_eval_info_transplant(self, x):
        ze = self.encoder(x)
        zt = self.quant_conv(ze)
        zm, _ = torch.chunk(zt, 2, dim=1)
        z_obj = self.post_quant_conv(zm)

        z_p = self.projector_in(ze)
        if self.args.pq == 2:
            z_1, z_2 = torch.chunk(z_p, 2, dim=1)
            z_q_1, token_1 = self.quantizer1.collect_eval_info(z_1)
            z_q_2, token_2 = self.quantizer2.collect_eval_info(z_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
            token = token_1 + token_2 * self.args.codebook_size
        elif self.args.pq == 4:
            z_1, z_2, z_3, z_4 = torch.chunk(z_p, 4, dim=1)
            z_q_1, token_1 = self.quantizer1.collect_eval_info(z_1)
            z_q_2, token_2 = self.quantizer2.collect_eval_info(z_2)
            z_q_3, token_3 = self.quantizer3.collect_eval_info(z_3)
            z_q_4, token_4 = self.quantizer4.collect_eval_info(z_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
            token = token_1 + token_2 * self.args.codebook_size + token_3 * self.args.codebook_size * self.args.codebook_size + token_4 * self.args.codebook_size * self.args.codebook_size * self.args.codebook_size
        z_q = z_q + self.projector_out(z_q)

        quant_error = F.mse_loss(z_q.detach(), z_obj.detach())
        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())

        histogram = token.bincount(minlength=self.overall_codebook_size).float()
        handler = tdist.all_reduce(histogram, async_op=True)
        handler.wait()
        return x_rec, rec_loss, quant_error, histogram

    def collect_eval_info_refinement(self, x):
        ze = self.encoder(x)
        z_p = self.projector_in(ze)
        if self.args.pq == 2:
            z_1, z_2 = torch.chunk(z_p, 2, dim=1)
            z_q_1, _ = self.quantizer1.collect_eval_info(z_1)
            z_q_2, _ = self.quantizer2.collect_eval_info(z_2)
            z_q = torch.cat((z_q_1, z_q_2), dim=1)
        elif self.args.pq == 4:
            z_1, z_2, z_3, z_4 = torch.chunk(z_p, 4, dim=1)
            z_q_1, _ = self.quantizer1.collect_eval_info(z_1)
            z_q_2, _ = self.quantizer2.collect_eval_info(z_2)
            z_q_3, _ = self.quantizer3.collect_eval_info(z_3)
            z_q_4, _ = self.quantizer4.collect_eval_info(z_4)
            z_q = torch.cat((z_q_1, z_q_2, z_q_3, z_q_4), dim=1)
        z_q = z_q + self.projector_out(z_q)

        x_rec = self.decoder(z_q).clamp_(-1, 1)
        rec_loss = F.mse_loss(x.contiguous(), x_rec.contiguous())
        return x_rec, rec_loss