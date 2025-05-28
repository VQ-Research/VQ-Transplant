import sys
import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
from model.encoder_decoder import Encoder, Decoder
from utils.util import Pack

class VAR_Substitution(nn.Module):
    def __init__(self, args):
        super(VAR_Substitution, self).__init__()
        self.args = args

