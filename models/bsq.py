import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import einsum
from einops import rearrange
from torch import distributed as tdist

class BSQ(nn.Module):
    def __init__(self, args):
        super(BSQ, self).__init__()
        self.args = args
        self.D = args.project_dim
        self.L = args.L
        self.codebook_size = self.L**self.D
        print("FSQ codebook size:", self.codebook_size)