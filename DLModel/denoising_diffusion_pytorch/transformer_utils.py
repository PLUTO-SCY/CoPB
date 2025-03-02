import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm
from typing import Optional
import numpy as np

from denoising_diffusion_pytorch.version import __version__
# helpers functions

def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE


def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 24) -> torch.Tensor:
    """Generate positional encoding with a given period.

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    period:
        Size of the pattern to repeat.
        Default is 24.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE


def generate_local_map_mask(chunk_size: int,
                            attention_size: int,
                            mask_future=False,
                            device: torch.device = 'cpu') -> torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)

    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size

    return torch.BoolTensor(local_map).to(device)



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
    
