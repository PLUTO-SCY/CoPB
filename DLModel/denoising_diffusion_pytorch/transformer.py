import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import sys
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

from denoising_diffusion_pytorch.version import __version__
from denoising_diffusion_pytorch.transformer_utils import *
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


# 不好用,不考虑
class TransformerDenoisingAdd(nn.Module):
    '''
    Transformer-based denoising with query-based conditioning using only `emb` for conditioning.
    '''

    def __init__(self,
                d_input: int,
                d_model: int,
                d_output: int,
                d_emb: int,  # Embedding size for the additional input variable
                N: int,  # Number of layers in transformer
                numCategory: int,
                dropout: float = 0.1,
                pe: str = None,
                learned_sinusoidal_cond: bool = False,
                random_fourier_features: bool = False,
                learned_sinusoidal_dim: int = 16,
                pe_period: int = None,
                ):
        """Create transformer structure with query-based conditioning."""
        super().__init__()

        self.d_model = d_model
        self.d_emb = d_emb  # New condition embedding size
        self.channels = d_input
        step_dim = d_model

        self.layers_encoding = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)
            for _ in range(N)])
    
        self._embedding = nn.Linear(d_input, d_model)

        # 定义 Embedding 层
        self.embCate_layer = nn.Embedding(num_embeddings=numCategory, embedding_dim=d_model)

        self._linear = nn.Linear(d_model, d_output)

        # Positional encoding setup
        if pe == "original":
            self._generate_PE = generate_original_PE
        elif pe == "regular":
            self._generate_PE = generate_regular_PE
        else:
            self._generate_PE = None
        self._pe_period = pe_period

        self.self_condition = False

        # Time embedding setup
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        # Embedding layers for the new condition variable (emb)
        self.emb_linear = nn.Linear(self.d_emb, d_model)

    def forward(self, x: torch.Tensor, t: torch.Tensor, emb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:
        '''
        x: The input sequence, shape [batch_size, channels, sequence_length]
        t: The time step embeddings, shape [batch_size, sequence_length]
        emb: The additional condition embedding, shape [batch_size, emb_dim]
        '''
        # print(x.shape)  # torch.Size([4, 2, 144])
        # print(emb.shape)  # torch.Size([4, 144])
        
        seqLength = x.shape[2]
        # x.shape:  [4, 2, 144] <- [batchsize, channel, length]
        xEmb = self._embedding(x.permute(0, 2, 1))  # xEmb.shape [4, 144, 256] <- [4, 144, 2] 所以其实是把channel的2变成了256

        cateEmbedding = self.embCate_layer(emb)  # [4, 144, 256]

        # Step embeddings
        step = self.step_mlp(t)
        step = step.unsqueeze(1)  # torch.Size([4, 1, 256])
        step_emb = torch.repeat_interleave(step, seqLength, dim=1)  # Repeat across seq_length

        # Prepare for transformer encoding
        encoding = xEmb
        encoding = encoding + step_emb  # Add step embedding to input

        # Add positional encoding if needed
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(seqLength, self.d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)  # Torch.Size([8, 64])
            encoding.add_(positional_encoding)  # Add positional encoding

        # Encoder stack
        for layer in self.layers_encoding:
            encoding = encoding + cateEmbedding  # 简单地将两个变量相加.
            encoding = layer(encoding)  # Pass through transformer layer

        output = self._linear(encoding)  # Final output projection

        # print('output.shape', output.shape)
        # print('here ok')
        # # 现在已经基本ok没啥问题了
        # sys.exit(0)

        return output.permute(0, 2, 1)  # Return with original shape [batch_size, seq_len, d_output]

# query 只加一次condition
class TransformerDenoisingQueryOnce(nn.Module):
    '''
    use x as key to query the condition embedding
    '''

    def __init__(self,
                d_input: int,
                d_model: int,
                d_output: int,
                d_emb: int,  # Embedding size for the additional input variable
                N: int,  # Number of layers in transformer
                numCategory: int,
                dropout: float = 0.1,
                pe: str = None,
                learned_sinusoidal_cond: bool = False,
                random_fourier_features: bool = False,
                learned_sinusoidal_dim: int = 16,
                pe_period: int = None,
                ):
        """Create transformer structure with query-based conditioning."""
        super().__init__()

        self.d_model = d_model
        self.d_emb = d_emb  # New condition embedding size
        self.channels = d_input
        step_dim = d_model

        self.layers_encoding = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)
            for _ in range(N)])
    
        self._embedding = nn.Linear(d_input, d_model)

        # 定义 Embedding 层
        self.embCate_layer = nn.Embedding(num_embeddings=numCategory, embedding_dim=d_model)

        self._linear = nn.Linear(d_model, d_output)

        # Positional encoding setup
        if pe == "original":
            self._generate_PE = generate_original_PE
        elif pe == "regular":
            self._generate_PE = generate_regular_PE
        else:
            self._generate_PE = None
        self._pe_period = pe_period

        self.self_condition = False

        # Time embedding setup
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        # Embedding layers for the new condition variable (emb)
        self.emb_linear = nn.Linear(self.d_emb, d_model)

        # Query function to integrate condition embeddings
        self.forQueryFunc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, emb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:
        '''
        x: The input sequence, shape [batch_size, channels, sequence_length]
        t: The time step embeddings, shape [batch_size, sequence_length]
        emb: The additional condition embedding, shape [batch_size, emb_dim]
        '''
        # print(x.shape)  # torch.Size([4, 2, 144])
        # print(emb.shape)  # torch.Size([4, 144])
        # print(x[10])
        # print(emb[10])
        # sys.exit(0)

        seqLength = x.shape[2]
        # x.shape:  [4, 2, 144] <- [batchsize, channel, length]
        xEmb = self._embedding(x.permute(0, 2, 1))  # xEmb.shape [4, 144, 256] <- [4, 144, 2] 所以其实是把channel的2变成了256

        cateEmbedding = self.embCate_layer(emb)  # [4, 144, 256]  # 还是先独立进行embedding

        # Step embeddings
        step = self.step_mlp(t)
        step = step.unsqueeze(1)  # torch.Size([4, 1, 256])
        step_emb = torch.repeat_interleave(step, seqLength, dim=1)  # Repeat across seq_length

        # Prepare for transformer encoding
        encoding = xEmb
        encoding = encoding + step_emb  # Add step embedding to input

        
        # 不是直接相加,而是要计算一个相关分数
        xQuery = self.forQueryFunc(xEmb)  # xQuery torch.Size([4, 144, 256])
        try:    
            score = torch.bmm(xQuery, cateEmbedding.transpose(1, 2))  # [4, 144, 256] * [4, 256, 144] -> [4, 144, 144]  # 计算了一个相关分数
        except:
            print('cateEmbedding.shape: ', cateEmbedding.shape)
            print('emb.shape: ', emb.shape)
            sys.exit(0)        
        score = F.softmax(score, dim=2)  # 在最后一个维度上应用softmax,其实就是value值归一化.
        condition = torch.bmm(score, cateEmbedding)  # [4, 144, 144] * [4, 144, 256] -> [4, 144, 256] 维度不变,经过加权

        # Add positional encoding if needed
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(seqLength, self.d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)  # Torch.Size([8, 64])
            encoding.add_(positional_encoding)  # Add positional encoding

        # Encoder stack
        encoding = encoding + condition  # Apply query-based conditioning with emb
        for layer in self.layers_encoding:
            # encoding = encoding + condition
            encoding = layer(encoding)  # Pass through transformer layer

        output = self._linear(encoding)  # Final output 

        return output.permute(0, 2, 1)  # Return with original shape [batch_size, seq_len, d_output]


class TransformerDenoisingQueryMulti(nn.Module):
    '''
    use x as key to query the condition embedding
    '''

    def __init__(self,
                d_input: int,
                d_model: int,
                d_output: int,
                d_emb: int,  # Embedding size for the additional input variable
                N: int,  # Number of layers in transformer
                numCategory: int,
                dropout: float = 0.1,
                pe: str = None,
                learned_sinusoidal_cond: bool = False,
                random_fourier_features: bool = False,
                learned_sinusoidal_dim: int = 16,
                pe_period: int = None,
                ):
        """Create transformer structure with query-based conditioning."""
        super().__init__()

        self.d_model = d_model
        self.d_emb = d_emb  # New condition embedding size
        self.channels = d_input
        step_dim = d_model

        self.layers_encoding = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)
            for _ in range(N)])
    
        self._embedding = nn.Linear(d_input, d_model)

        # 定义 Embedding 层
        self.embCate_layer = nn.Embedding(num_embeddings=numCategory, embedding_dim=d_model)

        self._linear = nn.Linear(d_model, d_output)

        # Positional encoding setup
        if pe == "original":
            self._generate_PE = generate_original_PE
        elif pe == "regular":
            self._generate_PE = generate_regular_PE
        else:
            self._generate_PE = None
        self._pe_period = pe_period

        self.self_condition = False

        # Time embedding setup
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, step_dim),
            nn.GELU(),
            nn.Linear(step_dim, step_dim)
        )

        # Embedding layers for the new condition variable (emb)
        self.emb_linear = nn.Linear(self.d_emb, d_model)

        # Query function to integrate condition embeddings
        self.forQueryFunc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, emb: torch.Tensor, x_self_cond: bool) -> torch.Tensor:
        '''
        x: The input sequence, shape [batch_size, channels, sequence_length]
        t: The time step embeddings, shape [batch_size, sequence_length]
        emb: The additional condition embedding, shape [batch_size, emb_dim]
        '''
        # print(x.shape)  # torch.Size([4, 2, 144])
        # print(emb.shape)  # torch.Size([4, 144])
        
        seqLength = x.shape[2]
        # x.shape:  [4, 2, 144] <- [batchsize, channel, length]
        xEmb = self._embedding(x.permute(0, 2, 1))  # xEmb.shape [4, 144, 256] <- [4, 144, 2] 所以其实是把channel的2变成了256

        cateEmbedding = self.embCate_layer(emb)  # [4, 144, 256]  # 还是先独立进行embedding

        # Step embeddings
        step = self.step_mlp(t)
        step = step.unsqueeze(1)  # torch.Size([4, 1, 256])
        step_emb = torch.repeat_interleave(step, seqLength, dim=1)  # Repeat across seq_length

        # Prepare for transformer encoding
        encoding = xEmb
        encoding = encoding + step_emb  # Add step embedding to input

        
        # 不是直接相加,而是要计算一个相关分数
        xQuery = self.forQueryFunc(xEmb)  # xQuery torch.Size([4, 144, 256])
        try:    
            score = torch.bmm(xQuery, cateEmbedding.transpose(1, 2))  # [4, 144, 256] * [4, 256, 144] -> [4, 144, 144]  # 计算了一个相关分数
        except:
            print('cateEmbedding.shape: ', cateEmbedding.shape)
            print('emb.shape: ', emb.shape)
            sys.exit(0)        
        score = F.softmax(score, dim=2)  # 在最后一个维度上应用softmax,其实就是value值归一化.
        condition = torch.bmm(score, cateEmbedding)  # [4, 144, 144] * [4, 144, 256] -> [4, 144, 256] 维度不变,经过加权

        # Add positional encoding if needed
        if self._generate_PE is not None:
            pe_params = {'period': self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(seqLength, self.d_model, **pe_params)
            positional_encoding = positional_encoding.to(encoding.device)  # Torch.Size([8, 64])
            encoding.add_(positional_encoding)  # Add positional encoding

        # Encoder stack
        # encoding = encoding + condition  # Apply query-based conditioning with emb
        for layer in self.layers_encoding:
            encoding = encoding + condition
            encoding = layer(encoding)  # Pass through transformer layer

        output = self._linear(encoding)  # Final output 

        return output.permute(0, 2, 1)  # Return with original shape [batch_size, seq_len, d_output]


