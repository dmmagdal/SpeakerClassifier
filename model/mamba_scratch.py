# mamba_scratch.py
# Implementation of the Mamba architecture from PeaBrane's mamba-tiny repo.
# Source: https://github.com/PeaBrane/mamba-tiny
# Windows/MacOS/Linux
# Python 3.11


"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


#######################################################################
# NOTE:
# torch JIT compiling the functions is proving to be a real headache 
# with a lot of issues in terms of what is and isnt supported between 
# JIT and regular pytorch. For now, eat the performance cost of not 
# having JIT for these functions.
#######################################################################

# @torch.jit.script
# def complex_log(input: torch.Tensor, eps: float = 1e-12):
#     eps = torch.tensor(eps, dtype=input.dtype, device=input.device)
#     real = input.abs().maximum(eps).log()
#     imag = (input < 0).to(input.dtype) * torch.pi
#     return torch.complex(real, imag)


# @torch.jit.script
# def selective_scan(u: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, mode: str = 'logcumsumexp'):
#     dA = torch.einsum('bld,dn->bldn', dt, A)
#     dB_u = torch.einsum('bld,bld,bln->bldn', dt, u, B)
#     dA = dA.clamp(min=-20)
    
#     padding =  (0, 0, 0, 0, 1, 0)
    
#     if mode == 'cumsum':            
#             dA_cumsum = F.pad(dA[:, 1:], padding).cumsum(1).exp()
#             x = dB_u / (dA_cumsum + 1e-12)
#             x = x.cumsum(1) * dA_cumsum
#             y = torch.einsum('bldn,bln->bld', x, C)
    
#     elif mode == 'logcumsumexp':  # more numerically stable (Heisen sequence)
#             dB_u_log = complex_log(dB_u)
#             dA_star = F.pad(dA[:, 1:].cumsum(1), padding)
#             x_log = torch.logcumsumexp(dB_u_log - dA_star, 1) + dA_star
#             y = torch.einsum('bldn,bln->bld', x_log.real.exp() * torch.cos(x_log.imag), C)

#     else:
#         raise ValueError(f"Expected mode argument to be either 'cumsum' or 'logcumsumexp'. Got {mode}")
            
#     return y + u * D


def complex_log(input: torch.Tensor, eps: float = 1e-12):
    eps = input.new_tensor(eps)
    real = input.abs().maximum(eps).log()
    imag = (input < 0).to(input.dtype) * torch.pi
    return torch.complex(real, imag)


def selective_scan(u, dt, A, B, C, D, mode = 'logcumsumexp'):
    dA = torch.einsum('bld,dn->bldn', dt, A)
    dB_u = torch.einsum('bld,bld,bln->bldn', dt, u, B)
    dA = dA.clamp(min=-20)
    
    padding =  (0, 0, 0, 0, 1, 0)
    
    match mode:
        case 'cumsum':            
            dA_cumsum = F.pad(dA[:, 1:], padding).cumsum(1).exp()
            x = dB_u / (dA_cumsum + 1e-12)
            x = x.cumsum(1) * dA_cumsum
            y = torch.einsum('bldn,bln->bld', x, C)
        
        case 'logcumsumexp':  # more numerically stable (Heisen sequence)
            dB_u_log = complex_log(dB_u)
            dA_star = F.pad(dA[:, 1:].cumsum(1), padding)
            x_log = torch.logcumsumexp(dB_u_log - dA_star, 1) + dA_star
            y = torch.einsum('bldn,bln->bld', x_log.real.exp() * torch.cos(x_log.imag), C)
            
    return y + u * D


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    scan_mode: str = 'cumsum'
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple
                                - self.vocab_size % self.pad_vocab_size_multiple)


class Mamba(nn.Module):
    def __init__(self, 
        d_model: int,
        n_layer: int,
        vocab_size: int,
        n_mels: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Union[int, str] = 'auto',
        d_conv: int = 4 ,
        pad_vocab_size_multiple: int = 8,
        conv_bias: bool = True,
        bias: bool = False,
        scan_mode: str = 'cumsum',
    ):
        """Full Mamba model."""
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_vocab_size_multiple = pad_vocab_size_multiple

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_model,
                    d_state,
                    expand,
                    dt_rank,
                    d_conv,
                    conv_bias,
                    bias,
                    scan_mode,
                ) 
                for _ in range(n_layer)
            ]
        )
        self.norm_f = RMSNorm(d_model)

        # self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        # self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights.
        #                                              # See "Weight Tying" paper
        self.output = nn.Linear(d_model, n_mels, bias=False)

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm_f(x)
        # return self.lm_head(x)
        return self.output(x)


class ResidualBlock(nn.Module):
    def __init__(self,        
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Union[int, str] = 'auto',
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
        scan_mode: str = 'cumsum',
    ):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.mixer = MambaBlock(
            d_model,
            d_state,
            expand,
            dt_rank, 
            d_conv,
            conv_bias,
            bias,
            scan_mode
        )
        self.norm = RMSNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
        """
        return self.mixer(self.norm(x)) + x
            

class MambaBlock(nn.Module):
    def __init__(self, 
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Union[int, str] = 'auto',
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
        scan_mode: str = 'cumsum',
    ):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.conv_bias = conv_bias
        self.bias = bias
        self.scan_mode = scan_mode

        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=self.bias
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)
        
    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        return self.out_proj(y)

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        return selective_scan(x, delta, A, B, C, D, mode=self.scan_mode)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]


class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output