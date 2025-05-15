# blocks.py


import torch
import torch.nn as nn


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)


    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        return x


class Conv1dActNorm(nn.Module):
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            kernel_size, 
            stride=1, 
            pad="same", 
            activation="relu", 
            dropout=0.5,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_dim, out_dim, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=pad,
        )
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "leakyrelu":
            self.act = nn.LeakyReLU()
        else:
            self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        # # self.norm = nn.LayerNorm(out_dim)
        # self.norm = nn.BatchNorm1d(out_dim)


    def forward(self, x):
        x = self.conv(x)
        # x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class Conv1dBlock(nn.Module):
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            kernel_size, 
            stride=1, 
            pad="same", 
            activation="relu", 
            dropout=0.5,
    ):
        super().__init__()

        self.conv1d = Conv1dActNorm(
            in_dim, 
            out_dim, 
            kernel_size, 
            stride=stride, 
            pad=pad, 
            activation=activation, 
            dropout=dropout,
        )
        # self.norm = nn.LayerNorm(out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.use_res = in_dim == out_dim


    def forward(self, x):
        if self.use_res:
            return self.norm(self.conv1d(x) + x)
        else:
            return self.norm(self.conv1d(x))


class TransformerBlock(nn.Module):
    def __init__(self, 
            n_dim, n_heads=1, dropout=0.5
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            n_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm()
        self.ff = nn.Sequential(
            LinearNorm(n_dim, n_dim),
            nn.ReLU(),
            LinearNorm(n_dim, n_dim),
        )
        self.norm2 = nn.LayerNorm(n_dim)


    def forward(self, x):
        out1 = self.norm1(self.attn(x) + x)
        out2 = self.norm2(self.ff(out1) + out1)
        return out2


class MambaBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, x):
        pass