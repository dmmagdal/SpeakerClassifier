# blocks.py


import math

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


    def forward(self, x):
        x = self.conv(x)
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
        # TODO:
        # Re-train after swapping BatchNorm layer with LayerNorm layer
        # (could also experiment with RMSNorm layer provided by pytorch 
        # too).
        # self.norm = nn.LayerNorm(out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.use_res = in_dim == out_dim


    def forward(self, x):
        if self.use_res:
            return self.norm(self.conv1d(x) + x)
        else:
            return self.norm(self.conv1d(x))


class Conv2dActNorm(nn.Module):
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

        self.conv = nn.Conv2d(
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


    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class Conv2dBlock(nn.Module):
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

        self.conv1d = Conv2dActNorm(
            in_dim, 
            out_dim, 
            kernel_size, 
            stride=stride, 
            pad=pad, 
            activation=activation, 
            dropout=dropout,
        )
        # TODO:
        # Re-train after swapping BatchNorm layer with LayerNorm layer
        # (could also experiment with RMSNorm layer provided by pytorch 
        # too).
        # self.norm = nn.LayerNorm(out_dim)
        self.norm = nn.BatchNorm2d(out_dim)
        self.use_res = in_dim == out_dim


    def forward(self, x):
        if self.use_res:
            return self.norm(self.conv1d(x) + x)
        else:
            return self.norm(self.conv1d(x))
        

class PositionalEncodings(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()

        # Initialize positional embeddings.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )

        # Apply sinusoidal positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Expand positional encodings to shape (1, max_len, d_model)
        # (this will help with tensor multiplication in the model).
        # Register the positional encodings to the layer buffer.
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)


    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x


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