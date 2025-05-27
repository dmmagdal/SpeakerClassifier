# mamba_model_cuda.py


import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .blocks import LinearNorm


class MambaBlock(nn.Module):
    def __init__(self, 
        d_model: int, 
        d_state: int = 32,
        d_conv: int = 4,
        dt_rank: str = "auto",
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            dt_rank=dt_rank,
            conv_bias=conv_bias,
            bias=bias
        )
        self.norm = nn.LayerNorm(d_model)


    def forward(self, x):
        return self.norm(self.mamba(x))


class MambaModel(nn.Module):
    def __init__(self, 
        n_mels: int, 
        n_classes: int, 
        d_model: int, 
        n_layers: int = 1,
        d_state: int = 32,
        d_conv: int = 4,
        dt_rank: str = "auto",
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()

        assert n_mels <= 256, "Number of channels in mel spectrogram is too high for this model."
        assert n_layers >= 1, "Number of layers is too low for this model."

        # Input projection.
        self.input_proj = LinearNorm(n_mels, d_model)

        # Encoder.
        self.enc = nn.Sequential(*[
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                dt_rank=dt_rank,
                conv_bias=conv_bias,
                bias=bias
            ) for _ in range(n_layers)
        ])

        # Global average pooling.
        self.global_avg_pool = lambda x: torch.mean(x, dim=1)

        # Feed forward block.
        # self.ff = nn.Sequential(
        #     LinearNorm(d_model, 1024),
        #     nn.ReLU(),
        #     LinearNorm(1024, d_model),
        # )

        # Classifier.
        self.out = nn.Linear(d_model, n_classes)


    def forward(self, x):
        # Pass input to the projection layer.
        proj_out = self.input_proj(x)
        
        # Pass input to encoder.
        enc_out = self.enc(proj_out)

        # Pass encoder outputs to the global average pooling layer.
        avg_out = self.global_avg_pool(enc_out)

        # Pass the pooled outputs to the fully connected block.
        # ff_out = self.ff(avg_out)

        # Pass the outputs to the classifier layer and return the 
        # logits.
        # return self.out(ff_out)
        return self.out(avg_out)