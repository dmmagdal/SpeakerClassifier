# mamba_model.py


import torch
import torch.nn as nn

from .blocks import LinearNorm
from .mamba_pytorch import ResidualBlock
    

class MambaTorchModel(nn.Module):
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
        scan_mode: str = "cumsum",
    ):
        super().__init__()

        assert n_mels <= 256, "Number of channels in mel spectrogram is too high for this model."
        assert n_layers >= 1, "Number of layers is too low for this model."

        # Input projection.
        self.input_proj = LinearNorm(n_mels, d_model)

        # Encoder.
        self.enc = nn.Sequential(*[
            ResidualBlock(
                d_model=d_model,
                d_state=d_state,
                dt_rank=dt_rank,
                d_conv=d_conv,
                conv_bias=conv_bias,
                bias=bias,
                scan_mode=scan_mode,
            )
        ])

        # Global average pooling.
        self.global_avg_pool = lambda x: torch.mean(x, dim=2)

        # Feed forward block.
        # self.ff = nn.Sequential(
        #     LinearNorm(512, 1024),
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