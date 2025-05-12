# mamba_tts.py
# Define Mamba text to speech model using mamba-ssm submodule 
# (state-spaces repo).
# Windows/MacOS/Linux
# Python 3.11


import torch
import torch.nn as nn
from mamba_ssm import Mamba, Mamba2


class MambaTTS(nn.Module):
    def __init__(self, 
            vocab_size, 
            n_mels=80, 
            use_mamba2=False,
            **kwargs
        ):
        super().__init__(**kwargs)

        # Parameteters.
        self.vocab_size = vocab_size
        self.n_mels = n_mels

        # Layers
        self.emb = nn.Embedding()
        self.mamba_layers = [
            Mamba2() if use_mamba2 else Mamba
        ]
        self.mamba = nn.Sequential(*self.mamba_layers)
        self.out = nn.Linear()


    def forward(x):
        pass