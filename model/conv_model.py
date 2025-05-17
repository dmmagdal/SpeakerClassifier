# conv_model.py


import torch
import torch.nn as nn

from .blocks import Conv1dBlock, LinearNorm


class Conv1DModel(nn.Module):
    def __init__(
            self,
            n_mels,
            n_classes,
            # kernel_size, 
            # stride=1, 
            # pad=0, 
            # activation="relu", 
            # dropout=0.5,  
            # n_layers=1, 
    ):
        super().__init__()

        assert n_mels <= 256, "Number of channels in mel spectrogram is too high for this model."

        # Prenet

        # Encoder
        self.enc = nn.Sequential(
            Conv1dBlock(n_mels, 128, 3),
            Conv1dBlock(128, 128, 3),
            Conv1dBlock(128, 256, 3),

            # Conv1dBlock(256, 256, 3),
            # Conv1dBlock(256, 256, 3),

            # Conv1dBlock(256, 256, 3),
            # Conv1dBlock(256, 256, 3),

            Conv1dBlock(256, 512, 3),
            # Conv1dBlock(256, 512, 3),
            # Conv1dBlock(512, 512, 3),
            # Conv1dBlock(512, 1024, 3),
        )

        self.global_avg_pool = lambda x: torch.mean(x, dim=2)

        self.ff = nn.Sequential(
            # LinearNorm(1024, 4096),
            # nn.ReLU(),
            # LinearNorm(4096, 4096),
            # nn.ReLU(),
            # LinearNorm(4096, 1024),
            # nn.ReLU(),
            # LinearNorm(1024, 512),
            #
            # LinearNorm(512, 1024),
            # nn.ReLU(),
            # LinearNorm(1024, 1024),
            # nn.ReLU(),
            # LinearNorm(1024, 512),
            # 
            LinearNorm(512, 1024),
            nn.ReLU(),
            LinearNorm(1024, 512),
        )

        # Classifier
        self.out = nn.Linear(512, n_classes)


    def forward(self, x):
        x = x.transpose(1, 2)
        enc_out = self.enc(x)
        avg_out = self.global_avg_pool(enc_out)
        ff_out = self.ff(avg_out)
        return self.out(ff_out)