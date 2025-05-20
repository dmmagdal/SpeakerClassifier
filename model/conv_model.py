# conv_model.py


import torch
import torch.nn as nn

from .blocks import Conv2dBlock, Conv1dBlock, LinearNorm


class Conv1DModel(nn.Module):
    def __init__(
            self,
            n_mels,
            n_classes,
    ):
        super().__init__()

        assert n_mels <= 256, "Number of channels in mel spectrogram is too high for this model."

        # Encoder.
        self.enc = nn.Sequential(
            Conv1dBlock(n_mels, 128, 3),
            Conv1dBlock(128, 128, 3),
            Conv1dBlock(128, 256, 3),
            Conv1dBlock(256, 512, 3),
        )

        # Global average pooling.
        self.global_avg_pool = lambda x: torch.mean(x, dim=2)

        # Feed forward block.
        self.ff = nn.Sequential(
            LinearNorm(512, 1024),
            nn.ReLU(),
            LinearNorm(1024, 512),
        )

        # Classifier.
        self.out = nn.Linear(512, n_classes)


    def forward(self, x):
        # Transpose (B, L, n_mels) to (B, n_mels, L) to allow for 
        # passing to Conv1d layers in the encoder.
        x = x.transpose(1, 2)

        # Pass input to encoder.
        enc_out = self.enc(x)

        # Pass encouder outputs to the global average pooling layer.
        avg_out = self.global_avg_pool(enc_out)

        # Pass the pooled outputs to the fully connected block.
        ff_out = self.ff(avg_out)

        # Pass the outputs to the classifier layer and return the 
        # logits.
        return self.out(ff_out)
    


class Conv2DModel(nn.Module):
    def __init__(
            self,
            n_mels,
            n_classes,
    ):
        super().__init__()

        assert n_mels <= 256, "Number of channels in mel spectrogram is too high for this model."

        # Encoder.
        self.enc = nn.Sequential(
            Conv2dBlock(n_mels, 128, 3),
            Conv2dBlock(128, 128, 3),
            Conv2dBlock(128, 256, 3),
            Conv2dBlock(256, 512, 3),
        )

        # Global average pooling.
        self.global_avg_pool = lambda x: torch.mean(x, dim=2)

        # Feed forward block.
        self.ff = nn.Sequential(
            LinearNorm(512, 1024),
            nn.ReLU(),
            LinearNorm(1024, 512),
        )

        # Classifier.
        self.out = nn.Linear(512, n_classes)


    def forward(self, x):
        # Expand the inputs from (B, L, n_mels) to (B, 1, L, n_mels)
        # to allow for passing to Conv2d layers in the encoder.
        x = x.unsqueeze(1)

        # Pass input to encoder.
        enc_out = self.enc(x)

        # Pass encouder outputs to the global average pooling layer.
        avg_out = self.global_avg_pool(enc_out)

        # Pass the pooled outputs to the fully connected block.
        ff_out = self.ff(avg_out)

        # Pass the outputs to the classifier layer and return the 
        # logits.
        return self.out(ff_out)