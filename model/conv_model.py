# conv_model.py


import torch
import torch.nn as nn

from .blocks import Conv2dBlock, Conv1dBlock, LinearNorm


class Conv1DModel(nn.Module):
    def __init__(
            self,
            n_mels: int,
            n_classes: int,
            d_model: int,
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
            LinearNorm(1024, d_model),
        )

        # Classifier.
        self.out = nn.Linear(d_model, n_classes)


    def forward(self, x):
        # Transpose (B, L, n_mels) to (B, n_mels, L) to allow for 
        # passing to Conv1d layers in the encoder.
        x = x.transpose(1, 2)

        # Pass input to encoder.
        enc_out = self.enc(x)

        # Pass encoder outputs to the global average pooling layer.
        avg_out = self.global_avg_pool(enc_out)

        # Pass the pooled outputs to the fully connected block.
        ff_out = self.ff(avg_out)

        # Pass the outputs to the classifier layer and return the 
        # logits.
        return self.out(ff_out)
    


class Conv2DModel(nn.Module):
    def __init__(
            self,
            n_mels: int,
            n_classes: int,
            d_model: int,
    ):
        super().__init__()

        assert n_mels <= 256, "Number of channels in mel spectrogram is too high for this model."

        # NOTE:
        # In the self.enc (encoder block), there are two possible 
        # starting convolutions. One has n_mels set for the 
        # filters argument and another has 1 for the same argument. Use 
        # the former for when we want to have n_mels dim as a channel 
        # feature rather than as part of the "image" features (shape is
        # (B, n_mels, L, 1)). Use the latter for when we want to treat
        # the mel spectrogram samples as images (shape is 
        # (B, 1, L, n_mels)).

        # Encoder.
        self.enc = nn.Sequential(
            Conv2dBlock(n_mels, 128, 3),
            Conv2dBlock(128, 128, 3),
            Conv2dBlock(128, 256, 3),
            Conv2dBlock(256, 512, 3),
        )

        # Global average pooling.
        self.global_avg_pool = lambda x: torch.mean(x, dim=(2, 3))

        # Feed forward block.
        self.ff = nn.Sequential(
            LinearNorm(512, 1024),
            nn.ReLU(),
            LinearNorm(1024, d_model),
        )

        # Classifier.
        self.out = nn.Linear(d_model, n_classes)


    def forward(self, x):
        # Transpose (B, L, n_mels) to (B, n_mels, L) to allow for 
        # passing to Conv1d layers in the encoder.
        x = x.transpose(1, 2)

        # Expand the inputs from (B, n_mels, L) to (B, n_mels, L, 1)
        # to allow for passing to Conv2d layers in the encoder.
        x = x.unsqueeze(-1)

        # NOTE:
        # Was originally going to have x in the shape of 
        # (B, 1, n_mels, L) where 1 is the channel but the model was
        # hitting OOM in the encoder. So now n_mels is the channel 
        # (shape is (B, n_mels, L, 1)).

        # Pass input to encoder.
        enc_out = self.enc(x)

        # Pass encoder outputs to the global average pooling layer.
        avg_out = self.global_avg_pool(enc_out)

        # Pass the pooled outputs to the fully connected block.
        ff_out = self.ff(avg_out)

        # Pass the outputs to the classifier layer and return the 
        # logits.
        return self.out(ff_out)