# transformer_model.py


import torch
import torch.nn as nn

from .blocks import LinearNorm, PositionalEncodings


class TransformerModel(nn.Module):
    def __init__(self, 
        n_mels: int, 
        n_classes: int, 
        d_model: int, 
        n_layers: int = 1,
        n_heads: int = 1,
        max_len: int = 2048
    ):
        super().__init__()

        assert n_mels <= 256, "Number of channels in mel spectrogram is too high for this model."
        assert n_layers >= 1, "Number of layers is too low for this model."

        # Input projection and embedding.
        self.input_proj = LinearNorm(n_mels, d_model)
        self.embedding = PositionalEncodings(d_model, max_len)

        # Encoder.
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_heads, d_model, batch_first=True
            ),
            n_layers
        )

        # Global average pooling.
        self.global_avg_pool = lambda x: torch.mean(x, dim=1)

        # Feed forward block.
        # self.ff = nn.Sequential(
        #     LinearNorm(d_model, 1024),
        #     nn.ReLU(),
        #     LinearNorm(1024, d_model),
        # ) # Old. Was considered to have this FFN component before classifier but number of parameters (& VRAM usage) was already high without it.

        # Classifier.
        self.out = nn.Linear(d_model, n_classes)


    def forward(self, x, mask=None):
        # Pass input to the projection and positional embedding layers.
        proj_out = self.input_proj(x)
        emb_out = self.embedding(proj_out)

        # Pass embedding outputs to encoder.
        enc_out = self.enc(emb_out, src_key_padding_mask=mask)

        # Pass encoder outputs to the global average pooling layer.
        if mask is not None:
            avg_out = self.global_avg_pool(enc_out)
        else:
            # Set padded positions to 0.
            mask = (~mask).unsqueeze(-1).float()  # (batch_size, time_steps, 1)
            enc_out = enc_out * mask
            lengths = mask.sum(dim=1)
            avg_out = enc_out.sum(dim=1) / lengths.clamp(min=1e-6)

        # Pass the pooled outputs to the fully connected block.
        # ff_out = self.ff(avg_out)

        # Pass the outputs to the classifier layer and return the 
        # logits.
        # return self.out(ff_out)
        return self.out(avg_out)