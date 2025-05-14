# tacomamba.py
# Implementation of the higher text-to-speech model that leverages the 
# Mamba blocks from the mamba_scratch.py module containing the Mamba 
# architecture from PeaBrane's mamba-tiny repo.
# Windows/MacOS/Linux
# Python 3.11


from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_scratch import ResidualBlock, RMSNorm


# @torch.jit.script
def monotonic_alignment_search(attn_map: torch.Tensor) -> torch.Tensor:
    """
    Implements hard monotonic alignment search.
    @param: attn_map (torch.Tensor), a tensor of shape 
        (B, T_dec, T_enc) attention log-probability or score matrix.
    @return: returns an alignments tensor of shape (B, T_dec, T_enc) 
        hard 0-1 alignment matrix.
    """
    B, T_dec, T_enc = attn_map.size()
    alignments = torch.zeros_like(attn_map)
    
    for b in range(B):
        t_enc = 0
        for t_dec in range(T_dec):
            best = torch.argmax(attn_map[b, t_dec, t_enc:])
            t_enc += best.item()
            if t_enc >= T_enc:
                break
            alignments[b, t_dec, t_enc] = 1.0
    return alignments


#######################################################################
# First implementation of expand() (non-parallelized)
#######################################################################

# # @torch.jit.script
# def expand(encoder_outputs, durations):
#     """
#     Expands encoder outputs according to predicted durations.

#     Args:
#         encoder_outputs: Tensor of shape [B, T_enc, D]
#         durations: Tensor of shape [B, T_enc] (int durations per token)

#     Returns:
#         expanded_outputs: Tensor of shape [B, T_dec, D], where T_dec = max sum(durations)
#         masks: Bool tensor of shape [B, T_dec] indicating valid positions
#     """
#     B, T_enc, D = encoder_outputs.size()
#     device = encoder_outputs.device
#     durations = durations.clamp(min=0).round().long()  # safety

#     # Calculate max expanded length for padding
#     T_dec = durations.sum(dim=1).max().item()

#     expanded_outputs = []
#     masks = []

#     for b in range(B):
#         enc_out = encoder_outputs[b]  # [T_enc, D]
#         dur = durations[b]            # [T_enc]
        
#         expanded = [enc_out[i].unsqueeze(0).expand(dur[i], -1) for i in range(T_enc) if dur[i] > 0]
#         if len(expanded) > 0:
#             expanded = torch.cat(expanded, dim=0)  # [T_dec_b, D]
#         else:
#             expanded = torch.zeros(1, D, device=device)
        
#         pad_len = T_dec - expanded.size(0)
#         if pad_len > 0:
#             expanded = F.pad(expanded, (0, 0, 0, pad_len))  # pad in time dim
        
#         expanded_outputs.append(expanded.unsqueeze(0))  # [1, T_dec, D]
#         mask = torch.zeros(T_dec, dtype=torch.bool, device=device)
#         mask[:expanded.size(0)-pad_len] = 1
#         masks.append(mask.unsqueeze(0))  # [1, T_dec]

#     expanded_outputs = torch.cat(expanded_outputs, dim=0)  # [B, T_dec, D]
#     masks = torch.cat(masks, dim=0)  # [B, T_dec]
#     return expanded_outputs, masks


#######################################################################
# New implementation of expand() (parallelized)
#######################################################################

# @torch.jit.script
def expand(encoder_outputs: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """
    Expands encoder outputs according to predicted durations.
    """
    B, T_enc, D = encoder_outputs.size()
    device = encoder_outputs.device
    durations = durations.clamp(min=0).round().long()  # safety

    # Calculate max expanded length for padding
    T_dec = durations.sum(dim=1).max().item()

    # Create an empty tensor for expanded outputs
    expanded_outputs = torch.zeros(B, T_dec, D, device=device)
    masks = torch.zeros(B, T_dec, dtype=torch.bool, device=device)

    for b in range(B):
        enc_out = encoder_outputs[b]  # [T_enc, D]
        dur = durations[b]            # [T_enc]
        
        expanded = [enc_out[i].unsqueeze(0).expand(dur[i], -1) for i in range(T_enc) if dur[i] > 0]
        if len(expanded) > 0:
            expanded = torch.cat(expanded, dim=0)  # [T_dec_b, D]
        else:
            expanded = torch.zeros(1, D, device=device)
        
        pad_len = T_dec - expanded.size(0)
        if pad_len > 0:
            expanded = F.pad(expanded, (0, 0, 0, pad_len))  # pad in time dim
        
        expanded_outputs[b, :expanded.size(0), :] = expanded
        masks[b, :expanded.size(0)-pad_len] = 1

    return expanded_outputs, masks


class MelEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transform = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # self.transform = nn.Linear(in_channels, out_channels)


    def forward(self, x):
        """
        Perform a transformation on the input mel spectrogram sequence 
            to transform the feature dimensions into some shape that is
            compatible with the encoder output for computing the 
            similarity score tensor required for monotonic alignment.
        @param: x (torch.Tensor), mel spectrogram in the shape of 
            (B, T_mel, n_mels).
        @return: returns the transformed tensor in the shape of 
            (B, T_mel, h_dim).
        """
        # NOTE: Perform the transpose of the second and third dimenions 
        # for the conv1d layer. Conv1d expects inputs in the form of 
        # (B, channel, seq_len).
        x = x.transpose(1, 2)
        x = self.transform(x)
        x = x.transpose(1, 2)
        return x
    

class TextEncoder(nn.Module):
    def __init__(self, 
        d_model: int,
        n_layer: int,
        vocab_size: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Union[int, str] = 'auto',
        d_conv: int = 4 ,
        pad_vocab_size_multiple: int = 8,
        conv_bias: bool = True,
        bias: bool = False,
        scan_mode: str = 'cumsum',
    ):
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


    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        return x


class DurationPredictor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.LayerNorm(d_model),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, 1, kernel_size=1)
        )


    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.layers(x)
        return out.squeeze(1)
    

class MelDecoder(nn.Module):
    def __init__(self, 
        d_model: int,
        n_layer: int,
        n_mels: int,
        d_state: int = 16,
        expand: int = 2,
        dt_rank: Union[int, str] = 'auto',
        d_conv: int = 4,
        conv_bias: bool = True,
        bias: bool = False,
        scan_mode: str = 'cumsum',
    ):
        super().__init__()

        # self.input_proj = nn.Linear(n_mels, d_model)
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
        self.output = nn.Linear(d_model, n_mels, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        return self.output(x)


class TacoMamba(nn.Module):
    def __init__(self, 
        d_model: int,
        n_enc_layer: int,
        n_dec_layer: int,
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
        super().__init__()
        self.encoder = TextEncoder(
            d_model,
            n_enc_layer,
            vocab_size,
            d_state,
            expand,
            dt_rank,
            d_conv,
            pad_vocab_size_multiple,
            conv_bias,
            bias,
            scan_mode
        )
        self.mel_projection = MelEncoder(n_mels, d_model)
        self.duration_predictor = DurationPredictor(d_model)
        self.decoder = MelDecoder(
            d_model,
            n_dec_layer,
            n_mels,
            d_state,
            expand,
            dt_rank,
            d_conv,
            conv_bias,
            bias,
            scan_mode,
        )

    def forward(self, text):
        # Encode the text.
        enc_out = self.encoder(text)

        # Compute the predicted durations via the duration prediction 
        # module.
        durations = self.duration_predictor(enc_out)

        # Expand encoder outputs using the durations.
        dec_in, mask = expand(enc_out, durations)

        # Pass the expanded encoder outputs to the decoder to predict 
        # the mel spectrogram signal.
        mel_pred = self.decoder(dec_in)

        return mel_pred[mask]


    def train_step(self, text, mel):
        # Encode the text.
        enc_out = self.encoder(text)

        # Project the mel spectrogram to the same dimension as the text 
        # encoder output.
        mel_proj = self.mel_projection(mel)

        # Similarity across mel spectrograms and encoder output. Uses
        # dot product similarity.
        sim = torch.bmm(mel_proj, enc_out.transpose(1, 2))

        # Compute durations via monotonic alignment search.
        with torch.no_grad():
            mas_durations = monotonic_alignment_search(sim)
            mas_durations = mas_durations.sum(dim=1)

        # Compute the predicted durations via the duration prediction 
        # module.
        pred_durations = self.duration_predictor(enc_out)

        # Expand encoder outputs using the durations.
        dec_in, mask = expand(enc_out, mas_durations)

        # Pass the expanded encoder outputs to the decoder to predict 
        # the mel spectrogram signal.
        mel_pred = self.decoder(dec_in)

        # Return the predicted durations, mask, and mel spectrogram.
        return mas_durations.float(), pred_durations, mask, mel_pred


# from mamba_seq2seq import MambaEncoder

# class MonotonicAttention(nn.Module):
#     def __init__(self, query_dim, key_dim, attn_dim):
#         super().__init__()
#         self.query_proj = nn.Linear(query_dim, attn_dim)
#         self.key_proj = nn.Linear(key_dim, attn_dim)
#         self.energy_proj = nn.Linear(attn_dim, 1)

#     def forward(self, query, keys, prev_attention=None, mask=None):
#         """
#         query: (B, query_dim)
#         keys: (B, T_enc, key_dim)
#         prev_attention: (B, T_enc), previous timestep attention weights (for monotonic constraint)
#         """
#         B, T_enc, _ = keys.size()

#         query_proj = self.query_proj(query).unsqueeze(1)     # (B, 1, attn_dim)
#         keys_proj = self.key_proj(keys)                      # (B, T_enc, attn_dim)
#         energies = self.energy_proj(torch.tanh(query_proj + keys_proj)).squeeze(-1)  # (B, T_enc)

#         # Monotonic constraint with sigmoid
#         p_select = torch.sigmoid(energies)  # (B, T_enc)

#         # Soft attention approximation (similar to MoChA)
#         if prev_attention is None:
#             # First timestep, attend only to first token
#             prev_attention = F.one_hot(torch.zeros(B, dtype=torch.long, device=query.device), T_enc).float()

#         # Compute attention weights
#         attention = p_select * prev_attention + 1e-8  # smooth
#         attention = attention / attention.sum(dim=1, keepdim=True)

#         if mask is not None:
#             attention = attention.masked_fill(mask, 0.0)
#             attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)

#         context = torch.bmm(attention.unsqueeze(1), keys).squeeze(1)  # (B, key_dim)

#         return context, attention


# class DynamicConvAttention(nn.Module):
#     def __init__(self, query_dim, key_dim, conv_kernel_size=5, conv_out_channels=64):
#         super().__init__()
#         self.query_proj = nn.Linear(query_dim, conv_out_channels * conv_kernel_size)
#         self.key_proj = nn.Linear(key_dim, conv_out_channels)
#         self.kernel_size = conv_kernel_size
#         self.out_proj = nn.Linear(conv_out_channels, key_dim)

#     def forward(self, query, keys, mask=None):
#         """
#         query: (B, query_dim)
#         keys: (B, T_enc, key_dim)
#         """
#         B, T_enc, _ = keys.size()
#         dynamic_filters = self.query_proj(query)  # (B, conv_out_channels * kernel_size)
#         dynamic_filters = dynamic_filters.view(B, -1, self.kernel_size)  # (B, C_out, K)

#         keys_proj = self.key_proj(keys).transpose(1, 2)  # (B, C_out, T_enc)
#         keys_proj_padded = F.pad(keys_proj, (self.kernel_size // 2, self.kernel_size // 2), mode="reflect")

#         # Dynamic conv: for each batch, convolve with its own filter
#         output = torch.zeros(B, keys_proj.size(1), T_enc, device=keys.device)
#         for b in range(B):
#             output[b] = F.conv1d(keys_proj_padded[b:b+1], dynamic_filters[b:b+1], groups=keys_proj.size(1))

#         output = output.transpose(1, 2)  # (B, T_enc, C_out)
#         context = self.out_proj(output)  # (B, T_enc, key_dim)
#         attention_weights = F.softmax(context.sum(dim=-1), dim=-1)  # (B, T_enc)
#         weighted_context = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)

#         return weighted_context, attention_weights
