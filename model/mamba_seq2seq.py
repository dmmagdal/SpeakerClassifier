# mamba_seq2seq.py
# Implementation of the higher text-to-speech model that leverages the 
# Mamba blocks from the mamba_scratch.py module containing the Mamba 
# architecture from PeaBrane's mamba-tiny repo.
# Windows/MacOS/Linux
# Python 3.11


from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_scratch import *


class MambaTTS(nn.Module):
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

        self.encoder = MambaEncoder(
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
        # self.decoder = DecoderLinearCrossAttn(
        #     d_model,
        #     n_dec_layer,
        #     n_mels,
        #     d_state,
        #     expand,
        #     dt_rank,
        #     d_conv,
        #     conv_bias,
        #     bias,
        #     scan_mode,
        # )
        self.decoder = DecoderCrossAttn(
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
        # self.decoder = DecoderSingleCrossAttn(
        #     d_model,
        #     n_dec_layer,
        #     n_mels,
        #     d_state,
        #     expand,
        #     dt_rank,
        #     d_conv,
        #     conv_bias,
        #     bias,
        #     scan_mode,
        # )
        # self.decoder = DecoderSoftAttn(
        #     d_model,
        #     n_dec_layer,
        #     n_mels,
        #     d_state,
        #     expand,
        #     dt_rank,
        #     d_conv,
        #     conv_bias,
        #     bias,
        #     scan_mode,
        # )


    def forward(self, input_ids, mels):
        enc_outputs = self.encoder(input_ids)
        return self.decoder(mels, enc_outputs)
    

class DecoderLinearCrossAttn(nn.Module):
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

        # ##############################################################
        # NOTE:
        # Still OOMs even on lower parameters (3 - 4M)
        # ##############################################################

        self.input_proj = nn.Linear(n_mels, d_model)
        self.mamba_layers = nn.ModuleList(
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
                for _ in range(n_layer // 2)
            ]
        )
        self.attn_layers = nn.ModuleList(
            [
                LinearCrossAttnBlock(d_model)
                for _ in range(n_layer // 2)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.output = nn.Linear(d_model, n_mels, bias=False)

    def forward(self, x, enc_output):
        x = self.input_proj(x)
        
        for mamba, attn in zip(self.mamba_layers, self.attn_layers):
            x = mamba(x)
            x = attn(x, enc_output, enc_output)
        
        x = self.norm_f(x)
        return self.output(x)


class DecoderCrossAttn(nn.Module):
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
        n_heads: int = 4,
    ):
        super().__init__()

        # ##############################################################
        # NOTE:
        # Still OOMs even on lower parameters (3 - 4M)
        # ##############################################################

        self.input_proj = nn.Linear(n_mels, d_model)
        self.mamba_layers = nn.ModuleList(
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
                for _ in range(n_layer // 2)
            ]
        )
        self.attn_layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model, n_heads
                )
                for _ in range(n_layer // 2)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.output = nn.Linear(d_model, n_mels, bias=False)

    def forward(self, x, enc_output):
        x = self.input_proj(x)
        
        for mamba, attn in zip(self.mamba_layers, self.attn_layers):
            x = mamba(x)
            x = attn(x, enc_output, enc_output)
        
        x = self.norm_f(x)
        return self.output(x)


class DecoderSingleCrossAttn(nn.Module):
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
        n_heads: int = 1,
    ):
        super().__init__()

        ###############################################################
        # NOTE:
        # Still OOMs even on lower parameters (3 - 4M)
        ###############################################################

        self.input_proj = nn.Linear(n_mels, d_model)
        self.mamba_prenet = nn.Sequential(
            *[
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
                for _ in range(n_layer // 2)
            ]
        )
        self.attn = CrossAttentionBlock(
            d_model, n_heads
        )
        self.mamba_postnet = nn.Sequential(
            *[
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
                for _ in range(n_layer // 2)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.output = nn.Linear(d_model, n_mels, bias=False)

    def forward(self, x, enc_output):
        x = self.input_proj(x)

        x = self.mamba_prenet(x)
        x = self.attn(x, enc_output, enc_output)
        x = self.mamba_postnet(x)
        
        x = self.norm_f(x)
        return self.output(x)


class DecoderSoftAttn(nn.Module):
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

        ###############################################################
        # NOTE:
        # OOMs on 3M parameters (3+ layers). At 2M parameters AND with
        # lower batch size (8), would get nan loss after first epoch.
        # Using logcumsumexp did provide a bit better stability over
        # cumsum (for the mamba module) but still got NaN loss at end
        # of first epoch.
        ###############################################################

        self.input_proj = nn.Linear(n_mels, d_model)
        self.mamba_prenet = nn.Sequential(
            *[
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
                for _ in range(n_layer // 2)
            ]
        )
        self.attn = SoftAttention()
        self.mamba_postnet = nn.Sequential(
            *[
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
                for _ in range(n_layer // 2)
            ]
        )
        self.norm_f = RMSNorm(d_model)
        self.output = nn.Linear(d_model, n_mels, bias=False)

    def forward(self, x, enc_output):
        x = self.input_proj(x)

        x = self.mamba_prenet(x)
        x = self.attn(x, enc_output)
        x = self.mamba_postnet(x)
        
        x = self.norm_f(x)
        return self.output(x)


class SoftAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, decoder_input, encoder_output):
        """
        Perform soft attention from decoder_inpu over encoder_output.

        decoder_input: (B, L2, h_dim) - target
        encoder_output: (B, L1, h_dim) - source
        Returns:
            attended_B: (B, L2, h_dim) - attention result where each decoder_input_i attends over encoder_output
        """
        # Compute attention scores: each decoder_input_i attends over encoder_output_j
        attention_scores = torch.bmm(
            decoder_input, encoder_output.transpose(1, 2)
        )  # (B, L2, L1)

        # Get attention weights
        attn_weights = F.softmax(attention_scores, dim=-1)  # (B, L2, L1)

        # Apply attention weights to encoder_output
        attended_B = torch.bmm(attn_weights, encoder_output)  # (B, L2, h_dim)

        return attended_B
    

class LinearCrossAttnBlock(nn.Module):
    def __init__(self, d_model, feature_map=F.elu):
        super().__init__()
        self.attn = LinearCrossAttention(feature_map)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        # query: (batch, tgt_seq, dim), key/value: (batch, src_seq, dim)
        attn_output = self.attn(query, key, value)
        out = self.norm(query + self.dropout(attn_output))
        return out


class LinearCrossAttention(nn.Module):
        def __init__(self, feature_map=F.elu):
            super().__init__()
            self.feature_map = feature_map  # The feature map function (ELU by default)
        
        def forward(self, Q, K, V):
            """
            Q: (B, Q_len, D) - query tensor (decoder input: mel spectrograms or previous predictions)
            K: (B, K_len, D) - key tensor (encoder output: text embeddings)
            V: (B, K_len, D) - value tensor (same as K for this attention mechanism)
            """
            # Apply feature map to query and key
            Q_phi = self.feature_map(Q)  # (B, Q_len, D)
            K_phi = self.feature_map(K)  # (B, K_len, D)

            # Precompute context: (B, D, D)
            KV = torch.einsum('bnd,bne->bde', K_phi, V)

            # Normalize denominator: (B, Q_len, 1)
            Z = 1 / (torch.einsum('bnd,bd->bn', Q_phi, K_phi.sum(dim=1)) + 1e-6).unsqueeze(-1)

            # Compute output: (B, Q_len, D)
            out = torch.einsum('bnd,bde->bne', Q_phi, KV)

            return out * Z


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        # query: (batch, tgt_seq, dim), key/value: (batch, src_seq, dim)
        attn_output, _ = self.attn(query, key, value)
        out = self.norm(query + self.dropout(attn_output))
        return out


class MambaDecoder(nn.Module):
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

        self.input_proj = nn.Linear(n_mels, d_model)
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
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        return self.output(x)


class MambaEncoder(nn.Module):
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