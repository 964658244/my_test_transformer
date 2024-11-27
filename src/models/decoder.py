import torch.nn as nn
import torch as th
from .attention import MultiHeadAttention, PositionWiseFFN


class DecoderLayer(nn.Module):
    def __init__(self, h_head, d_model, dropout):
        super(DecoderLayer).__init__()
        # define two multi_head attention layer
        self.mha = nn.ModuleList([MultiHeadAttention(h_head, d_model, dropout) for _ in range(2)])
        # define a feedforward network layer
        self.ffn = PositionWiseFFN(d_model, dropout)

    def forward(self, yz, xz, yz_look_ahead_mask, xz_pad_mask):
        # execute the first attention layer calculation, using self attention
        dec_output = self.mha[0](yz, yz, yz, yz_look_ahead_mask)
        # execute the second attention layer calculation, Q from previous attention layer, K and V from encoder output
        dec_output = self.mha[1](dec_output, xz, xz, xz_pad_mask)
        dec_output = self.ffn(dec_output)
        return dec_output


class Decoder(nn.Module):
    def __init__(self, n_head, d_model, dropout, n_layer):
        super().__init__()
        # define n_layer decoderLayer restored in ModuleList
        self.num_layers = n_layer
        self.decoder_layers = nn.ModuleList([DecoderLayer(n_head, d_model, dropout) for _ in range(n_layer)])

    def forward(self, yz, xz, yz_look_ahead_mask, xz_pad_mask):
        for decoder in self.decoder_layers:
            yz = decoder(yz, xz, yz_look_ahead_mask, xz_pad_mask)
        return yz