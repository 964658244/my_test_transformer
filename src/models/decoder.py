import torch.nn as nn
import torch as th
from attention import MultiHeadAttention, PositionWiseFFN


class DecoderLayer(nn.Module):
    def __init__(self, h_head, d_model, dropout):
        super(DecoderLayer).__init__()
        # define two multi_head attention layer
        self.mha = nn.ModuleList([MultiHeadAttention(h_head, d_model, dropout) for _ in range(2)])
        # define a feedforward network layer
        self.ffn = PositionWiseFFN(d_model, dropout)