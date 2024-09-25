import numpy as np
import torch as th
from torch import nn

class MultiHeadAttention(nn.module):
    def __init__(self, n_head, d_model, drop_rate=0.1):
        super().__init__()
        # the dimensions of each head
        self.head_dim = d_model // n_head
        # the number of heads
        self.n_head = n_head
        # the dimensions of the model
        self.d_model = d_model
        # the linear transform layer, used to generate query, key and value
        self.wq = nn.Linear(d_model, n_head * self.head_dim)
        self.wk = nn.Linear(d_model, n_head * self.head_dim)
        self.wv = nn.Linear(d_model, n_head * self.head_dim)
        # the dense layer for outputs
        self.output_dense = nn.Linear(d_model, d_model)
        # dropout layer
        self.output_drop = nn.Dropout(drop_rate)
        # layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = None

    def forward(self, q, k, v, mask=None):
        # save q for residual
        residual = q
        # do linear transform for q, k ,v separately
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)
        # do head split for query, key and value
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        # calculate the context vector
        context = self.scaled_dot_product_attention(q, k, v, mask)
        # linear transform for the context vector
        output = self.output_dense(context)
        # add dropout
        output = self.output_drop(output)
        # add Layer normalization
        output = self.layer_norm(output + residual)
        return output

    def split_heads(self, x):
        # change the input x shape to (n, step, n_head, head_dim), then reshape to (n, n_head, step, head_dim)
        x = th.reshape(x, (x.shape[0], x.shape[1], self.n_head, self.head_dim))
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # calculate dk
        dk = th.tensor(k.shape[-1]).type(th.float)
        # calculate the attention score
        score = th.matmul(q, k.permute(0, 1, 3, 2)) / (th.sqrt(dk) + 1e-8)
        if mask is not None:
            # if a mask is given, then set the masked position value to -np.inf
            # to ensure the respective softmax values become 0
            score = score.masked_fill_(mask, -np.inf)
        # apply the softmax to calculate the attention weight
        self.attention = th.softmax(score, dim=-1)
        # calculate the context vector
        context = th.matmul(self.attention, v)
        # reshape the context dimension and do dimension combination
        context = context.permute(0, 2, 1, 3)
        context = context.reshape((context.shape[0], context.shape[1], -1))
        return context
    