import torch.nn as nn
import numpy as np
import torch as th


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, emb_dim, n_vocab):
        super().__init__()
        # generate the (Sinusoidal) position embedding matrix
        pos = np.expand_dims(np.arange(max_len), 1)
        pe = pos / np.power(10000, 2 * np.expand_dims(np.arange(emb_dim) // 2, 0) / emb_dim)
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = np.expand_dims(pe, 0)
        self.pe = th.from_numpy(pe).type(th.float32)

        # define the embedding layer
        self.embeddings = nn.Embedding(n_vocab, emb_dim)
        # initiate vocab embedding layer weight
        self.embeddings.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # makesure position embedding and vocab weight as the same device
        device = self.embeddings.weight.device
        self.pe = self.pe.to(device)
        # calculate the input vocab weight, add the position embedding
        x_embed = self.embeddings(x) + self.pe
        return x_embed
