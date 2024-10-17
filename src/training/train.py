import torch as th
import torch.nn as nn
from src.models.transformer import *
from src.data.date_dataset import DateDataset
from torch.utils.data import DataLoader


# initiate a transformer model, set vocab table size, max sequence length, layer numbers, embedding dimension,
# heads number, dropout rate and padding index
dataset = DateDataset(1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = Transformer(n_vocab=dataset.num_word, max_len=MAX_LENGTH, n_layer=3, emb_dim=32, n_head=8, drop_rate=0.1, padding_idx=0)
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = model.to(device)
for i in range(10):
    # 