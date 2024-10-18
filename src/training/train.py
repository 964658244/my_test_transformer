import torch as th
import torch.nn as nn
from src.models.transformer import *
from src.data.date_dataset import DateDataset
from torch.utils.data import DataLoader


n_epochs = 100

# initiate a transformer model, set vocab table size, max sequence length, layer numbers, embedding dimension,
# heads number, dropout rate and padding index
dataset = DateDataset(1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = Transformer(n_vocab=dataset.num_word, max_len=MAX_LENGTH, n_layer=3, emb_dim=32, n_head=8, drop_rate=0.1, padding_idx=0)
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = model.to(device)
for i in range(10):
    # zero padding for each input and target batch data in Dataloader, makes it's length max, then transform to tensor
    for input_tensor, target_tensor, _ in dataloader:
        input_tensor = th.from_numpy(
            pad_zero(input_tensor, max_len=MAX_LENGTH)
        ).long().to(device)
        target_tensor = th.from_numpy(
            pad_zero(target_tensor, max_len=MAX_LENGTH+1)
        ).long().to(device)
        loss, _ = model.step(input_tensor, target_tensor)
    print(f"epoch: {i+1}, \tloss: {loss:}")