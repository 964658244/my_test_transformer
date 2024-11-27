import torch as th
import torch.nn as nn
from .position_embedding import *
from .encoder import *
from .decoder import *


class Transformer(nn.Module):
    def __init__(self, n_vocab, max_len, n_layer=6, emb_dim=512, n_head=8, drop_rate=0.1, padding_idx=0):
        super().__init__()
        # initiate the max length, padding_idx, vocab table size
        self.max_len = max_len
        self.padding_idx = th.tensor(padding_idx)
        self.dec_v_emb = n_vocab
        # initiate the position embedding, encoder, decoder and output layer
        self.embed = PositionEmbedding(max_len, emb_dim, n_vocab)
        self.encoder = Encoder(n_head, emb_dim, drop_rate, n_layer)
        self.decoder = Decoder(n_head, emb_dim, drop_rate, n_layer)
        self.output = nn.Linear(emb_dim, n_vocab)
        # initiate optimizer
        self.opt = th.optim.Adam(self.parameters(), lr=0.002)

    def forward(self, x, y):
        # embedding for input and target
        x_embed, y_embed = self.embed(x), self.embed(y)
        # create the padding mask code
        pad_mask = self._pad_mask(x)
        # encode the input
        encoded_x = self.encoder(x_embed, pad_mask)
        # create the look forward mask code
        yz_look_ahead_mask = self._look_ahead_mask(y)
        # transport the encoded input and look forward mask to the decoder
        decoded_x = self.decoder(y_embed, encoded_x, yz_look_ahead_mask, pad_mask)
        # get the final output by output layer
        output = self.output(decoded_x)
        return output

    def step(self, x, y):
        # clean the gradient
        self.opt.zero_grad()
        # calculate the output and loss
        logits = self(x, y[:, :-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1, self.dec_v_emb), y[:, 1:].reshape(-1))
        # do backward transport
        loss.backward()
        # update the parameter
        self.opt.step()
        return loss.cpu().data.numpy(), logits

    def _pad_bool(self, seqs):
        # create the mask code, label the padded place
        return th.eq(seqs, self.padding_idx)

    def _pad_mask(self, seqs):
        # expand the padding mask to suit dimension
        len_q = seqs.size(1)
        mask = self._pad_bool(seqs).unsqueeze(1).expand(-1, len_q, -1)
        return mask.squeeze(1)

    def _look_ahead_mask(self, seqs):
        # create the look forward mask, prevent to look the future info when generating the sequences
        device = next(self.parameters()).device
        _, seq_len = seqs.shape
        mask = th.triu(th.ones(seq_len, seq_len, dtype=th.long),
                       diagonal=1).to(device)
        mask = th.where(self._pad_bool(seqs)[:, None, None, :], 1, mask[None, None, :, :]).to(device)
        return mask > 0

def pad_zero(seqs, max_len):
    PAD_token = 0
    # initiate a total PAD_token padding 2d matrix, shape is (len(seqs), max_len)
    padded = np.full((len(seqs), max_len), fill_value=PAD_token, dtype=np.int32)
    for i, seq in enumerate(seqs):
        # filled seq from seqs to padded rows, which keep empty cell PAD_token
        padded[i, :len(seq)] = seq
    return padded