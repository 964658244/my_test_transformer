from torch.utils.data import Dataset
from numpy import random
import datetime
import numpy as np


class DateDataset(Dataset):
    def __init__(self, n):
        # initiate two empty list to store chinese and english date
        self.date_cn = []
        self.date_en = []
        for _ in range(n) :
            # generate year, month and day randomly
            year = random.randint(1950, 2050)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            date = datetime.date(year, month, day)
            # format the date and add it to the corresponding list
            self.date_cn.append(date.strftime("%y-%m-%d"))
            self.date_en.append(date.strftime("%d/%b/%Y"))
        # create a vocabulary set which including 0 ~ 9, -, / and month abbreviation for an english date
        self.vocab = set([str(i) for i in range(0, 10)] +
                         ["-", "/"] + [i.split("/")[1] for i in self.date_en])
        # create a lexicon to index mapping, <SOS>, <EOS>, <PAD> corresponding to START, END and PAD
        self.word2index = {v: i for i, v in enumerate(
            sorted(list(self.vocab)), start=2
        )}
        SOS_token = 0  # Start of Sentence token
        EOS_token = 1  # End of Sentence token
        PAD_token = 2  # Padding token
        self.word2index["<SOS>"] = SOS_token
        self.word2index["<EOS>"] = EOS_token
        self.word2index["<PAD>"] = PAD_token
        # add START, END and PAD tags to vocabulary set
        self.vocab.add("<SOS>")
        self.vocab.add("<EOS>")
        self.vocab.add("<EOS>")
        # create a index to lexicon mapping
        self.index2word = {i: v for v, i in self.word2index.items()}
        # initialize the input and target list
        self.input, self.target = [], []
        for cn, en in zip(self.date_cn, self.date_en):
            # transform date string to vocabulary index list, and add to input and target list
            self.input.append([self.word2index[v] for v in cn])
            self.target.append(
                [self.word2index["<SOS>"], ] +
                [self.word2index[v] for v in en[:3]] +
                [self.word2index[en[3:6]]] +
                [self.word2index[v] for v in en[6:]] +
                [self.word2index["<EOS>"], ]
            )
        # transform input and target list to Numpy array
        self.input, self.target = np.array(self.input), np.array(self.target)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index], len(self.target[index])

    @property
    def num_word(self):
        return len(self.vocab)