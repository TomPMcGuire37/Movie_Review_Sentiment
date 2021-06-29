import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from torch.utils.data import DataLoader
from torchtext.data.utils import ngrams_iterator
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab 
import torchtext
from torchtext.data import get_tokenizer
from torch import nn


class MovieReviews(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.train = train 
        self.root = root_dir
        self.path = self.train_file if self.train else self.test_file
        self.reviews = pd.read_csv(self.path, sep="\t")
        self.transform = transform
    
    @property
    def test_file(self):
        return os.path.join(self.root, "test.tsv")
    
    @property
    def train_file(self):
        return os.path.join(self.root, "train.tsv")
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        phrase = self.reviews.iloc[idx, 2]
        if self.transform:
            phrase = self.transform(phrase)
        
        if self.train:
            y = self.reviews.iloc[idx, 3]
            return phrase, y
        else:
            return phrase



def to_device(data, device=None):
    """Move tensor(s) to chosen device"""
    if not device:
        device = _device

    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """
    Wrap a dataloader to move data to a device
    """
    def __init__(self, dl, device=None):
        self.dl = dl
        if device:
            device = device
        else:
            device = _device

        self.device = device

    def __iter__(self):
        """
        Yield a batch of data after moving it to device
        """
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """
        Number of batches
        """
        return len(self.dl)


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)



class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, hidden_dim=3):
        super(TextLSTM, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.LSTM = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=1, 
                    batch_first=True, padding_idx=0)
        self.linear1 = nn.Linear(hidden_dim, num_class)
        # self.batch_norm1 = nn.BatchNorm1d(128)
        # self.linear2 = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(0.4)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        # self.linear2.weight.data.uniform_(-initrange, initrange)
        # self.linear2.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = self.dropout(embedded)
        lstm_out, (ht, ct) = self.LSTM(x)
        # x = self.linear1(embedded)
        # x = self.batch_norm1(x)
        return self.linear1(ht)