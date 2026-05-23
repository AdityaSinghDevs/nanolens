import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.config_loader import load_configs

cfg = load_configs("default")

n_embd = cfg['hyperparameters']['n_embd']
block_size = cfg['hyperparameters']['block_size']
dropout = cfg['hyperparameters']['dropout']

class Head(nn.Module):
    '''One head of attention'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x,):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out 
