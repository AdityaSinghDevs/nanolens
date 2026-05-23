import torch
import torch.nn as nn

from utils.config_loader import load_configs
from nanolens.model.head import Head

cfg = load_configs("default")

n_embd = cfg['hyperparameters']['n_embd']
dropout = cfg['hyperparameters']['dropout']

class MultiHeadAttention(nn.Module):
    '''Multiple heads of attention in parallel'''

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out 
