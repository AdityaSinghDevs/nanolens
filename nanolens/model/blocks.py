import torch.nn as nn

from nanolens.model.attention import MultiHeadAttention
from nanolens.model.ffwd import FeedForward


class Block(nn.Module):
    "Transformer Blocks : Communication dim, n_head : num of heads I want"

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) # Residual connections
        return x 

