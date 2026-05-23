import torch.nn as nn

from utils.config_loader import load_configs

cfg = load_configs("default")

dropout = cfg['hyperparameters']['dropout']

class FeedForward(nn.Module):
    """ Linear layer followed by non linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self,x):
        return self.net(x)