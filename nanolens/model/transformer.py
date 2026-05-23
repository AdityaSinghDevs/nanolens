import torch
import torch.nn as nn
from torch.nn import functional as F


from nanolens.data.tokenizer import vocab_size
from nanolens.data.loader import get_device
from nanolens.model.blocks import Block


from utils.config_loader import load_configs



cfg = load_configs("default")

n_embd = cfg['hyperparameters']['n_embd']
block_size = cfg['hyperparameters']['block_size']
n_head = cfg['hyperparameters']['n_head']


dropout = cfg['hyperparameters']['dropout']
n_layer = cfg['hyperparameters']['n_layer']

device = get_device()



class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):

        B,T = idx.shape
        #idx(input indexes (xb)) and targets(yb) are both (Batch,Time) tensor of integers [taken when the model is called]
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        x = token_emb + pos_emb 
        x = self.blocks(x)

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss 
    
    def generate(self, idx, max_new_tokens):
       #idx is (B,T) array of indices in this context here
       for _ in range(max_new_tokens):
           idx_cond = idx[:, -block_size:] #crop idx to last block size token for context
           logits, loss = self(idx_cond)

           logits = logits[:, -1, :] #focusing on the last time step only, becomes (B,C)
           probs = F.softmax(logits, dim = -1)
           idx_next = torch.multinomial(probs, num_samples=1) #sampled index taken out

           idx = torch.cat((idx, idx_next), dim = 1) #append sampled index to running sequence
       return idx
