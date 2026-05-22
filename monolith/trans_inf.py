import torch
import torch.nn as nn
from torch.nn import functional as F

# =========================================================
# HYPERPARAMETERS
# MUST MATCH TRAINING CONFIG EXACTLY
# =========================================================

batch_size = 1
block_size = 256
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================================================
# LOAD DATA + VOCAB
# MUST MATCH TRAINING VOCAB EXACTLY
# =========================================================

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# =========================================================
# ATTENTION HEAD
# =========================================================

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * (k.shape[-1] ** -0.5)

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)

        out = wei @ v

        return out


# =========================================================
# MULTI HEAD ATTENTION
# =========================================================

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)

        out = self.proj(out)

        out = self.dropout(out)

        return out


# =========================================================
# FEED FORWARD
# =========================================================

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(n_embd, 4 * n_embd),

            nn.ReLU(),

            nn.Linear(4 * n_embd, n_embd),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# TRANSFORMER BLOCK
# =========================================================

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(n_head, head_size)

        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)

        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        x = x + self.sa(self.ln1(x))

        x = x + self.ffwd(self.ln2(x))

        return x


# =========================================================
# GPT MODEL
# =========================================================

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B,T = idx.shape

        token_emb = self.token_embedding_table(idx)

        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )

        x = token_emb + pos_emb

        x = self.blocks(x)

        logits = self.lm_head(x)

        loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = BigramLanguageModel()

model.load_state_dict(
    torch.load(
        "char_trans_weights.pth",
        map_location=device
    )
)

model = model.to(device)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
print(
    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
)

model.eval()

# =========================================================
# GENERATE TEXT
# =========================================================

def generate_response(prompt, max_new_tokens=200):

    context = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device
    )

    context = context[:, -block_size:]

    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens
    )

    output = decode(generated[0].tolist())

    response = output[len(prompt):]

    return response


# =========================================================
# STANDALONE GENERATION MODE
# =========================================================

if __name__ == "__main__":

    context = torch.zeros(
        (1,1),
        dtype=torch.long,
        device=device
    )

    generated = model.generate(
        context,
        max_new_tokens=1000
    )

    print(decode(generated[0].tolist()))