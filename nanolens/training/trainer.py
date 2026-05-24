import torch
import math
from pathlib import Path

from nanolens.model.transformer import TransformerModel
from nanolens.training.optimizer import build_optim
from nanolens.data.loader import get_batch, get_device

from utils.config_loader import load_configs

cfg = load_configs("default")

eval_iters = cfg['hyperparameters']['eval_iters']
max_iters = cfg['hyperparameters']['max_iters']
eval_interval = cfg['hyperparameters']['eval_interval']
learning_rate = float(cfg['hyperparameters']['learning_rate'])
weight_decay  = float(cfg['hyperparameters']['weight_decay'])
n_layer       = cfg['hyperparameters']['n_layer']
n_head        = cfg['hyperparameters']['n_head']
n_embd        = cfg['hyperparameters']['n_embd']

device = get_device()

def get_lr(current_iter):
    warmup_iters =100
    min_lr = learning_rate/10

    if current_iter < warmup_iters:
        return learning_rate * (current_iter + 1) / warmup_iters
    
    decay_ratio = (current_iter - warmup_iters) / (max_iters - warmup_iters)
    coeff  = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def trainer():
    
    model = TransformerModel().to(device)
    optimizer = build_optim(model, learning_rate, weight_decay)

    print("Everything sticked together well, Training Started..")
    for iter in range(max_iters):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")

        if iter % 500 == 0 and iter > 0 :
            save_checkpoint(model, iter)

        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none = True)
        loss.backward()
        optimizer.step()
    
    save_checkpoint(model, max_iters)

    return model

def save_checkpoint(model, iter):
    name = f"gpt_L{n_layer}_H{n_head}_E{n_embd}_{iter}.pt"
    path = Path("checkpoints")/name
    path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved -> {path}")
    return path

def load_model(checkpoint_path):
    model = TransformerModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded weights from {checkpoint_path}")
    return model

