import torch

from utils.config_loader import load_configs
from nanolens.data.tokenizer import encode,text

cfg = load_configs("default")

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def train_test_split():
    data = torch.tensor(encode(text), dtype = torch.long)
    train_test_split = cfg['hyperparameters']['n_train_test_split']

    n = int(train_test_split*len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data

def get_batch(split):

    train_data, val_data = train_test_split()
    block_size = cfg['hyperparameters']['block_size']
    batch_size = cfg['hyperparameters']['batch_size']
    device = get_device()

    data = train_data if split=='train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)

    return x,y