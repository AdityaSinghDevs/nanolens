import torch
 
from utils.config_loader import load_configs
from nanolens.data.tokenizer import encode, text
 
cfg = load_configs("default")
 
block_size = cfg['hyperparameters']['block_size']
batch_size = cfg['hyperparameters']['batch_size']
split_ratio = cfg['hyperparameters']['n_train_test_split']

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def _build_splits():
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(split_ratio * len(data))
    return data[:n], data[n:]
 
train_data, val_data = _build_splits()

def get_batch(split):

    """
    Returns a random (x, y) batch from the train or val split.
 
    x : input tokens  — shape (batch_size, block_size)
    y : target tokens — shape (batch_size, block_size), offset by 1
    """

    device = get_device()

    data = train_data if split=='train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)

    return x,y