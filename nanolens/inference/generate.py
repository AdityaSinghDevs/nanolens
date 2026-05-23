import torch
from nanolens.data.loader import get_device
from nanolens.data.tokenizer import decode

device = get_device()

def generate(model, max_new_tokens = 500):
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = model.generate(context, max_new_tokens= max_new_tokens)
    print(decode(output[0].tolist()))