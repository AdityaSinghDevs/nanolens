import torch

from nanolens.data.tokenizer import encode

def inspect(model, prompt, device):
    tokens = encode(prompt)

    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) #adding batch dim -> shape (1,T)

    hidden_states = {}
    hooks = []

    for i, block in enumerate(model.blocks):
        def make_hook(idx):
            def hook_fn(module, input, output):
                #output is hidden state after this block
                hidden_states[f'block_{idx}'] = output[0].detach().cpu()
            return hook_fn
        handle = block.register_forward_hook(make_hook(i))
        hooks.append(handle)

    model.eval()
    with torch.no_grad():
        logits, _, all_weights = model(x, return_weights = True)

    #removing hooks immediately
    for handle in hooks:
        handle.remove()

    return {
        'tokens' : tokens,
        'prompt' : prompt,
        'hidden_states': hidden_states,
        'attention_weights': all_weights,
    }
