import torch

def build_optim(model, learning_rate, weight_decay):

    decay_params = [p for n, p in model.named_parameters() if p.dim() >=2]
    no_decay_params = [p for n,p in model.named_parameters() if p.dim() <2]

     # dim >= 2 → weight matrices (Linear, Embedding) → apply decay
    # dim < 2  → biases (dim=1) and LayerNorm (dim=1) → no decay

    param_groups = [
        {'params': decay_params,    'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate)

    return optimizer

