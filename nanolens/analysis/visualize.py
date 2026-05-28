import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from nanolens.data.tokenizer import decode

LAYER_CMAPS = [
    'Blues',    # Layer 0
    'Purples',  # Layer 1
    'Greens',   # Layer 2
    'Oranges',  # Layer 3
    'Reds',     # Layer 4
    'YlOrBr',   # Layer 5
    'PuRd',     # Layer 6
    'BuGn',     # Layer 7
]

def plot_attention_head(result, layer=0, head=0, output_dir="results/attention"):
    weights = result['attention_weights'][layer]  # (1, 8, 29, 29)
    weights = weights[0, head].cpu().numpy()            # (29, 29) — one head

    # decode each token index back to character for axis labels
    tokens = [decode([t]) for t in result['tokens']]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cmap = LAYER_CMAPS[layer % len(LAYER_CMAPS)]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap=cmap,
        ax=ax
    )
    ax.set_title(f'Layer {layer} — Head {head} — "{result["prompt"]}"')
    plt.tight_layout()

    filename = out_path / f"L{layer}_H{head}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved → {filename}")

def plot_all_heads(result, output_dir = "results/attention"):
    n_layers = len(result['attention_weights'])
    n_heads = result['attention_weights'][0].shape[1]

    print(f"Plotting {n_layers * n_heads} heatmaps...")

    for layer in range(n_layers):
        for head in range(n_heads):
            plot_attention_head(result, layer=layer, head=head, output_dir=output_dir)
    print(f"Done. All heads saved to {output_dir}/ successfully")