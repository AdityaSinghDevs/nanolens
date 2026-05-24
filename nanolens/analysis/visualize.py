import torch
import matplotlib.pyplot as plt
import seaborn as sns
from nanolens.data.tokenizer import decode

def plot_attention_head(result, layer=0, head=0):
    weights = result['attention_weights'][layer]  # (1, 8, 29, 29)
    weights = weights[0, head].cpu().numpy()            # (29, 29) — one head

    # decode each token index back to character for axis labels
    tokens = [decode([t]) for t in result['tokens']]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        ax=ax
    )
    ax.set_title(f'Layer {layer} — Head {head} — "{result["prompt"]}"')
    plt.tight_layout()
    plt.savefig(f'attention_L{layer}_H{head}.png', dpi=150)
    print(f"Saved → attention_L{layer}_H{head}.png")
    plt.show()