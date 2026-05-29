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


def plot_hidden_state_norms(result, positions, labels, output_dir="results/hidden_states"):
    from utils.config_loader import load_configs
    cfg = load_configs("default")
    n_layer = cfg["hyperparameters"]["n_layer"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 11), 
                                    gridspec_kw={'height_ratios': [2, 1]})

    punct = [',', '.']
    all_norms =[]
    bar_width = 0.15
    
    for i, pos in enumerate(positions):
        norms = []
        for n in range(n_layer):
            hidden_state = result['hidden_states'][f'block_{n}'][0, pos, :]
            norm = torch.norm(hidden_state).item()
            norms.append(norm)
        all_norms.append(norms)
        
        sizes = [40]
        for n in range(1, n_layer):
            delta = abs(norms[n] - norms[n-1])
            sizes.append(40 + delta * 3)

        linestyle = '--' if labels[i] in punct else '-'
        line, =ax1.plot(range(n_layer), norms, marker='o', 
                label=labels[i], linewidth=2, linestyle=linestyle)

        ax1.scatter(range(n_layer), norms, s=sizes, color=line.get_color(), zorder=5)

        # delta bars on ax2
        deltas = [0] + [all_norms[i][n] - all_norms[i][n-1] for n in range(1, n_layer)]
        x_positions = [n + (i - len(positions)/2) * bar_width for n in range(n_layer)]
        ax2.bar(x_positions, deltas, width=bar_width, 
                label=labels[i], color=line.get_color(), alpha=0.7)

    ax1.text(0.99, 0.02,
             f'prompt: "{result["prompt"]}"',
             transform=ax1.transAxes,
             fontsize=8, color='grey',
             ha='right', va='bottom', style='italic')

    ax1.set_ylabel('Hidden State Norm', fontsize=11)
    ax1.set_title('Token Representation Magnitude Across Layers', fontsize=12)
    ax1.set_xticks(range(n_layer), labels=range(1, n_layer + 1))
    ax1.yaxis.set_major_locator(plt.MultipleLocator(3))
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel('Layer(1-indexed)', fontsize=11)
    ax2.set_ylabel('Norm Delta', fontsize=11)
    ax2.set_title('Per-Layer Norm Change (how much work done at each layer)', fontsize=10)
    ax2.set_xticks(range(n_layer), labels=range(1, n_layer + 1))
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    filename = out_path / "norm_plot_delta.png"
    plt.savefig(filename, dpi=150)
    print(f"Norm plot saved → {filename}")