import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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


def plot_hidden_state_norms(result, positions, labels, title, filename, output_dir="results/hidden_states"):
    from utils.config_loader import load_configs
    cfg = load_configs("default")
    n_layer = cfg["hyperparameters"]["n_layer"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    punct = [',', '.']
    all_norms = []

    fig, ax1 = plt.subplots(figsize=(14, 8))

    for i, pos in enumerate(positions):
        norms = []
        for n in range(n_layer):
            hidden_state = result['hidden_states'][f'block_{n}'][0, pos, :]
            norm = torch.norm(hidden_state).item()
            norms.append(norm)
        all_norms.append(norms)

        sizes = [30]
        for n in range(1, n_layer):
            delta = abs(norms[n] - norms[n-1])
            sizes.append(30 + delta * 2)

        linestyle = '--' if labels[i] in punct else '-'
        line, = ax1.plot(range(1, n_layer + 1), norms, marker='o',
                         label=labels[i], linewidth=2.5, linestyle=linestyle)
        ax1.scatter(range(1, n_layer + 1), norms, s=sizes,
                   color=line.get_color(), zorder=5, alpha=0.7)

    ax1.text(0.99, 0.02,
             f'prompt: "{result["prompt"]}"',
             transform=ax1.transAxes,
             fontsize=8, color='grey',
             ha='right', va='bottom', style='italic')

    ax1.set_ylabel('Hidden State Norm', fontsize=15)
    ax1.set_title(title, fontsize=15)
    ax1.tick_params(axis='both', labelsize=13)
    ax1.set_xticks(range(1, n_layer + 1))
    ax1.set_xlim(0.5, n_layer + 0.5)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(6))
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left', fontsize=12, framealpha=0.6)

    plt.tight_layout()
    save_path = out_path / filename
    plt.savefig(save_path, dpi=150)
    print(f"Norm trajectory saved → {save_path}")
    plt.close()


def plot_norm_deltas(result, positions, labels, output_dir="results/hidden_states"):
    from utils.config_loader import load_configs
    cfg = load_configs("default")
    n_layer = cfg["hyperparameters"]["n_layer"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bar_width = 0.1
    fig, ax2 = plt.subplots(figsize=(20, 7))

    for i, pos in enumerate(positions):
        norms = []
        for n in range(n_layer):
            hidden_state = result['hidden_states'][f'block_{n}'][0, pos, :]
            norm = torch.norm(hidden_state).item()
            norms.append(norm)

        deltas = [0] + [norms[n] - norms[n-1] for n in range(1, n_layer)]
        x_positions = [n + 1 + (i - len(positions)/2) * bar_width 
                       for n in range(n_layer)]
        ax2.bar(x_positions, deltas, width=bar_width, label=labels[i], alpha=0.75)

    ax2.set_xlabel('Layer (1-indexed)', fontsize=15)
    ax2.set_ylabel('Norm Delta', fontsize=15)
    ax2.set_title('Per-Layer Norm Change (how much work done at each layer)', fontsize=14)
    ax2.tick_params(axis='both', labelsize=13)
    ax2.set_xticks(range(1, n_layer + 1), labels=range(1, n_layer + 1))
    ax2.set_xlim(0.3, n_layer + 0.7)
    ax2.axhline(y=0, color='black', linewidth=0.9)
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.6, ncol=2)

    plt.tight_layout()
    filename = out_path / "norm_deltas.png"
    plt.savefig(filename, dpi=150)
    print(f"Norm deltas saved → {filename}")
    plt.close()


def plot_cosine_similarity(result, positions, labels, output_dir="results/hidden_states"):
    from utils.config_loader import load_configs
    cfg = load_configs("default")
    n_layer = cfg["hyperparameters"]["n_layer"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sims = []
    for i in range(n_layer - 1):
        layer_sims = []
        for pos in positions:
            a = result['hidden_states'][f'block_{i}'][0, pos, :]
            b = result['hidden_states'][f'block_{i+1}'][0, pos, :]
            sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            layer_sims.append(sim)
        sims.append(layer_sims)

    sims = np.array(sims)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        sims,
        xticklabels=labels,
        yticklabels=[f'L{i+1}→L{i+2}' for i in range(n_layer - 1)],
        cmap='RdYlGn',
        vmin=float(np.array(sims).min()) - 0.2,
        vmax=1.0,
        annot=True,
        fmt='.2f',
        ax=ax,
        linewidths=0.5,
        linecolor='grey'
    )
    ax.set_title(f'Layer-to-Layer Cosine Similarity per Token\n"{result["prompt"]}"',
                 fontsize=14)
    ax.set_xlabel('Token', fontsize=13)
    ax.set_ylabel('Layer Transition', fontsize=13)
    plt.tight_layout()

    filename = out_path / "cosine_similarity.png"
    plt.savefig(filename, dpi=150)
    print(f"Cosine similarity plot saved → {filename}")
    plt.close()