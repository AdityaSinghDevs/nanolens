# NanoLens 

> *A mechanistic interpretability toolkit built from scratch — because understanding transformers from the outside was never going to be enough.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

NanoLens is a fully modular autoregressive transformer implemented from scratch in Python and PyTorch, trained on the complete prose of Fyodor Dostoevsky, and extended with dual-mechanism inspection infrastructure for mechanistic interpretability research. It is not a wrapper. It is not a fine-tuned model. Every component — attention heads, residual stream, training loop, inspection hooks — was built and understood from first principles.

This project began as an implementation of Andrej Karpathy's nanoGPT and evolved into a proper research toolkit. The dataset choice was deliberate: Crime and Punishment, The Brothers Karamazov, and The Idiot — 4.4 million characters of dense, philosophically rich prose with complex sentence structure, recurring character names, and heavy punctuation. A harder target than Shakespeare. More interesting to watch a small model learn.

---

## What This Is

A 25-million parameter character-level transformer that:

- Generates text with authentic Dostoevsky texture — correct character names, em-dash usage mid-thought, dramatic register
- Exposes its internal attention patterns and hidden states through dual-mechanism inspection
- Has documented functional specialisation across its 64 attention heads (8 layers × 8 heads)
- Serves as the direct foundation for **PRISMA** — ongoing research into mechanistic changes under LoRA and QLoRA compression

---

## The Model

| Hyperparameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 25,441,380 |
| Layers | 8 |
| Attention Heads | 8 per layer |
| Embedding Dimension | 512 |
| Head Size | 64 (512 ÷ 8) |
| Context Window | 256 characters |
| Vocabulary Size | 100 unique characters |
| Tokenisation | Character-level |
| Training Data | Dostoevsky — 4.4M characters |
| Final Val Loss | 1.1144 |
| Final Train Loss | 1.0610 |
| Train/Val Gap | 0.053 |

### Training Details

The first training run overfit — validation loss climbed back to 1.27 after step 1500 while training loss continued decreasing. The diagnosis and fix were developed empirically:

- **Data**: Expanded from one book to all three Dostoevsky novels
- **LR Schedule**: Cosine decay with 100-step linear warmup (3e-4 → near zero)
- **Regularisation**: Dropout increased to 0.3
- **Optimiser**: AdamW with separated decay/no-decay parameter groups (weight matrices get decay 0.1, biases and LayerNorm parameters do not)

Result: clean training, no overfitting, train/val gap of 0.053 at step 5000.

### The Attention Equation

At the core of every attention head:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where Q, K, V are three independent learned projections of the same input vector — query, key, and value. The scaling factor $\sqrt{d_k}$ prevents the dot products from growing large enough to push softmax into saturation. Each head operates in a restricted subspace of dimension 64, with 8 heads running in parallel before their outputs are concatenated and projected back to the full embedding dimension.

---

## Inspection Infrastructure

NanoLens uses a dual-mechanism architecture to expose model internals during a single forward pass:

### Pull-based (return_weights)
`Head.forward(return_weights=True)` returns `(out, wei)` where `wei` is the raw `(B, T, T)` attention map for that head. This propagates up through `MultiHeadAttention → Block → TransformerModel`, collecting a list of 8 tensors — one per layer — each of shape `(1, n_heads, T, T)`.

### Push-based (PyTorch Hooks)
Forward hooks registered on each `Block` fire automatically after each block's forward pass, capturing the hidden state `(1, T, 512)` in a dictionary keyed by block index. Hooks are removed immediately after the forward pass to prevent accumulation.

Both mechanisms run in a single call to `inspect()`, returning:

```python
{
    'tokens':            List[int],           # encoded prompt
    'prompt':            str,                 # original prompt string  
    'hidden_states':     Dict[str, Tensor],   # 8 tensors of shape (1, T, 512)
    'attention_weights': List[Tensor],        # 8 tensors of shape (1, 8, T, T)
}
```

---

## Research Findings

> **Epistemic note**: These findings are from a single trained checkpoint on a single prompt. They are directional and exploratory — presented as observations from a character-level model, not claims about transformers in general. The patterns are consistent with prior interpretability literature, which lends them credibility, but further experiments are needed for stronger claims.

### The Headline Finding

**Early layers encode local sequential structure. Late layers perform abstract semantic routing.** This layer hierarchy is visible in the raw heatmap grid without opening a single individual image — layers 0-2 are almost entirely diagonal, layers 5-7 are sparse and non-local.

### Documented Head Types

Across 64 attention heads, the following functional types have been identified as of now:

---

**Type 1 — Previous Token Head** `L0_H2`

A near-perfect sub-diagonal stripe: every token attends almost exclusively to the token immediately before it. This head has learned to propagate local sequence order with high precision. Previous token heads are among the most documented head types in interpretability literature and their presence in layer 0 confirms that local sequential structure is the first thing this model learned.

---

**Type 2 — Self + Local Context Head** `L0_H0`

The brightest attention is on the diagonal (self), with a soft backward wedge fading into recent history. This head preserves a token's own identity while maintaining a short contextual window. Distinct from the previous token head — this one says "I matter most, but I remember where I came from."

---

**Type 3 — Word Boundary / Space Head** `L1_H5`

Vertical stripes at space characters. Every token in the sequence attends strongly to the space that precedes its word. This head has learned that spaces are structural boundaries and uses them as positional anchors — remarkable for a character-level model that has no explicit notion of words. A more selective version appears at `L5_H5`, attending specifically to syntactically meaningful boundaries (the comma, the space before a verb).

---

**Type 4 — BOS / First Token Sink Head** `L5_H6`, `L6_H3`

The entire left column is lit. Nearly every token routes strong attention back to the first character of the sequence. This is a well-documented phenomenon in interpretability research — the first token functions as a global information sink, accumulating sequence-level context that individual tokens can query. Its presence across multiple layers (5 and 6) suggests this is a stable, load-bearing circuit in this model.

---

**Type 5 — Abstract Semantic Routing Head** `L6_H2`, `L7_H1`

No diagonal. Sparse, non-local, high-contrast hits at semantically meaningful positions. These heads are no longer tracking sequence order — they are routing information based on learned abstract features. By layer 6, attention has moved from "what came before me" to "what is relevant to me."

---

**Type 6 — High Entropy / Diffuse Head** `L7_H7`

Soft, broadly distributed attention across many positions with no dominant pattern. Low-contrast, high-entropy. This head may be performing something like context averaging rather than focused retrieval — a counterpoint to the sparse routing heads in the same layer.

---

### Layer Distribution

| Layer | Dominant Type |
|---|---|
| 0 | Local sequence (diagonal) — all 8 heads |
| 1 | Local sequence with boundary detection emerging |
| 2 | Transitional — diagonal breaking into chunks |
| 3 | Transitional — local giving way to structured patterns |
| 4 | Mixed — mid-layer refinement, word-level chunking |
| 5 | Mixed — BOS sink and selective boundary heads appearing |
| 6 | Abstract — BOS sink, semantic routing, sparse patterns |
| 7 | Abstract — semantic routing and diffuse heads |

---

## Generated Output

The model generates with authentic Dostoevsky texture — correct character names, philosophical register, em-dash usage mid-thought:

```
And as forgotten, Dmitri Fyodorovitch, who has proved, a conscious 
with the last fraud coffin efforts. He was not to be alarmed, his 
seemed to go without to, for the the struggle which he recovered 
himself or it sits the toff and at once am playing to be for shome 
to be a student place of it? We will trether if you came to
```

25 million parameters. Character-level. Trained from scratch. Gloriously janky in the best possible way.

---

## Project Structure

```
nanolens/
├── config/
│   └── default.yaml          # all hyperparameters — nothing hardcoded
├── nanolens/
│   ├── data/
│   │   ├── training/         # Crime and Punishment, Brothers Karamazov, The Idiot
│   │   ├── tokenizer.py      # character-level encode/decode
│   │   └── loader.py         # batching, train/val split
│   ├── model/
│   │   ├── head.py           # single attention head
│   │   ├── attention.py      # multi-head attention
│   │   ├── ffwd.py           # feedforward block
│   │   ├── blocks.py         # full transformer block
│   │   └── transformer.py    # full model + generate()
│   ├── training/
│   │   ├── optimizer.py      # AdamW with param group separation
│   │   └── trainer.py        # training loop + LR decay + checkpointing
│   ├── inference/
│   │   └── generate.py       # generation wrapper
│   └── analysis/
│       ├── inspector.py      # dual-mechanism inspection
│       └── visualize.py      # attention heatmap plotting
├── results/
│   └── attention/            # heatmaps — L{layer}_H{head}.png
├── notebooks/
│   └── colab-30Mparam-train-run.ipynb
├── utils/
│   └── config_loader.py      # YAML config reader
├── cli.py                    # argparse entry point (train/generate/inspect)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/your-handle/nanolens
cd nanolens
pip install -r requirements.txt
```

**Train:**
```bash
python cli.py
```

**Generate from checkpoint:**
```bash
python cli.py --checkpoint checkpoints/gpt_L8_H8_E512_5000.pt
```

**Inspect attention patterns:**
```bash
python cli.py --checkpoint checkpoints/gpt_L8_H8_E512_5000.pt \
              --inspect "Raskolnikov hesitated at the threshold, his hands trembling."
```

This runs a full forward pass, captures all 64 attention heatmaps and 8 hidden state tensors, and saves PNG visualisations to `results/attention/`.

---

## What's Next

NanoLens is the foundation for **PRISMA** — a planned research paper investigating mechanistic changes in transformer attention circuits under LoRA and QLoRA compression. The core question: does compression preserve, destroy, or reroute the circuits identified here? Are certain head types disproportionately affected by rank and scale? Does quantisation introduce changes beyond those caused by LoRA alone?

The inspection infrastructure built here — dual-mechanism capture, attention heatmaps, hidden state hooks — will be ported and extended for PRISMA using TransformerLens on GPT-2 Medium and Pythia-1.4B.

Pending NanoLens work:
- Hidden state norm plots across layers for specific tokens
- Attention entropy metric (H = -Σ p log p) across all 64 heads
- Single-head vs multi-head controlled comparison experiment

---

## References

- Vaswani et al. — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Elhage et al. — [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- Andrej Karpathy — [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Neel Nanda — [TransformerLens](https://github.com/neelnanda-io/TransformerLens)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built during final year B.Tech, amidst majestically worthless engineering exams.*