# NanoLens



```        

               `7MN.   `7MF'                           `7MMF'                                  
                 MMN.    M                               MM                                    
                 M YMb   M  ,6"Yb.  `7MMpMMMb.  ,pW"Wq.  MM         .gP"Ya `7MMpMMMb.  ,pP"Ybd 
                 M  `MN. M 8)   MM    MM    MM 6W'   `Wb MM        ,M'   Yb  MM    MM  8I   `" 
                 M   `MM.M  ,pm9MM    MM    MM 8M     M8 MM      , 8M""""""  MM    MM  `YMMMa. 
                 M     YMM 8M   MM    MM    MM YA.   ,A9 MM     ,M YM.    ,  MM    MM  L.   I8 
               .JML.    YM `Moo9^Yo..JMML  JMML.`Ybmd9'.JMMmmmmMMM  `Mbmmd'.JMML  JMML.M9mmmP'

                          mechanistic interpretability toolkit · built from scratch
```



> *A mechanistic interpretability toolkit built from scratch, because understanding transformers from the outside was never going to be enough.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Autoregressive_Transformer-5C4EE5)]()
[![Model](https://img.shields.io/badge/Model-25M_params-8A2BE2)](https://github.com/AdityaSinghDevs/nanolens/releases)
[![Training](https://img.shields.io/badge/Trained_on-Dostoevsky-darkgreen)](https://www.gutenberg.org)
[![Tokenisation](https://img.shields.io/badge/Tokenisation-Character--level-orange)]()
[![Interpretability](https://img.shields.io/badge/Research-Mechanistic_Interp-navy)]()
[![Heads](https://img.shields.io/badge/Attention_Heads-64_(8×8)-teal)]()

NanoLens is a fully modular autoregressive transformer implemented from scratch in Python and PyTorch, trained on the complete prose of Fyodor Dostoevsky, and extended with dual-mechanism inspection infrastructure for mechanistic interpretability research. It is not a wrapper. It is not a fine-tuned model. Every component, from attention heads to the residual stream to the training loop to the inspection hooks, was built and understood from first principles.

This project began as an implementation of Andrej Karpathy's nanoGPT and evolved into a proper research toolkit. The dataset choice was deliberate: Crime and Punishment, The Brothers Karamazov, and The Idiot, totalling 4.4 million characters of dense, philosophically rich prose with complex sentence structure, recurring character names, and heavy punctuation. A harder target than Shakespeare. More interesting to watch a small model learn.

---

## Motivation


```
                                                 __
                                                 ||
                                                ====
                                                |  |__
                                                |  |-.\
                                                |__|  \\
                                                 ||   ||
                                               ======__|
                                              ________||__
                                             /____________\
```



There is a specific kind of frustration that comes from using a system you do not fully understand. You can prompt it, fine-tune it, benchmark it, and deploy it. But if someone asks you what it is actually doing, the honest answer is usually "I am not sure."

Mechanistic interpretability is the field that takes that question seriously. The goal is not to describe transformer behaviour statistically but to identify the actual computational mechanisms responsible for it, the circuits, the attention patterns, the features, the information routing. It is the difference between knowing that a model performs well and knowing why.

That question is what NanoLens is built around. I wanted to understand transformers from the inside, not from the API surface. Building one from scratch was the only way to earn that understanding. Inspecting it was the natural next step.

---

## What is NanoLens ?

A 25-million parameter character-level auto-regressive transformer that:

- Generates text with authentic Dostoevsky texture, correct character names, philosophical register, and dramatic sentence structure
- Exposes its internal attention patterns and hidden states through dual-mechanism inspection
- Has documented functional specialisation across its 64 attention heads (8 layers x 8 heads)
- Serves as active interpretability research infrastructure, with ongoing work on hidden state norm analysis and attention entropy metrics

---

## The Model

| Hyperparameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 25,441,380 |
| Layers | 8 |
| Attention Heads | 8 per layer |
| Embedding Dimension | 512 |
| Head Size | 64 (512 / 8) |
| Context Window | 256 characters |
| Vocabulary Size | 100 unique characters |
| Tokenisation | Character-level |
| Training Data | Dostoevsky, 4.4M characters |
| Final Val Loss | 1.1144 |
| Final Train Loss | 1.0610 |
| Train/Val Gap | 0.053 |

### The Attention Mechanism

Every attention head in this model computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Q, K, and V are three independent learned projections of the same input vector. The scaling factor $\sqrt{d_k}$ keeps the dot products from growing large enough to push softmax into saturation. Each of the 8 heads per layer operates in a restricted subspace of dimension 64, runs in parallel, and gets concatenated and projected back to the full 512-dimensional embedding before the next layer sees it.\

Given NanoLens is a transformer trained with 8 layers, with 8 attention head in each leading to 64 total heads, The research question this project investigates is what each of those 64 heads and 8 layers actually learn to do.

### Training Details

The first training run overfit badly. Validation loss climbed back to 1.27 after step 1500 while training loss kept falling. The fix was developed empirically:

- **Data**: Expanded from one book to all three Dostoevsky novels
- **LR Schedule**: Cosine decay with 100-step linear warmup (3e-4 to near zero)
- **Regularisation**: Dropout increased to 0.3
- **Optimiser**: AdamW with separated decay/no-decay parameter groups (weight matrices get decay 0.1, biases and LayerNorm parameters do not)

Result: clean training, negligible overfitting, train/val gap of 0.053 at step 5000.

---

## Repository Structure

```
nanolens/
├── config/
│   └── default.yaml          # all hyperparameters, nothing hardcoded
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
│   └── attention/            # heatmaps, L{layer}_H{head}.png
├── notebooks/
│   └── colab-30Mparam-train-run.ipynb
├── utils/
│   └── config_loader.py      # YAML config reader
├── cli.py                    # argparse entry point (train/generate/inspect)
├── requirements.txt
└── README.md
```

---

## Inspection Infrastructure

NanoLens uses a dual-mechanism architecture to expose model internals during a single forward pass. Two mechanisms, one pass, complete picture.

### Pull-based (return_weights)
`Head.forward(return_weights=True)` returns `(out, wei)` where `wei` is the raw `(B, T, T)` attention map for that head. This propagates up through `MultiHeadAttention -> Block -> TransformerModel`, collecting a list of 8 tensors, one per layer, each of shape `(1, n_heads, T, T)`.

### Push-based (PyTorch Hooks)
Forward hooks registered on each `Block` fire automatically after each block's forward pass, capturing the hidden state `(1, T, 512)` in a dictionary keyed by block index. Hooks are removed immediately after the forward pass to prevent accumulation across runs.

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

> **Epistemic note**: These findings are from a single trained checkpoint on a single prompt. They are directional and exploratory, presented as observations from a character-level model and not as claims about transformers in general. The patterns are consistent with prior interpretability literature, which lends them credibility, but further experiments are needed before stronger claims can be made.

### The Headline Finding

**Early layers encode local sequential structure. Late layers perform abstract semantic routing.** This gradient is visible in the raw heatmap grid without opening a single individual image. Layers 0-2 are almost entirely diagonal. Layers 5-7 are sparse and non-local. The model builds its understanding of text from sequence order up to abstract meaning, layer by layer.

### Documented Head Types

Across 64 attention heads, the following functional types have been identified:

---

**Type 1 — Previous Token Head** `L0_H2`

A near-perfect sub-diagonal stripe: every token attends almost exclusively to the token immediately before it. This head has learned to propagate local sequence order with high precision. Previous token heads are among the most well-documented head types in interpretability literature, and their presence in layer 0 confirms that local sequential structure is the first thing this model learns.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L0_H2.png" width="48%">

*Left: L0_H2 — previous token head, near-perfect sub-diagonal. Right: L0_H0 — self + local context, diagonal with soft backward wedge.*

---

**Type 2 — Self + Local Context Head** `L0_H0`

The brightest attention is on the diagonal (self), with a soft backward wedge fading into recent history. This head preserves a token's own identity while maintaining a short contextual window. Distinct from the previous token head: this one says "I matter most, but I remember where I came from."

 <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L0_H0.png" width="48%">

---

**Type 3 — Word Boundary / Space Head** `L1_H5`

Vertical stripes at space characters. Every token in the sequence attends strongly to the space that precedes its word. This head has learned that spaces are structural boundaries and uses them as positional anchors, which is a remarkable finding for a character-level model with no explicit notion of words. A more selective version appears at `L5_H5`, attending specifically to syntactically meaningful boundaries like the comma and the space before a verb.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L1_H5.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H5.png" width="48%">

*Left: L1_H5 — broad space/boundary head, vertical stripes at every word boundary. Right: L5_H5 — selective boundary head, syntactically meaningful positions only.*

---

**Type 4 — BOS / First Token Sink Head** `L5_H6`, `L6_H3`

The entire left column is lit. Nearly every token routes strong attention back to the first character of the sequence. The first token functions as a global information sink, accumulating sequence-level context that individual tokens can query. Its presence across multiple layers (5 and 6) suggests this is a stable, load-bearing circuit in this model, not a one-off pattern.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H6.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L6_H3.png" width="48%">

*Left: L5_H6 — BOS sink at layer 5. Right: L6_H3 — BOS sink persisting at layer 6, confirming it as a stable circuit.*

---

**Type 5 — Abstract Semantic Routing Head** `L6_H2`, `L7_H1`

No diagonal. Sparse, non-local, high-contrast hits at semantically meaningful positions. These heads are no longer tracking sequence order. They are routing information based on learned abstract features. By layer 6, attention has moved from "what came before me" to "what is relevant to me."

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L6_H2.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L7_H1.png" width="48%">

*Left: L6_H2 — sparse semantic routing emerging at layer 6. Right: L7_H1 — fully abstract routing at layer 7, diagonal completely absent.*

---

**Type 6 — High Entropy / Diffuse Head** `L7_H7`

Soft, broadly distributed attention across many positions with no dominant pattern. Low-contrast, high-entropy. This head may be performing something like context averaging rather than focused retrieval, a counterpoint to the sparse routing heads in the same layer.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L7_H7.png" width="48%">

*L7_H7 — high entropy diffuse head, attention distributed broadly with no dominant structure.*

---

### Layer Distribution

| Layer | Dominant Type |
|---|---|
| 0 | Local sequence (diagonal), all 8 heads |
| 1 | Local sequence with boundary detection emerging |
| 2 | Transitional, diagonal breaking into chunks |
| 3 | Transitional, local giving way to structured patterns |
| 4 | Mixed, mid-layer refinement and word-level chunking |
| 5 | Mixed, BOS sink and selective boundary heads appearing |
| 6 | Abstract, BOS sink, semantic routing, sparse patterns |
| 7 | Abstract, semantic routing and diffuse heads |

---

## Generated Output

The model generates with authentic Dostoevsky texture, correct character names, philosophical register, em-dash usage mid-thought:

```
And as forgotten, Dmitri Fyodorovitch, who has proved, a conscious
with the last fraud coffin efforts. He was not to be alarmed, his
seemed to go without to, for the the struggle which he recovered
himself or it sits the toff and at once am playing to be for shome
to be a student place of it? We will trether if you came to
```

25 million parameters. Character-level. Trained from scratch. Gloriously janky in the best possible way.

---

## Quickstart

Pretrained weights are available in [Releases](https://github.com/AdityaSinghDevs/nanolens/releases). Download instructions below.

```bash
git clone https://github.com/AdityaSinghDevs/nanolens
cd nanolens
pip install -r requirements.txt
```

**Train from scratch:**
```bash
python cli.py
```

**Generate from pretrained checkpoint:**
```bash
python cli.py --checkpoint checkpoints/nanolens_v1.pt
```

**Inspect attention patterns:**
```bash
python cli.py --checkpoint checkpoints/nanolens_v1.pt \
              --inspect "Raskolnikov hesitated at the threshold, his hands trembling."
```

This runs a full forward pass, captures all 64 attention heatmaps and 8 hidden state tensors, and saves PNG visualisations to `results/attention/`.

---

## Weights

Pretrained checkpoint released under [Releases](https://github.com/AdityaSinghDevs/nanolens/releases/tag/v1.0).

Download and place in the `checkpoints/` directory:

```bash
# Download nanolens_v1.pt from the releases page and move it to checkpoints/
mv nanolens_v1.pt checkpoints/nanolens_v1.pt

# Run inference
python cli.py --checkpoint checkpoints/nanolens_v1.pt
```

Checkpoint details: 8 layers, 8 heads, 512 embedding dim, trained for 5000 steps on 4.4M characters of Dostoevsky. Final val loss 1.1144.
> Size : ~113MB
---

## What's Next ?

Pending work on NanoLens:

- Hidden state norm plots across layers for specific tokens (common words, character names, punctuation, rare words), to track how representation magnitude evolves with depth
- Single-head vs multi-head controlled comparison experiment, same architecture and data, to study what multi-head parallelism actually contributes

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