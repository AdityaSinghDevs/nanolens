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
- Serves as active interpretability research infrastructure, completed hidden state norm analysis revealing layer-wise representation dynamics.

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

> **Full classification grid, deep layer analysis, literature connections, anomalies, and what would strengthen these claims: [research_findings/attention_circuit_analysis.md](research_findings/attention_circuit_analysis.md)**

---

### The Headline Finding

**Early layers encode local sequential structure. Late layers perform abstract semantic routing.** The transition is not gradual in the way you might expect. Layer 3 is a convergence point where abstract routing commits across all 8 heads simultaneously, while local and global signals persist as secondaries. It is not a clean switch. It is a commit.

From layer 3 onward, three circuit families run in parallel in every layer without exception: abstract routing, global aggregation, and boundary detection. The model does not pipeline these sequentially. It runs them together at every depth past the transition point.

---

### Six Head Types Documented Across 64 Heads

**Previous Token Head** `L0_H6`

A near-perfect sub-diagonal stripe running the full length of the sequence. Every token attends almost exclusively to the token immediately before it, consistent from position 1 to the final token without degradation. The model's first and most precisely learned behaviour. Previous token heads are documented in Elhage et al. 2021 as foundational components of the induction circuit.

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L0_H6.png" width="45%">
</div>

---

**Identity Preservation Head** `L0_H0`

The brightest attention on the diagonal, every token attending most strongly to itself, with a soft backward wedge fading into recent history. The diagonal degrades progressively for later tokens, dissolving into broadly distributed attention by the final third of the sequence. This head appears exclusively in layer 0 with one secondary appearance in layer 1. Once contextual representations are built, the model has no further use for heads that attend primarily to the token itself.

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L0_H0.png" width="45%">
</div>

---

**Word Boundary Head** `L1_H5`

Vertical stripes of concentrated attention at space characters. Every token attends strongly to the space that precedes its word, producing a column pattern aligned precisely with word boundaries. A character-level model with no explicit notion of words has independently discovered word-level structure from statistical patterns alone. Notably, word boundary detection never centralises into a single dominant specialist across all 64 heads. It is present in every layer from 1 through 7 as a permanently distributed computation.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L1_H5.png" width="44%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H5.png" width="44%">

*Left: L1_H5, broad word boundary head attending to every space. Right: L5_H5, selective boundary head attending to syntactically meaningful spaces only. The same circuit type becomes more selective with depth.*

---

**First Token Sink Head** `L5_H6`, `L6_H3`

The entire left column is lit. Nearly every token routes strong attention back to the first character of the sequence regardless of its own position. The first token functions as a global information sink, accumulating sequence-level context that individual tokens can query. This circuit follows a complete lifecycle across the model: weak background signal from layer 1, growing distributed presence through layers 2 to 4, crystallising into dedicated specialist heads at layers 5 and 6, then dispersing back to secondary status at layer 7. Global context aggregation is most active in the middle of the network, not at the end.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H6.png" width="44%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L6_H3.png" width="44%">

*Left: L5_H6, first dedicated first token sink head. Right: L6_H3, strongest first token sink head in the model. Their presence across two consecutive layers confirms a stable load-bearing circuit.*

---

**Abstract Routing Head** `L4_H4`

No diagonal. No vertical stripes. Sparse, high-contrast hits at specific non-local positions scattered across the heatmap, with the majority of cells near zero. This head is no longer tracking sequence order or structural boundaries. It is routing information based on learned abstract features, connecting tokens that share semantic or syntactic relevance regardless of distance. L4_H4 is notably the earliest layer where a strong unambiguous sparse head appears, suggesting abstract routing begins earlier in this model than the layer summary alone implies.

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L4_H4.png" width="45%">
</div>

---

**Extended Lookback Head** `L2_H1`

A previous token pattern where the brightest attention sits not one step back but two to three positions back, producing a sub-diagonal stripe offset further from the main diagonal than standard previous token heads. This head extends the local context window beyond single-step lookback. Its position at layer 2, the final layer before the layer 3 convergence, is the key observation. The model widens its local receptive field immediately before committing to abstract routing.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L0_H6.png" width="44%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L2_H3.png" width="44%">

*Left: L0_H6, standard previous token head, one step back. Right: L2_H3, extended lookback head, two to three steps back. The stripe moves further from the main diagonal.*

---

### Layer Summary

| Layer | Dominant Type | Character |
|---|---|---|
| 0 | Previous token, identity preservation | Pure local, first and most precise behaviours |
| 1 | Previous token, boundary detection emerging | Local with first word boundary signal |
| 2 | Previous token transitioning, extended lookback | Local widening before the transition |
| 3 | Abstract routing across all 8 heads | Convergence point, the commit layer |
| 4 | Abstract routing, BOS sink emerging | Abstract dominant, aggregation building |
| 5 | Abstract routing, BOS sink fully formed | Three circuit families running in parallel |
| 6 | Abstract routing, BOS sink peak | Strongest first token sink heads in the model |
| 7 | Abstract routing across all 8 heads | Most functionally committed layer in the model |

---
---

***For the full classification of all 64 heads, deep analysis of each finding, literature connections to Elhage et al., Clark et al., and Wang et al., documented anomalies, and the experiments that would move these observations to defensible claims, read the full analysis:***
[research_findings/attention_circuit_analysis.md](research_findings/attention_circuit_analysis.md)

---
---
### Hidden State Norm Analysis

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_plot_delta.png" width="75%">
</div>

Hidden state norm analysis was run on the prompt *"Raskolnikov hesitated at the threshold, his hands trembling."* tracking five tokens across all 8 layers: `R` , `space`, `h`, `,`, and `.`

`R` starts with the highest norm of all tracked tokens at layer 1 and maintains that lead throughout, peaking at layer 6, directly corroborating the BOS sink heads identified at layers 5 and 6. At layer 7, R's norm drops for the first time. This is not a loss of importance - it is redistribution. The model has finished accumulating context into R and begins spreading that information outward as it moves toward prediction.

`space` and `h` grow at a steady near-linear rate across all 8 layers, standard tokens building stable representations at consistent depth. Layers 1 through 4 show steady norm growth across all tokens with no sharp transitions, consistent with the diagonal attention patterns observed in early heatmaps but requiring further analysis to characterise in detail.

The most striking finding is in the punctuation tokens. `,` and `.` track closely with `space` and `h` through layer 6, then diverge sharply upward in layers 7 and 8 — the final layers building terminal syntactic representations right before prediction. This independently corroborates the abstract routing heads found at layers 6 and 7. Two separate mechanisms, same conclusion: the deepest layers are where structural meaning resolves.

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
**Verify everything is wired correctly before training**
```
python cli.py -d
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