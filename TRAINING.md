# NanoLens — Training, Configuration, and Scaling Guide

This document covers everything needed to train your own NanoLens model from scratch, tune the architecture, scale parameters, resume from checkpoints, and run the inspection toolkit on your trained model.

Interpretability usage is covered in a dedicated section at the end.

---

## Quick Setup

```bash
git clone https://github.com/AdityaSinghDevs/nanolens
cd nanolens
pip install -r requirements.txt
```

Verify everything is wired correctly before training:

```bash
python cli.py -d
```

This runs a dry run that checks the tokenizer, data loader, model instantiation, forward pass, return_weights mechanism, optimizer, and inspector. All checks should pass before proceeding.

---

## Your Data

Place plain text files in `nanolens/data/training/`. The tokenizer reads all `.txt` files in that directory and concatenates them. Remove the existing Dostoevsky files and replace with your own corpus.

**Character-level tokenization means vocabulary size equals the number of unique characters in your training data.** The model automatically builds the vocabulary from whatever characters appear. If your data contains 80 unique characters, vocab_size will be 80. If it contains 120, it will be 120. You do not set this manually.

### How much data do you need?

The rule of thumb for character-level models is the Chinchilla scaling principle adapted to this architecture: you want roughly 20 tokens of training data per model parameter for efficient training.

| Target Parameters | Minimum Characters | Recommended Characters |
|---|---|---|
| 5M | 50M | 100M |
| 10M | 100M | 200M |
| 25M (current) | 250M | 500M |
| 30M | 300M | 600M |

NanoLens was trained on 4.4M characters for a 25M parameter model, which is significantly below these recommendations. The model still converges to val loss 1.1144 because Dostoevsky's prose is dense and repetitive enough to provide sufficient signal, but more data would improve generalisation. If you are training on a domain with more varied vocabulary, follow the recommendations above more strictly.

**Signs you need more data:**
- Training loss continues falling but val loss plateaus early
- Train/val gap exceeds 0.1 at convergence
- Generated output lacks coherent long-range structure

**Signs your data is sufficient:**
- Train and val loss curves track closely through training
- Val loss continues improving past step 2000
- Train/val gap stays below 0.08 at convergence

### Data quality matters more than quantity at small scale

Clean text, consistent encoding (UTF-8), and domain coherence matter more than raw character count for a model this size. A 2M character corpus of high-quality domain-specific text will train better than a 10M character corpus of mixed-domain noise.

---

## Configuration

All hyperparameters live in `config/default.yaml`. Nothing is hardcoded in the model files. Edit this file before training.

```yaml
batch_size : 128
block_size : 256
max_iters : 5000
eval_interval : 300
learning_rate : 3e-4
eval_iters : 200
n_embd : 512
n_head : 8
n_layer : 8
dropout : 0.3
n_train_test_split : 0.9
weight_decay : 0.1
```

### Hyperparameter guide

**n_embd** — embedding dimension. This is the width of the model. Every token is represented as a vector of this size throughout the network. Increasing this increases parameters quadratically because attention projections, feedforward layers, and the language model head all scale with n_embd. The feedforward layer uses 4 times n_embd internally, so a jump from 512 to 768 increases feedforward size from 2048 to 3072. Safe values: 128, 256, 384, 512, 768, 1024.

**n_head** — number of attention heads per layer. Must divide evenly into n_embd. the size of each head equals n_embd divided by n_head **( head_size = n_embd/n_head )**, and head size needs to be even and non-zero.
With n_embd 512 and n_head 8, each head operates in a 64-dimensional subspace. More heads means each head operates in a smaller subspace but there are more of them running in parallel. Do not go below 4 heads unless for research purposes. Common values: 4, 6, 8, 12, 16.

**n_layer** — number of transformer blocks or simply layers. This is the depth of the model. More layers gives the model more processing steps but increases training time linearly. The NanoLens research findings are specific to 8 layers. If you change n_layer you will get different circuit behaviour and the documented findings will not transfer directly. For research replication keep n_layer at 8. For your own experiments, 4 to 12 layers is a reasonable range at this scale.

**block_size** — context window in characters. The model can attend to this many previous characters when making predictions. Increasing this allows the model to use longer-range context but increases memory usage quadratically because attention matrices scale as block_size squared. At 256 with 8 layers and n_embd 512, memory is manageable on a T4. Going to 512 will roughly quadruple attention memory requirements.

**batch_size** — number of sequences processed per gradient step. Larger batch sizes give more stable gradient estimates but require more memory. On a Colab T4 with the current architecture, 128 is near the ceiling. If you reduce n_embd or block_size you can increase batch_size. If you increase either, reduce batch_size first.

**learning_rate** — peak learning rate used during the cosine decay schedule warmup. 3e-4 is standard for Adam-family optimizers on models of this scale. If your loss is unstable in early training, reduce to 1e-4. If training is very slow to start, you can try 5e-4 but watch for instability.

**dropout** — regularization. Set to 0.3 for the current model after the first training run overfit with lower values. If your model is underfitting (val loss much higher than it should be given data size), reduce dropout to 0.1 or 0.2. If it is overfitting (val loss rising while train loss falls), increase to 0.4.

**weight_decay** — applied to weight matrices (dim >= 2) via AdamW. Biases and LayerNorm parameters do not receive weight decay. 0.1 is a standard value. Do not change this unless you have a specific reason.

**max_iters** — total training steps. At 5000 steps with batch_size 128 and block_size 256 the model sees approximately 163 million character-level training examples. More steps will continue improving loss up to a point determined by model capacity and data size. Watch the val loss curve: if it has plateaued before max_iters you can stop early.

**eval_interval** — how often to print train and val loss. Does not affect training. Set lower if you want more granular loss curves.

**n_train_test_split** — fraction of data used for training. 0.9 means 90 percent training, 10 percent validation. Do not go below 0.85 unless your corpus is very large.

---

## The Learning Rate Schedule

NanoLens uses cosine decay with linear warmup. This is not optional or configurable through the YAML — it is implemented in `trainer.py`.

**Linear warmup** runs for the first 100 steps. The learning rate starts near zero and rises linearly to the peak learning_rate. This prevents large gradient updates early in training before the model has established any structure.

**Cosine decay** runs from step 100 to max_iters. The learning rate follows a cosine curve from peak learning_rate down to learning_rate divided by 10. This smooth decay allows the model to make large updates early and progressively finer adjustments as training proceeds.

The minimum learning rate at the end of training is always learning_rate divided by 10. With learning_rate 3e-4 the floor is 3e-5.  
This is intentional: keeping a small nonzero learning rate at the end prevents the model from stalling completely before convergence.

If you are training for more than 5000 steps, the cosine curve stretches automatically. The schedule adapts to whatever max_iters is set to.

---

## Scaling the Architecture

### Parameter count formula

Total parameters in NanoLens scale approximately as:

```
params = vocab_size * n_embd                        # token embedding
       + block_size * n_embd                        # position embedding
       + n_layer * (
           4 * n_embd * n_embd                      # attention QKV + proj
           + 2 * n_embd * 4 * n_embd                # feedforward
           + 4 * n_embd                             # layernorm params
         )
       + n_embd * vocab_size                        # lm_head
```

For the current configuration (n_embd = 512, n_layer = 8, vocab_size = 100):

```
token_emb  = 100 * 512          =    51,200
pos_emb    = 256 * 512          =   131,072
per_layer  = 4 * 512 * 512      = 1,048,576   (attention)
           + 2 * 512 * 2048     = 2,097,152   (feedforward)
           + 4 * 512            =     2,048   (layernorm)
           = 3,147,776 per layer
8 layers   = 8 * 3,147,776      = 25,182,208
lm_head    = 512 * 100          =    51,200
total      ≈ 25,415,680
```

*The small difference from the reported 25,441,380 comes from bias terms and LayerNorm scales not accounted for in this simplified formula. The formula is accurate enough for planning purposes.*

### Scaling targets

| n_embd | n_layer | n_head | Approx Parameters |
|---|---|---|---|
| 256 | 6 | 4 | ~5M |
| 384 | 6 | 6 | ~11M |
| 512 | 8 | 8 | ~25M (current) |
| 640 | 8 | 8 | ~39M |
| 768 | 8 | 8 | ~56M |
| 512 | 12 | 8 | ~38M |
| 768 | 12 | 12 | ~85M |

### What to change to scale up

To go from 25M to approximately 30M to 40M parameters on a Colab T4:

Increase n_embd from 512 to 640. Keep n_layer at 8. Keep n_head at 8 (head_size becomes 80, which is valid). Reduce batch_size from 128 to 96 or 64 to compensate for increased memory. This gives approximately 39M parameters.

Alternatively, keep n_embd at 512 and increase n_layer from 8 to 12. This gives approximately 38M parameters with the same width but more depth. The attention behaviour documented in the research findings will change because the layer distribution shifts.

## Theoretical Scaling Limits

How far can NanoLens scale with its current training setup — 
character-level tokenization, AdamW with cosine decay, and the 
decoder-only transformer architecture?

**The architecture has no hard ceiling.** The transformer scales 
arbitrarily in theory. The practical limits come from four 
independent constraints that interact: data, memory, optimization 
stability, and tokenization efficiency.

#### Constraint 1 — Data

Character-level models require substantially more data than subword 
models to learn the same linguistic patterns because the model must 
learn word structure, morphology, and semantics from individual 
characters rather than from pre-segmented tokens. The empirical 
observation from prior character-level work is that performance 
improvements from scaling parameters plateau when data is 
insufficient, regardless of model size.

Using the Chinchilla 20 tokens per parameter guideline:

| Parameters | Required Characters | Equivalent Words (approx) |
|---|---|---|
| 25M (current) | 500M | 83M |
| 50M | 1B | 167M |
| 100M | 2B | 333M |
| 200M | 4B | 667M |
| 500M | 10B | 1.7B |

Project Gutenberg contains approximately 60,000 books totalling 
roughly 20 billion characters. This is sufficient to train a 
character-level model up to approximately 1 billion parameters 
under Chinchilla recommendations. Beyond that, data becomes the 
binding constraint.

**Practical limit from data alone: approximately 1B parameters 
with a large enough corpus.**

#### Constraint 2 — Optimization Stability

AdamW with cosine decay is robust but not infinite. Empirically, 
character-level transformer training with this optimizer becomes 
increasingly sensitive to learning rate and batch size as model 
size grows. The key ratio to maintain is:

effective_batch_tokens = batch_size * block_size

For the current model: 128 * 256 = 32,768 tokens per step.

As parameters scale, this effective batch size needs to scale 
proportionally to keep gradient estimates stable. The linear 
scaling rule for learning rate says: if you double the batch size, 
multiply the learning rate by sqrt(2). The cosine decay schedule 
handles this automatically if learning_rate is tuned correctly.

Practical guidance for scaling:

| Parameters | Recommended batch_size | Recommended block_size | Peak LR |
|---|---|---|---|
| 25M (current) | 128 | 256 | 3e-4 |
| 50M | 256 | 512 | 2e-4 |
| 100M | 512 | 512 | 1e-4 |
| 200M | 1024 | 1024 | 7e-5 |

Beyond approximately 500M parameters, the cosine decay schedule 
typically needs to be replaced with more sophisticated schedules 
such as warmup-stable-decay or multi-cycle cosine. The single-cycle 
cosine used here works well up to roughly 100M parameters with 
careful tuning.

**Practical limit from optimization alone: approximately 100M to 
200M parameters with the current schedule. Beyond that, schedule 
modifications are needed.**

#### Constraint 3 — Tokenization Efficiency

Character-level tokenization is the binding architectural constraint 
for very large models. The reason is sequence length. A sentence of 
50 words is approximately 250 characters. At block_size 256 the 
model sees roughly one sentence of context. A subword model with 
the same block_size sees approximately 5 to 8 sentences.

As the model scales and gains capacity to learn longer-range 
dependencies, the character-level tokenization prevents it from 
actually using that capacity because the context window in characters 
does not grow proportionally with linguistic complexity. Increasing 
block_size compensates but quadruples attention memory per doubling.

The practical consequence: character-level models hit a quality 
ceiling relative to subword models at approximately 100M to 200M 
parameters because the tokenization efficiency gap becomes larger 
than what additional parameters can overcome. Beyond this scale, 
switching to byte-pair encoding or a learned tokenizer is the 
correct architectural choice.

**Practical limit from tokenization: approximately 100M to 200M 
parameters before subword tokenization becomes strictly necessary 
for quality.**

#### Constraint 4 — Hardware

On accessible hardware without distributed training:

| Hardware | VRAM | Practical Parameter Ceiling |
|---|---|---|
| Colab T4 | 15 GiB | ~40M parameters |
| Colab A100 | 40 GiB | ~150M parameters |
| 2x A100 | 80 GiB | ~400M parameters |
| 8x A100 | 320 GiB | ~1.5B parameters |

These estimates assume the current training setup without gradient 
checkpointing or mixed precision training. With fp16 mixed precision, 
the parameter ceiling approximately doubles on each hardware tier. 
With gradient checkpointing, memory scales sub-linearly with model 
depth at a 30 to 40 percent compute overhead.

**Practical limit on a single Colab T4 with current setup: 
approximately 40M parameters.**

#### The Unified Estimate

Taking all four constraints together:

| Constraint | Practical Ceiling |
|---|---|
| Data (Chinchilla, large corpus) | ~1B parameters |
| Optimization (cosine decay) | ~100M to 200M parameters |
| Tokenization (character-level quality) | ~100M to 200M parameters |
| Hardware (single T4) | ~40M parameters |

**The binding constraint for most users of this codebase is 
hardware, capping useful scale at approximately 40M parameters 
on a single T4. With an A100 and corpus expansion, the architecture 
scales meaningfully to 100M to 150M parameters before tokenization 
efficiency becomes the quality ceiling. Beyond 200M parameters, 
switching to subword tokenization and a more sophisticated training 
schedule is the correct path forward rather than scaling NanoLens 
further.**

The current 25M parameter model sits well within all four 
constraints and is not hardware-limited, data-limited, or 
optimization-limited in any fundamental sense. It is undertrained 
relative to Chinchilla recommendations, which means the primary 
lever for improving the current checkpoint is more data, not more 
parameters.

### The Colab T4 ceiling

The current architecture uses approximately 12.7 GiB of T4 GPU memory during training. The T4 has 15 GiB total, leaving about 2 GiB headroom. The primary memory consumers are:

- Activations stored for backpropagation: scales with batch_size, block_size, n_embd, and n_layer
- Model parameters: scales with the formula above
- Optimizer states: AdamW stores first and second moment estimates, approximately 2x parameter count in additional memory

To push beyond the current ceiling on a T4, reduce batch_size first. Going from 128 to 64 frees significant activation memory and allows n_embd to increase to 640 or 768. Gradient accumulation can compensate for the smaller effective batch size.

For architectures above approximately 60M parameters, upgrade to a Colab A100 (40 GiB) or use gradient checkpointing, which trades compute for memory by recomputing activations during backpropagation rather than storing them.

### How much data to scale with parameters

Using the 20 tokens per parameter guideline:

| Target Parameters | Recommended Characters |
|---|---|
| 10M | 200M |
| 25M (current) | 500M (current data is 4.4M, undertrained) |
| 30M | 600M |
| 40M | 800M |
| 60M | 1.2B |

NanoLens at 25M with 4.4M characters is approximately 100x undertrained relative to the Chinchilla recommendation. The model converges because Dostoevsky's repetitive prose provides dense signal, but a more diverse or larger dataset would produce a model with better generalisation. For your own domain, follow the recommended column.

---

## CLI Reference

### Dry run

```bash
python cli.py -d
```

Runs all system checks without training or loading a checkpoint. Use this first to verify the environment is configured correctly.

### Train from scratch

> Refer to CLI help using `python cli.py -h` for information on each flag

```bash
python cli.py
```

Trains a new model using the current config/default.yaml. Checkpoints are saved to checkpoints/ every 500 steps and at completion. Checkpoint filenames encode the architecture: `nanolens_L{n_layer}_H{n_head}_E{n_embd}_{step}.pt`

### Resume from checkpoint

```bash
python cli.py --resume 3000 --checkpoint checkpoints/nanolens_L8_H8_E512_3000.pt
```

Resumes training from step 3000. The learning rate schedule continues from the correct point in the cosine curve. Always pass both `--resume` with the step number and `--checkpoint` with the path.

### Generate from checkpoint

```bash
python cli.py --checkpoint checkpoints/nanolens_L8_H8_E512_5000.pt -g -t 500
```

Generates 500 tokens from the trained checkpoint. Change `-t` to control generation length. Generation uses the trained weights with no gradient computation.

### Inspect attention patterns

```bash
python cli.py --checkpoint checkpoints/nanolens_L8_H8_E512_5000.pt \
              -i "Your prompt here" -aw
```

Runs a forward pass on the prompt, captures all 64 attention weight matrices, and saves heatmap PNGs to results/attention/. Filenames follow the pattern `L{layer}_H{head}.png`.

### Inspect hidden state norms

```bash
python cli.py --checkpoint checkpoints/nanolens_L8_H8_E512_5000.pt \
              -i "Your prompt here" -hs
```

Runs a forward pass, captures hidden states via PyTorch forward hooks at each block, and saves norm trajectory plots, delta plots and cosine similarity heatmap to results/hidden_states/.

### Inspect both simultaneously

```bash
python cli.py --checkpoint checkpoints/nanolens_L8_H8_E512_5000.pt \
              -i "Your prompt here" -aw -hs
```

Both mechanisms run in a single forward pass. All plots are saved to their respective results subdirectories.

### Full flag reference

| Flag | Long form | Type | Default | Purpose |
|---|---|---|---|---|
| -d | --dry_run | bool | False | Run system checks |
| -c | --checkpoint | str | None | Path to checkpoint file |
| -g | --generate | bool | False | Generate text |
| -t | --tokens | int | 500 | Tokens to generate |
| -i | --inspect | str | None | Prompt for inspection |
| -aw | --attention_weights | bool | False | Save attention heatmaps |
| -hs | --hidden_states | bool | False | Save hidden state plots |
| --resume | --resume | int | 0 | Step to resume from |

---

## Checkpoints

Checkpoints are saved automatically during training:
- Every 500 steps to `checkpoints/` ( a tunable number)
- At step max_iters on completion
- If Google Drive is mounted at `/content/drive/MyDrive/nanolens/`, also saved there automatically

Checkpoint filenames encode the architecture used to create them. Always load a checkpoint with the same config/default.yaml settings that produced it. Loading a checkpoint trained with n_embd 512 into a model configured for n_embd 384 will fail.

The pretrained NanoLens v1.0 checkpoint is available at: [Releases](https://github.com/AdityaSinghDevs/nanolens/releases/tag/v1.0)


---

## Training on Google Colab

The recommended Colab workflow for training NanoLens:

```python
# Mount Drive for checkpoint persistence
from google.colab import drive
drive.mount('/content/drive')

# Clone and install
!git clone https://github.com/AdityaSinghDevs/nanolens
%cd nanolens
!pip install -r requirements.txt

# Verify setup
!python cli.py -d

# Train
!python cli.py
```
Checkpoints save to both local Colab storage and Drive automatically when Drive is mounted. If your Colab session disconnects, resume with:

```python
!python cli.py --resume 3000 \
               --checkpoint checkpoints/nanolens_L8_H8_E512_3000.pt
```

You may refer to [Colab Run Demo](https://github.com/AdityaSinghDevs/nanolens/notebooks/colab-30Mparam-train-run.ipynb) 

### Colab memory management

The current architecture uses 12.7 GiB on a T4 GPU (15 GiB total). If you hit out-of-memory errors:

1. Reduce batch_size from 128 to 64 first. This is the most effective single change.
2. Reduce block_size from 256 to 128 if still OOM.
3. Reduce n_embd if you need to reduce model size.

If upgrading to a larger architecture, use Colab Pro with A100 access (40 GiB) for architectures above approximately 60M parameters.

---


## Interpretability Toolkit
> For the full analysis methodology and documented findings from the 
> pretrained checkpoint, the research_findings/ directory is the 
> primary reference. This section covers the technical usage only.

Once you have a trained checkpoint, the inspection infrastructure exposes two types of internal data through a single forward pass.

**Attention weights** are captured via return_weights propagation through the model. Every head's (T, T) attention matrix is returned after softmax, showing which tokens attended to which at each layer. Saved as heatmap PNGs to results/attention/.

**Hidden states** are captured via PyTorch forward hooks registered on each Block. The full (1, T, n_embd) residual stream state after each block is stored in a dictionary keyed by block index. Norm trajectories and cosine similarity plots are saved to results/hidden_states/.

Both run in a single call to inspect() in analysis/inspector.py. The CLI flags -aw and -hs control which plots are generated. Using both flags together adds no extra computation.

For detailed methodology, findings from the pretrained checkpoint, and interpretation guidance, see:

- [research_findings/attention_circuit_analysis.md](research_findings/attention_circuit_analysis.md)
- [research_findings/hidden_state_analysis.md](research_findings/hidden_state_analysis.md)
- [research_findings/conclusions.md](research_findings/conclusions.md)

The documented findings from the 25M parameter Dostoevsky checkpoint serve as a reference for what patterns to look for when inspecting your own trained model. Different data and different architectures will produce different circuits. The classification framework and the dual-mechanism inspection approach transfer to any checkpoint trained with this codebase.
