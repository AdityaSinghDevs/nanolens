# NanoLens :  Attention Circuit Analysis

## Epistemic Notes
> These findings are from a single trained checkpoint on a single prompt. They are directional and exploratory, presented as observations from a character-level model and not as claims about transformers in general. The patterns are consistent with prior interpretability literature, which lends them credibility, but further experiments are needed before stronger claims can be made.

> This document classifies individual attention head behaviour and 
identifies functional specialisation patterns across 64 heads. 
Full circuit tracing, meaning the identification of compositional relationships between heads across layers, requires activation patching experiments.

## Head Type Definitions

**Previous token heads :**  attention heads where every token attends almost exclusively to the token immediately before it, tracking local sequence order.

**Identity preservation heads :** heads where every token attends primarily to itself, preserving the token's own representation without mixing in context.

**Word boundary heads :** heads where attention concentrates on space characters, using word boundaries as structural anchors in the sequence.

**Punctuation separator heads :** heads where attention concentrates specifically on punctuation characters, treating commas and periods as clause boundaries.

**First token sink heads :** heads where nearly every token routes attention back to the first character of the sequence, using position 0 as a global information aggregator.

**Abstract routing heads :** heads with sparse, high-contrast, non-local attention patterns where tokens attend to semantically relevant positions regardless of distance.


## 1. Headline Findings

1. The exact convergence point where abstract routing becomes dominant across all eight heads simultaneously, while local and global signals persist as secondaries, lies at **Layer 3**. It's not gradual transition. It is a commit.

2. Previous token heads at **Layer 2** show extended lookback range, attending to 2-3 positions back rather than one, suggesting the model widens its local context window in the layer immediately before the layer 3 transition.

3. From Layer 3 onward, three circuit families run simultaneously in every layer without exception: abstract routing, global aggregation, and boundary detection. the model does not pipeline these sequentially. It runs them in parallel at every depth past the transition point.

4. Every circuit type except local attention appears first as a secondary pattern in earlier layers before becoming dominant, The model rehearses new behaviours before comitting.

5. The First token sink head circuits follow a complete lifecycle: weak background signal from layer 1, growing distributed presence through layers 2 to 4, crystallizing into dedicated specialist heads at Layer 5 and 6, then dispersing back to secondary status at Layer 7.

6. Word boundary heads in Layers 1 through 4 attend to first space in the sequence only. From layer 5 onward they shift to later, syntactically  meaningful spaces and word-edge positions. The same circuit type becomes progressively more selective with depth.

7. Punctuation separator head appears as secondary in exactly two layers, 4 and 5, and nowhere else. This narrow mid-network window suggests punctuation-as-boundary is a transitional computation, active only during the shift from local to abstract processing, not a persistent architectural feature.

8. Word boundary detection, marked by attention to space characters, is present in every layer from 1 through 7 but never produces a single unambiguous specialist head, it is a permanently distributed computation, Not a localisable circuit.

9. Self-attention and identity preservation are exclusive to layer 0, with one secondary appearance in layer 1. Once contextual representations are built, the model has no further use for heads that attend primarily to the token itself.

10. Layer 7 has the highest proportion of heads with a clear dominant type and the lowest proportion of ambiguous mixed patterns of any layer in the model, including layer 0. The final layer is the most functionally committed.

---

## 2. Classification Grid

| Layer | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 |
|---|---|---|---|---|---|---|---|---|
| 0 | D<sup>*</sup><sub>(X)</sub> | P | P<sup>*</sup> | P | D<sub>(P)</sub> | P<sup>*</sup> | P | P |
| 1 | P<sub>(V)</sub> | P<sub>(D)</sub> | P<sup>*</sup> | P<sub>(3rd token)</sub> | V<sub>(BOS,X)</sub> | V | P<sup>*</sup> | S |
| 2 | S<sub>(BOS)</sub> | P<sub>(3rd&2nd)</sub> | P<sub>(BOS)</sub> | P<sub>(4th&3rd)</sub> | P<sub>(S)</sub> | P<sub>(3-4)</sub> | S<sub>(BOS,V)</sub> | S<sup>*</sup> |
| 3 | S<sub>(P)</sub> | S<sub>(BOS,V)</sub> | S<sub>(BOS,P)</sub> | S<sub>(BOS)</sub> | S<sub>(P)</sub> | S<sub>(V)</sub> | S<sub>(P)</sub> | S<sup>*</sup> |
| 4 | S<sub>(BOS)</sub> | S | S<sub>(BOS)</sub> | S<sub>(BOS,V,Sep)</sub> | S<sup>*</sup> | S<sub>(V,BOS)</sub> | S | S<sub>(V,BOS)</sub> |
| 5 | S<sub>(BOS)</sub> | S<sub>(V)</sub> | S<sub>(BOS,V)</sub> | S<sub>(P)</sub> | S<sub>(Sep,BOS)</sub> | BOS<sub>(V)</sub> | BOS<sup>*</sup> | S<sub>(V)</sub> |
| 6 | S<sub>(V,BOS)</sub> | S<sup>*</sup> | S<sub>(BOS,V)</sub> | BOS<sup>*</sup> | S | S<sub>(BOS,V)</sub> | S<sub>(BOS)</sub> | S<sub>(BOS)</sub> |
| 7 | S | S<sub>(BOS)</sub> | S<sub>(V)</sub> | S<sub>(BOS)</sub> | S<sub>(BOS)</sub> | S<sub>(BOS,V)</sub> | S<sup>*</sup> | S<sup>*</sup><sub>(BOS)</sub> |
---


## 3. Head Type Distribution by Layer

| Layer | P | D | V | Sep | BOS | S | X |
|---|---|---|---|---|---|---|---|
| 0 | 6 (2*) + 1sub | 2 (1*) | — | — | — | — | 1sub |
| 1 | 5 (2*) + 1sub | 1sub | 2 (1sub) | — | 1sub | 1 | 1sub |
| 2 | 5 (1sub) | — | 1sub | — | 3sub | 3 (1*) | — |
| 3 | 4sub | — | 2sub | — | 3sub | 8 (1*) | — |
| 4 | — | — | 3sub | 1sub | 5sub | 8 (1*) | — |
| 5 | 1sub | — | 4sub | — | 2 (1*) + 3sub | 6 | — |
| 6 | — | — | 3sub | — | 1* + 5sub | 7 (1*) | — |
| 7 | — | — | 2sub | — | 5sub | 8 (2*) | — |

**Legend:** `*` = strong unambiguous example  `sub` = secondary/mixed behaviour  `—` = absent

## Legend

| Symbol | Full Name | What It Means |
|---|---|---|
| P | Previous Token | Sub-diagonal stripe — every token attends to the one before it (local sequence tracking) |
| D | Diagonal | Main diagonal dominant — token attends primarily to itself (identity preservation) |
| V | Vertical Stripes | Bright columns at space characters — word boundary detection (structural anchoring) |
| Sep | Separator | Vertical stripes at punctuation specifically — clause boundary detection |
| BOS | First Token Sink | Entire left column lit — every token routes attention back to position 0 (global context aggregation) |
| S | Sparse Scattered | High contrast non-local hits — abstract feature routing (semantic relationship encoding) |
| X | Diffuse | Broadly distributed low contrast attention — signal degrades across sequence length |
| * | Strong Example | Clearest unambiguous representative of that type in the model |
| (secondary) | Mixed Behaviour | Head exhibits secondary pattern alongside dominant type |

---

## 4. Layer Summary

| Layer | Dominant Type | Character |
|---|---|---|
| 0 | P, D | Pure local — previous token and self attention |
| 1 | P, V emerging | Local with first boundary detection appearing |
| 2 | P transitioning to S | Local giving way — BOS secondary signal begins |
| 3 | S | Convergence point, abstract routing commits across all 8 heads simultaneously |
| 4 | S, BOS emerging | Abstract dominant — dedicated BOS sink heads appearing |
| 5 | S, BOS | Mixed abstract — BOS sink heads fully formed |
| 6 | S, BOS | Abstract — sparse routing and BOS coexisting |
| 7 | S | Fully abstract — semantic routing across all heads |


## 5. Sharp Examples by Type

### Previous Token Head — L0_H6

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L0_H6.png" width="60%">
</div>

**What you see:** A near-perfect sub-diagonal stripe running the full length of the sequence, every token attending almost exclusively to the token immediately before it, consistent from position 1 to the final token without degradation.

**Mechanism:** This head has learned to propagate local sequence order with high precision. By routing each token's attention to its predecessor, it builds a chain of positional dependencies that encodes sequential structure into the residual stream at layer 0.

**Literature:** Previous token heads are among the most consistently documented head types in transformer interpretability research. Elhage et al. (2021) identifies them as foundational components of the induction circuit, where a previous token head in an early layer provides the shifted context that induction heads in later layers use for pattern completion.

**Model specific:** L0_H6 is the cleanest previous token head in NanoLens, showing no degradation across sequence length and minimal secondary signal. Its presence in layer 0 confirms that local sequential structure is the first and most precisely learned behaviour in this model.

---

### Identity Preservation Head — L0_H0

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L0_H0.png" width="60%">
</div>

**What you see:** A bright main diagonal in the upper left of the heatmap, every token attending most strongly to itself, with a soft backward wedge of decreasing attention fading into recent history. The diagonal degrades progressively for later tokens, dissolving into broadly distributed low-contrast attention by the final third of the sequence.

**Mechanism:** This head preserves each token's own identity while maintaining a short contextual window into recent history. It says "I matter most, but I remember where I came from." The degradation across sequence length suggests identity preservation becomes harder to maintain as competing contextual signals accumulate.

**Literature:** Identity preservation heads, also called current token heads, are documented in the circuits framework as norm-preserving components that pass information through the attention layer without significant mixing. Their presence in layer 0 is expected and their degradation under long context is consistent with observations in small transformer models.

**Model specific:** L0_H0 is the only identity preservation head in NanoLens. It appears exclusively in layer 0 with one secondary appearance in layer 1, confirming that once contextual representations are built the model has no further use for heads that attend primarily to the token itself.

---

### Word Boundary Head — L1_H5

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L1_H5.png" width="60%">
</div>

**What you see:** Vertical stripes of concentrated attention at space characters, every token in the sequence attending strongly to the space that precedes its word, producing a column pattern that aligns precisely with word boundaries across the full prompt.

**Mechanism:** This head has learned that space characters are structural boundaries and uses them as positional anchors. In a character-level model with no explicit notion of words, this head has independently discovered word-level structure purely from statistical patterns in the training data.

**Literature:** The emergence of word boundary detection in character-level transformers is documented in prior work on character-level language models, which observed that attention heads implicitly learn word-level structure without explicit word-level supervision. NanoLens provides direct mechanistic evidence for this through a dedicated specialist head visible from layer 1.

**Model specific:** L1_H5 is the earliest and broadest word boundary head in NanoLens, attending to every space in the sequence. It represents the first appearance of structural boundary detection and is never matched by a head of equal sharpness at any later layer, consistent with the finding that word boundary detection remains permanently distributed rather than centralising into specialists.

---

### First Token Sink Head — L5_H6 and L6_H3

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H6.png" width="48%">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L6_H3.png" width="48%">
</div>

**What you see:** The entire left column is lit in both heads. Nearly every token routes strong attention back to the first character of the sequence regardless of its own position, producing a uniform vertical stripe at position 0 with minimal attention elsewhere.

**Mechanism:** The first token functions as a global information sink, accumulating sequence-level context that individual tokens can query. Because softmax attention cannot produce zero weight across all positions simultaneously, heads with no useful local signal to route default to a stable always-available position. Position 0 is never masked by the causal mask and serves this role consistently.

**Literature:** The first token sink phenomenon is explicitly documented in Elhage et al. (2021) as a widespread and stable pattern across transformer models. It emerges as a solution to the constraint that attention weights must sum to one, providing a reliable default routing target when no other position carries more relevant information.

**Model specific:** L5_H6 is the first dedicated first token sink head in NanoLens, appearing at layer 5 after four layers of BOS as background secondary signal. L6_H3 is the strongest BOS head in the model. Their presence across two consecutive layers confirms this is a stable load-bearing circuit, not a one-off pattern.

---

### Abstract Routing Head — L4_H4

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L4_H4.png" width="60%">
</div>

**What you see:** No diagonal, no vertical stripes, no column concentration. Sparse high-contrast hits at specific non-local positions scattered across the heatmap, with the majority of cells near zero and a small number of positions receiving strong concentrated attention.

**Mechanism:** This head is no longer tracking sequence order or structural boundaries. It is routing information based on learned abstract features, connecting tokens that share semantic or syntactic relevance regardless of their distance in the sequence. By layer 4 the model has built sufficient contextual representations to support this kind of non-local computation.

**Literature:** Sparse non-local attention patterns are associated with semantic routing and higher-order feature detection in the interpretability literature. Wang et al. (2022) documents similar sparse heads as components of the indirect object identification circuit in GPT-2, where specific heads route information between semantically related positions to support downstream prediction.

**Model specific:** L4_H4 is the clearest abstract routing head in NanoLens and notably the earliest layer where a strong unambiguous sparse head appears. Its presence at layer 4 rather than layers 6 or 7 suggests abstract routing begins earlier in this model than the layer summary alone implies, with layer 4 already supporting fully formed non-local computation.

---

### Extended Lookback Head — L2_H1

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L2_H1.png" width="60%">
</div>

**What you see:** A previous token pattern where the brightest attention sits not one step back but two to three positions back, producing a sub-diagonal stripe offset further from the main diagonal than standard previous token heads, with attention distributed across a small local window rather than concentrated on a single predecessor.

**Mechanism:** This head extends the local context window beyond the single-step lookback of standard previous token heads. Rather than asking "what came immediately before me" it asks "what came two or three steps back," widening the model's local temporal receptive field in the layer immediately before the layer 3 transition to abstract processing.

**Literature:** Extended lookback heads are less consistently documented than single-step previous token heads in the circuits literature. Their presence immediately before a major processing transition is consistent with the hypothesis that the model widens its local context window as preparation for committing to abstract routing, though this interpretation requires replication across prompts to confirm.

**Model specific:** L2_H1 is the clearest extended lookback head in NanoLens and its position at layer 2, the final layer before the layer 3 convergence, is the key observation. It suggests the model's transition from local to abstract processing is not abrupt but preceded by a deliberate widening of local context that may facilitate the handoff.

---
---
## 6. Deep Analysis

### Layer 3 is a Convergence Point, Not a Clean Switch

The dominant-only read of layer 3 suggests a hard phase transition where abstract routing 
jumps to all 8 heads while local attention drops. The subscript read tells a more accurate 
story. Abstract routing is dominant across all 8 heads in layer 3, but previous token 
behaviour persists as secondary in 4 heads and first token sink persists as secondary in 
3 heads simultaneously. Three functional systems are running in the same layer at the same 
time, with abstract routing having taken command. This is not a clean switch. It is a 
convergence layer where the model commits to abstract processing as primary while keeping 
local and global signals alive as background. No head in layer 3 is doing only one thing.

---

### Layer 2 Widens Local Context Before the Transition

Standard previous token heads attend one position back. Layer 2 heads H1 and H3 attend 
two to three positions back, extending the local receptive field beyond anything seen in 
layers 0 or 1.

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L0_H6.png" width="48%">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L2_H3.png" width="48%">
</div>

*Left: L0_H6, standard previous token head, single step back. Right: L2_H3, extended 
lookback head, two to three steps back.*

The shift is visible in the heatmaps: the bright stripe moves further from the main 
diagonal in L2_H1 compared to L0_H6. The model is not simply repeating layer 0 behaviour 
in layer 2. It is gathering more local context in the final layer before committing to 
abstract routing. Whether this is deliberate preparation or an emergent consequence of 
the layer 3 transition is unclear without further experiments, but the timing is precise 
enough to be notable.

---

### Three Circuit Families Run in Parallel From Layer 3 Onward

From layer 3 through layer 7, abstract routing heads, first token sink heads, and word 
boundary heads are present in every layer without a single exception. None disappears. 
None takes over completely. The distribution table makes this visible: S is dominant in 
every layer from 3 onward, BOS appears as secondary or dominant in every layer from 2 
onward, V appears as secondary in every layer from 1 onward. This is not a sequential 
pipeline where local circuits hand off to global circuits which hand off to abstract 
circuits. It is a parallel architecture where all three are active simultaneously at 
every depth past the transition point, with their relative prominence shifting across 
layers but their co-presence constant.

---

### Circuits Build as Secondaries Before Becoming Dominant

Every circuit type except local attention follows the same developmental pattern in this 
model. It appears first as a secondary signal in earlier layers, gets reinforced across 
multiple layers as background behaviour, and graduates to dominance only when 
sufficiently established. First token sink heads appear as secondary in layer 1, grow 
through layers 2 to 4, and become dominant at layer 5. Abstract routing appears as a 
single isolated head at layer 1, expands to 3 heads at layer 2, and commits across all 
8 heads at layer 3. Word boundary detection appears as secondary at layer 1 and never 
graduates to dominance at all, remaining distributed throughout. The model does not 
switch circuits on. It grows them.

---

### First Token Sink Heads Follow a Complete Lifecycle

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H6.png" width="48%">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L6_H3.png" width="48%">
</div>

*Left: L5_H6, first dedicated first token sink head. Right: L6_H3, strongest first token 
sink head in the model.*

The first token sink circuit does not simply appear at layer 5. It builds across six 
layers before reaching peak dedication and then disperses. Single secondary appearance 
at layer 1. Three heads carrying it as secondary at layers 2 and 3. Five heads carrying 
it as secondary at layer 4. Two dedicated specialist heads at layer 5. Peak dedication 
at layer 6 with L6_H3 as the strongest BOS head in the model. Dispersal back to 
secondary across five heads at layer 7. The circuit has a clear arc: accumulation, 
crystallisation, dispersal. Global context aggregation is most active in the middle of 
the network, not at the end, which is where you might naively expect it.

---

### Word Boundary Detection is Permanently Distributed

Word boundary heads appear in every layer from 1 through 7. They are never absent. They 
also never produce a single strong unambiguous specialist head across all 64 heads in 
the model. Every V head carries at least one secondary signal alongside it.

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L1_H5.png" width="48%">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H5.png" width="48%">
</div>

*Left: L1_H5, broad word boundary head attending to every space. Right: L5_H5, selective 
boundary head attending to syntactically meaningful spaces only.*

Contrast this with first token sink heads, which centralise into dedicated specialists 
at layers 5 and 6. Global context aggregation solved the centralisation problem. Word 
boundary detection did not. These are two different architectural solutions to two 
different problems. The model distributes boundary detection across many heads because 
no single head needs to own it completely — many heads benefit from knowing where word 
boundaries are, so many heads carry a piece of that signal. First token sink, by 
contrast, requires a dedicated aggregation point that other heads can query reliably, 
which drives centralisation.

---

### Word Boundary Heads Shift Their Target Across Depth

Early layers attend to the first space in the sequence. Later layers attend to 
syntactically meaningful spaces and word-edge positions. The circuit type is the same. 
The target changes. Layers 1 through 4 show word boundary heads pointing to the first 
space only, using it as a simple positional anchor. From layer 5 onward the same heads 
shift to later spaces, attending to boundaries before content words and at the edges of 
words rather than just the first delimiter in the sequence. The functional role of word 
boundary detection becomes more selective as the model builds richer representations 
deeper in the network.

---

### Punctuation Separator Heads are a Mid-Network Exclusive

Punctuation separator behaviour appears as secondary signal in exactly two layers, 4 
and 5, and is absent from every other layer in the model. It does not appear in early 
layers where local structure dominates. It does not appear in late layers where abstract 
routing dominates. It occupies a narrow window precisely at the transition between local 
and abstract processing. This suggests punctuation-as-boundary is a transitional 
computation rather than a persistent architectural feature, active specifically during 
the period when the model is shifting from syntactic to semantic processing.

---

### Identity Preservation is a Layer 0 Primitive

Identity preservation heads appear in layer 0 only, with a single secondary appearance 
in layer 1 and complete absence from all 48 remaining heads. The reason is functional. 
Before the model has built contextual representations, a head that attends primarily to 
the current token is useful — it preserves raw token identity in the residual stream 
while other heads begin building context. Once higher layers have enriched each token's 
representation with information from surrounding tokens, there is no value in a head 
that echoes the uncontextualised embedding. Identity preservation is needed only at the 
beginning. The model learns this and uses it only there.

---

### Layer 7 is the Most Functionally Committed Layer

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L7_H6.png" width="48%">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L7_H7.png" width="48%">
</div>

*Left: L7_H6, strong abstract routing head. Right: L7_H7, strong abstract routing head 
with first token sink as secondary. Both show clean dominant patterns with minimal 
ambiguity.*

Layer 7 has abstract routing as dominant across all 8 heads, two strong unambiguous 
examples, and the lowest proportion of ambiguous mixed patterns of any layer in the 
model including layer 0. This is counterintuitive. You might expect the final layer to 
do general mixing before projection to vocabulary, accumulating signals from all previous 
computations. Instead it runs highly committed abstract routing in every head, with first 
token sink and word boundary signals maintained quietly as background. The model's final 
attention layer is its most specialised. Integration and clean-up appear to happen in the 
residual stream and the final LayerNorm rather than in the attention heads themselves.

## 7. Annotations and Anomalies

**L1_H7 — Earliest abstract routing head in the model**
One isolated sparse head in layer 1 while every other head in that layer is still doing 
local attention. Whether this represents genuine early abstract computation or a head 
that failed to specialise cleanly into local structure is unclear. It is the only head 
in layers 0 or 1 showing sparse non-local behaviour and stands as an outlier in an 
otherwise fully local region of the network. Replication across prompts would clarify 
whether this is a stable feature or prompt-specific noise.

**L4_H1 and L4_H6 — Plain S with no secondary signal**
In layer 4, almost every head carries BOS or V as a secondary signal alongside abstract 
routing. L4_H1 and L4_H6 show plain S with no detectable secondary. This makes them 
anomalous for their layer. Either these heads are doing purer abstract routing than 
their neighbours, or their secondary signals are present but below the visual detection 
threshold of the heatmap. Worth examining with quantitative attention entropy metrics 
to determine whether their distributions are genuinely cleaner or simply noisier.

**L0_H0 — Diagonal degrades across sequence length**
The identity preservation pattern holds cleanly for the first 15 to 20 tokens and 
dissolves into broadly distributed low-contrast attention for later positions. This is 
the only head in the model that shows a clear within-head degradation pattern across 
sequence length. Every other head maintains its dominant pattern consistently from 
early to late positions. L0_H0 is the exception, suggesting identity preservation is 
specifically difficult to maintain as competing contextual signals accumulate with 
sequence length.

**Head position 7 transitions to abstract routing earlier than positions 0 to 6**
H7 shows abstract routing as dominant at layer 1, one full layer before any other head 
position commits to it. Positions H0 through H6 mostly hold local attention through 
layer 2 before committing at layer 3. Whether this reflects a systematic pattern in how 
the model distributes specialisation across head indices, or is coincidental at this 
scale, cannot be determined from a single prompt and single checkpoint. Flagged as 
an observation requiring replication.

**L1_H4 — V pattern begins at third token, not sequence start**
The word boundary head at L1_H4 shows vertical stripe behaviour starting from the third 
token position rather than from the beginning of the sequence. The first two tokens, 
R and a, receive no vertical stripe signal. This suggests the head requires a minimum 
context window before its boundary detection behaviour activates, or that the first word 
boundary only becomes detectable after the model has seen enough of the sequence to 
establish a baseline.

---

## 8. Literature Connections

### Previous Token Heads — Elhage et al. 2021

Elhage et al., A Mathematical Framework for Transformer Circuits, Anthropic 2021, 
explicitly documents previous token heads as one of the most consistent and reproducible 
head types across transformer models. They identify these heads as foundational components 
of the induction circuit, where a previous token head in an early layer provides the 
shifted sequence context that induction heads in later layers use for in-context pattern 
completion. NanoLens reproduces this finding precisely at layer 0, with six previous 
token heads and two strong unambiguous examples. The literature predicts their presence 
and their layer position. The data confirms both in a character-level model at 25 million 
parameters.

### First Token Sink Heads — Elhage et al. 2021

The first token sink phenomenon is documented in the same circuits framework as a 
widespread and stable pattern arising from a fundamental constraint of softmax attention. 
Because attention weights must sum to one across all positions, heads with no useful 
local signal to route cannot produce zero attention everywhere simultaneously. Position 0 
is always available, never masked by the causal mask, and serves as a reliable default 
routing target. Elhage et al. describe this as a load-bearing circuit for global context 
aggregation rather than an artifact. NanoLens shows the same pattern with an additional 
observation the original framework does not specifically describe at this scale: the 
circuit builds gradually as background secondary signal across four layers before 
crystallising into dedicated specialist heads, suggesting the model develops the 
aggregation mechanism incrementally rather than switching it on at a fixed depth.

### Layer Hierarchy — Clark et al. 2019, Tenney et al. 2019

Clark et al., What Does BERT Look At, 2019, and Tenney et al., BERT Rediscovers the 
Classical NLP Pipeline, 2019, both document a processing hierarchy in transformer 
models where early layers handle syntax and local structure while later layers handle 
semantics and long-range dependencies. These findings were established in large 
bidirectional encoder models trained on subword tokens, which differ from NanoLens in 
architecture, tokenisation strategy, training objective, and scale. NanoLens reproduces 
the same local-to-abstract gradient in a fundamentally different setting: character-level 
tokenisation, causal decoder-only attention, autoregressive training, and 25 million 
parameters. The consistency of this hierarchy across such different model families 
suggests it is a fundamental property of how transformers organise learned representations 
under gradient descent, not an artifact of any specific architectural choice.

### Word Boundary Detection in Character-Level Models

Prior work on character-level language models, including Al-Rfou et al. 2019 and early 
GPT work from Radford et al. 2018, observed that character-level models implicitly learn 
word-level structure through attention without explicit word-level supervision. NanoLens 
provides direct mechanistic evidence for this at the head level: a dedicated word boundary 
head visible from layer 1, using space characters as structural anchors across the full 
sequence. The additional finding that word boundary detection never centralises into 
unambiguous specialist heads across 64 heads, remaining permanently distributed, is not 
directly addressed in the existing character-level literature. It may reflect a 
consequence of character-level tokenisation specifically: subword models can rely on 
token boundaries directly, while character-level models must construct word boundaries 
from scratch and appear to distribute that computation across many heads rather than 
centralising it into a dedicated circuit.

### Layer 3 Convergence

The sharpness of the layer 3 transition in NanoLens, where every head commits to 
abstract routing as primary while simultaneously maintaining local and global signals 
as secondaries, does not have a direct published precedent at this model scale that 
can be identified with confidence. The circuits framework describes layer hierarchies 
and circuit composition but does not document a single convergence layer of this 
character in 8-layer models. Larger model studies such as Wang et al. 2022 on GPT-2 
indirect object identification show complex multi-layer circuit structures but describe 
gradual rather than convergent transitions. This is the observation in NanoLens most 
worth attempting to replicate, because if it holds across prompts and checkpoints it 
represents a specific claim about how small transformers organise their processing 
hierarchy that is not currently documented in the literature.

---

## 9. What Would Strengthen These Claims

The findings in this document are from a single trained checkpoint on a single prompt. 
The following experiments would move them from exploratory observations to defensible 
claims.

**Multiple prompts.** Running the same inspection on five to ten prompts with different 
lengths, vocabulary, and syntactic structures would establish whether head classifications 
are stable across inputs or prompt-dependent. Any finding that holds across all prompts 
is substantially more credible than one observed on a single input.

**Multiple checkpoints.** Training a second model from a different random seed and 
classifying its 64 heads would determine whether the same heads specialise in the same 
ways across runs, or whether specialisation patterns are random across seeds. Stable 
specialisation across seeds would suggest the patterns are learned deterministically 
from the data rather than being arbitrary solutions the optimiser happened to find.

**Attention entropy metrics.** Computing H = -sum(p log p) for all 64 heads quantitatively 
would replace visual classification with a measurable signal. Low entropy confirms 
focused heads. High entropy confirms diffuse heads. The layer 3 convergence finding 
could be expressed as a quantitative entropy drop rather than a visual observation, 
making it falsifiable and reproducible without access to the heatmaps.

**Single-head versus multi-head comparison.** Training a second model with n-head set 
to 1, same architecture and data, and comparing its single attention pattern to the 
multi-head patterns documented here would isolate what multi-head parallelism 
contributes. If the single-head model learns a blended version of multiple types, it 
confirms that parallelism enables functional specialisation. If it learns the same 
dominant type, it suggests specialisation is not driven by parallelism alone.

**Activation patching for circuit verification.** The findings here describe individual 
head behaviour. Verifying that heads compose into functional circuits, where the output 
of one head is used as input to another in a meaningful way, requires activation 
patching experiments. Replacing activations from one forward pass with those from 
another mid-computation would establish causal relationships between heads that visual 
classification cannot confirm. This is the methodological step that separates 
observation from mechanism and is the primary direction for PRISMA.
