# NanoLens — Hidden State Analysis

## Epistemic Note

> These findings are from a single trained checkpoint on a single prompt: *"Raskolnikov hesitated at the threshold, his hands trembling."* They are directional and exploratory, presented as observations from a character-level model and not as claims about transformers in general. The patterns are consistent with prior interpretability literature where comparable, but replication across prompts and checkpoints is needed before stronger claims can be made.

> This document analyses hidden state representations through norm trajectories, per-layer deltas, and layer-to-layer cosine similarity. Connections between these findings and attention circuit behaviour are drawn in [research_findings/conclusions.md](conclusions.md).

> **Layer Nomenclature**: This document uses 1-indexed layer numbering (Layer 1 through Layer 8), matching plot axes. Layer 1 corresponds to Layer 0 in the attention circuit analysis.

> **Literature connections** were identified with AI assistance. The claims made about each cited work reflect the documented findings of those papers to the best of the author's knowledge, but primary sources should be consulted directly for full context.

---

## Tokens Tracked

| Token | Position | Category | Context |
|---|---|---|---|
| R | 0 | BOS / first character | First character of sequence and "Raskolnikov" |
| space1 | 11 | Structural / word boundary | Space after "Raskolnikov" |
| h | 12 | Content / word start | First character of "hesitated" |
| space3 | 28 | Structural / word boundary | Space before "the" |
| d | 37 | Content / word end | Last character of "hesitated" |
| , | 38 | Punctuation | Comma after "hesitated" |
| space5 | 49 | Structural / word boundary | Space before "his" |
| . | 59 | Punctuation | Period, end of sequence |

Eight tokens were selected to cover distinct functional categories: R as the sequence-initial BOS position, h and d as content word characters at word-start and word-end positions, space1, space3, and space5 as structurally identical tokens in syntactically distinct positions, and comma and period as the two punctuation characters. The selection was designed to test whether token type or token position drives representational differences, and to provide comparison baselines across categories.

---

## Measurements and Rationale

Four measurements are used in this analysis. Each captures a different dimension of how token representations change across layers.

**Hidden state norm trajectory** is the L2 magnitude of a token's representation vector at each layer. A growing norm means the model is adding information to that token's representation through the residual stream. A flat norm means a layer left that token largely unchanged. A declining norm means the model actively reduced the magnitude of that representation, which is rare and meaningful when it occurs.

**Per-layer norm delta** is the change in norm between consecutive layers for each token. Where the norm trajectory shows the accumulated state, the delta shows where the work is actually being done. A large delta at layer N means layer N contributed significantly to that token's representation. A near-zero delta means that layer passed the token through without significant modification.

**Layer-to-layer cosine similarity** measures the angle between a token's representation vector at adjacent layers. A value near 1.00 means the direction of the representation did not change, only its magnitude. A lower value means the layer reoriented the representation, not just scaled it. This separates two distinct kinds of change: building a stronger signal in the same direction versus changing what the representation points toward.

---

## 1. Headline Findings

1. **R dominates from layer 1 and never yields that lead.** It starts with the highest norm of all tracked tokens and maintains that lead throughout all 8 layers, peaking at layer 6 before redistribution. No other token comes close at any layer.

2. **R's norm drops at layers 7 and 8, the only token in the entire analysis to show negative deltas.** This is not degradation. It is redistribution. The model has finished accumulating context into the BOS position and begins spreading that information outward toward prediction.

3. **Representational direction stabilises earlier than magnitude.** Cosine similarity between adjacent layers is above 0.93 for every token at every transition. The model locks in what tokens mean before it finishes building how strongly they are represented.

4. **R's direction is fully locked by layer 4**, showing 1.00 cosine similarity across three consecutive transitions: Layer 4 to Layer 5, Layer 5 to Layer 6, and Layer 6 to Layer 7. The BOS position functions as a stable directional anchor precisely because its direction stopped changing early.

5. **The final layer transition does more directional work than any middle layer transition.** Layer 7 to Layer 8 shows the lowest cosine similarity values across all tokens compared to any transition in layers 2 through 7. The final layer is not just refining. It is reorienting.

6. **Standard content tokens h and space1 grow at a steady near-linear rate across all 8 layers with no sharp transitions.** They track closely through layers 1 to 5 before diverging slightly in the final layers. These are the baseline trajectories against which all other tokens are interpreted.

7. **d, the end character of "hesitated," starts lowest and grows slowest, then accelerates sharply at layers 7 and 8.** End-of-word positions show late acceleration patterns similar to punctuation, suggesting the model defers processing of boundary-adjacent positions until structural relationships across the sequence have been resolved.

8. **The three space tokens are not equivalent despite identical token type.** space3, the space before "the," grows faster than space1 and space5 across all layers and ends highest at layer 8. Position and syntactic role drive representational weight, not token identity alone.

9. **Punctuation tokens show two distinct phases: anomalously high norm at layer 1, then a sharp spike at layers 7 and 8.** Through the middle layers they track flat. Early layers encode punctuation identity. Final layers encode punctuation function. These are separable computations at different depths.

10. **The layer 8 norm delta for comma and period is the largest delta of any token at any layer in the entire analysis**, exceeding even R's peak delta at layer 2. The final layer does more representational work on punctuation than any layer does on any other token type.

---

## 2. Plots

### 2.1 Content Token Norm Trajectories

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_trajectory_content.png" width="75%">
</div>

Tracks R, h, d, and space1 across all 8 layers. R dominates from layer 1 and separates from all other tokens immediately. h and space1 grow at near-identical rates through layer 5 before diverging slightly. d starts lowest and grows slowest through layer 6 before accelerating sharply in the final layers.

---

### 2.2 Structural Token Norm Trajectories

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_trajectory_structural.png" width="75%">
</div>

Tracks R, space1, space3, space5, comma, and period across all 8 layers. The three space tokens show diverging trajectories despite identical token type. space3 grows fastest and ends highest. Comma and period start with anomalously high norm at layer 1 relative to their middle-layer trajectory, then spike sharply at layers 7 and 8.

---

### 2.3 Per-Layer Norm Delta

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_deltas.png" width="75%">
</div>

Shows how much representational work each layer does per token. R shows the largest deltas in early layers and the only negative deltas in the model at layers 7 and 8. Punctuation tokens show the largest deltas of any token at any layer at layer 8, exceeding even R's peak.

---

### 2.4 Layer-to-Layer Cosine Similarity

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/cosine_similarity.png" width="75%">
</div>

Shows directional change in token representations between adjacent layers. All values are between 0.93 and 1.00. Representations change in magnitude far more than in direction. The lowest similarities appear at early transitions and at the final Layer 7 to Layer 8 transition. R shows 1.00 similarity across three consecutive middle-layer transitions, confirming directional lock-in by layer 4.

---

## 3. Deep Findings

### R Dominates From Layer 1

R starts with the highest hidden state norm of all tracked tokens at layer 1 with a value of approximately 54, already separated from every other token before a single layer of processing has occurred. This initial advantage compounds across every layer. By layer 6 R reaches approximately 87, a gap of over 30 norm units above the next highest token. No other token approaches this trajectory at any point.

The separation is immediate and structural. R occupies position 0, the only position that is never masked by the causal mask and is always available as an attention target. Every head that develops first token sink behaviour across layers 1 through 6 is routing information into R's representation. The norm growth is the residual stream accumulation of that routing.

---

### The Layer 6 Peak and Redistribution

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_deltas.png" width="75%">
</div>

R peaks at layer 6 and shows negative norm deltas at layers 7 and 8, the only token in the entire analysis to show negative deltas at any layer. Every other token grows monotonically across all 8 layers without exception.

This is not degradation. The model has finished accumulating context into the BOS position by layer 6. Layers 7 and 8 are the final processing layers before projection to vocabulary. The negative delta reflects the model redistributing information outward from R to other token positions as it moves toward prediction. The first token sink circuit identified in the attention analysis peaks at layer 6 and begins dispersing at layer 7. The norm trajectory confirms this timing precisely through an independent measurement.

---

### Representational Direction Stabilises Before Magnitude

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/cosine_similarity.png" width="75%">
</div>

Every cosine similarity value in the entire heatmap falls between 0.93 and 1.00. Across all 8 tokens and all 7 layer transitions, no representation undergoes a fundamental directional reorientation between adjacent layers. The model is scaling, enriching, and refining token representations across depth, not redirecting them.

The implication is specific. The norm plots show large magnitude increases across layers, with R growing from 54 to 87 and punctuation nearly doubling in the final layers. But the cosine similarity shows the direction of these vectors barely changes. The model decides what a token means early and spends subsequent layers building how strongly it represents that meaning. Direction first, magnitude second.

---

### R's Direction Locks by Layer 4

R shows cosine similarity of 1.00 across three consecutive layer transitions: Layer 4 to Layer 5, Layer 5 to Layer 6, and Layer 6 to Layer 7. For three full layers the directional component of R's representation does not change at all. Only its magnitude continues growing.

This is precisely why first token sink heads are stable and useful. Other tokens route attention back to R across layers 4 through 6 because R is a reliable, directionally fixed reference point. A token that is still changing direction would be an unstable aggregation target. R's directional lock-in is the mechanical property that makes the BOS sink circuit function as a global context anchor.

---

### End-of-Word Tokens Show Late Acceleration

d, the final character of "hesitated," starts with the lowest norm of all tracked tokens at layer 1 and grows most slowly through layers 1 to 6. From layer 7 onward it accelerates sharply, closing the gap with h and space1 and ending at approximately 52 at layer 8.

This late acceleration pattern matches punctuation behaviour. Both d and the punctuation tokens show suppressed middle-layer growth followed by a sharp final-layer spike. The common factor is sequence boundary position. d ends a content word. Comma and period end clauses and the full sequence. The model appears to defer heavy processing of boundary-adjacent positions until the final layers when structural relationships across the full sequence have been resolved.

---

### Not All Spaces Are Equal

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_trajectory_structural.png" width="75%">
</div>

space1 (position 11, after "Raskolnikov"), space3 (position 28, before "the"), and space5 (position 49, before "his") are identical token types. They receive the same initial embedding. By layer 8 their norms have diverged significantly. space3 ends highest at approximately 67, space1 at approximately 59, space5 at approximately 59.

space3 precedes "the," a function word that introduces the noun phrase "the threshold." space1 follows a proper noun. space5 precedes a possessive pronoun. The model assigns different representational weight to syntactically distinct boundary positions despite their identical token identity. This is direct evidence that the model's representations are context-dependent at the level of individual positions, not just token types.

This finding connects directly to the attention circuit analysis. Word boundary heads in layers 1 through 4 attend to the first space in the sequence only. From layer 5 onward they shift to syntactically meaningful spaces. The norm data shows the consequence of that shift: the spaces that receive more selective late-layer attention accumulate more representational weight by the final layer.

---

### Punctuation Has Anomalous Early Norm Before Late Spike

Comma and period start at layer 1 with norm values of approximately 32 and 31 respectively, higher than h at 30 and d at 29, despite punctuation having no special structural role in early layers. Through layers 2 to 6 they grow slowly, tracking below the space tokens. Then at layers 7 and 8 they diverge sharply upward, ending at approximately 61 and 66 respectively.

Two distinct phases. Early layers encode punctuation identity strongly from the start. The model knows immediately that these are structurally special characters even before it has built the representations needed to use that information. Late layers add the semantic and syntactic weight when the full sequence context has been processed. The early high norm is identity encoding. The late spike is functional encoding. These are separable computations at different depths.

---

### Layer 8 Does the Most Directional Work of Any Transition

The Layer 7 to Layer 8 row in the cosine similarity heatmap shows the lowest values across all tokens compared to any middle layer transition. d drops to 0.93, comma and period to 0.94, space3 to 0.93. These are the lowest similarity values in the entire grid outside of the very first transition.

This is counterintuitive. You would expect final layers to refine rather than reorient, to polish representations that are already directionally settled. Instead layer 8 introduces more directional change than layers 3 through 7 combined for several tokens. The final layer is doing something fundamentally different from middle layers, not just continuing the same refinement process.

This connects directly to the attention finding that layer 7 is the most functionally committed layer in the model. High attention commitment and high representational reorientation are occurring simultaneously in the final processing stage. The attention heads are running their most specialised patterns while the residual stream is undergoing its largest directional shift. These are two measurements of the same underlying event.

---

### Standard Content Tokens Are the Baseline

h (first character of "hesitated") and space1 (space after "Raskolnikov") grow at near-identical near-linear rates across all 8 layers. No spikes, no plateaus, no negative deltas. They track closely through layers 1 to 5 before diverging slightly in layers 6 to 8 with h pulling marginally ahead.

These tokens represent the baseline processing trajectory. Positions with no special structural role showing what normal representational growth looks like in this model. Every departure from this baseline in other tokens is meaningful: R's dominance, punctuation's late spike, d's slow start and late acceleration, space3's faster growth. The linear tokens provide the reference against which all other trajectories are interpreted.

---

### Punctuation Spike at Layer 8 Exceeds Every Other Token

The layer 8 norm delta for comma and period is the largest delta of any token at any layer in the entire analysis. It exceeds R's largest delta at layer 2. It exceeds every structural token at every layer. The final layer does more representational work on punctuation than any layer does on any other token type.

In a character-level autoregressive model predicting the next character, knowing whether the next character is punctuation or a letter is a high-stakes distinction. The model concentrates its final-layer representational work on the tokens most directly predictive of structural boundaries. The punctuation spike is not incidental. It is the model preparing for prediction.

---

## 4. Literature Connections

### Residual Stream Norm Growth — Elhage et al. 2021

The mathematical framework for transformer circuits describes the residual stream as an information highway where each layer reads from and writes to a shared vector. Norm growth across layers is the natural consequence of successive layers adding information to this stream. Elhage et al. does not make specific quantitative predictions about norm trajectories for different token types, but the framework predicts that tokens receiving more attention routing across layers will accumulate larger representations. This is precisely what the R dominance finding shows.

### Representational Geometry — Voita et al. 2019

Voita et al., Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned, 2019, documents that different attention heads contribute differently to token representations and that this contribution varies by position and token type. The finding that not all spaces receive equal representational treatment is consistent with this. Different boundary positions receive different attention routing, producing different norm trajectories despite identical initial embeddings.

### Layer Hierarchy and Representation Depth — Tenney et al. 2019

Tenney et al., BERT Rediscovers the Classical NLP Pipeline, 2019, documents that representations become increasingly abstract with depth in transformer models. The cosine similarity finding, that direction stabilises early while magnitude continues growing, is consistent with this. Early layers establish what a token means directionally. Later layers build the magnitude of that meaning. The model's representational geometry is set before its representational strength is fully developed.

### BOS Position Representations

The finding that R's direction locks completely by layer 4 while its magnitude continues growing through layer 6 does not have a direct quantitative precedent in the published literature at this model scale. The circuits framework predicts BOS sink behaviour in attention but does not make predictions about the norm and directional properties of the BOS position's hidden state trajectory. This is an observation that extends the existing literature rather than simply confirming it.

### Punctuation in Character-Level Models

Prior work on character-level language models including Al-Rfou et al. 2019 observed that such models develop implicit structural awareness without explicit supervision. The two-phase punctuation finding, anomalously high early norm followed by late-layer spike, provides mechanistic evidence for how this structural awareness develops. Punctuation identity is encoded early. Punctuation function is encoded late. These are separable computations at different depths, which the prior literature observed statistically but did not trace mechanistically.

---

## 5. What Would Strengthen These Claims

**Multiple prompts.** All findings here are from a single prompt of 57 characters. Running the same analysis on prompts of different lengths, different syntactic structures, and different vocabulary would establish whether norm trajectories and cosine similarity patterns are stable properties of the model or prompt-dependent artifacts. The space inequality finding is particularly in need of replication. Whether space3 always grows faster than space1 requires testing across contexts where the syntactic roles are reversed.

**Multiple checkpoints.** Training a second model from a different random seed and comparing norm trajectories would determine whether the specific magnitude values and crossing points are reproducible or arbitrary. The directional lock-in finding for R is the strongest candidate for replication. If 1.00 cosine similarity across three consecutive transitions appears in every trained checkpoint, it becomes a structural property of the architecture rather than a training artifact.

**Full sequence analysis.** The current analysis tracks 8 selected tokens. Running norm and cosine similarity analysis across all 57 tokens in the prompt would show whether the patterns identified here generalise across all positions or are specific to the selected representatives. A full heatmap of norm trajectories across all positions and all layers would be the definitive version of this analysis.

**Quantitative connection to attention weights.** The norm findings and attention findings are corroborated narratively but not quantitatively linked. Computing the correlation between attention weight received by R across layers and R's norm growth would make the BOS sink to norm growth connection precise rather than observational.

**Probing classifiers.** Training linear classifiers on hidden states at each layer to predict token category, punctuation versus content versus boundary, would establish at which layer the model linearly separates these categories. Combined with the norm findings, this would show whether the punctuation early norm reflects linear separability of punctuation identity or something more distributed.

---

## 6. Closing Note

Hidden state analysis complements attention analysis in a specific way. Attention maps show where information flows. Norm trajectories show how much accumulates. Cosine similarity shows when representations stabilise. Together they describe not just the routing of information through the model but the building of meaning: which positions matter, when they matter, and how the model prepares for prediction in its final layers.

The findings here are from one model, one prompt, one checkpoint. But they are grounded in the data, honest about their scope, and consistent with the attention circuit findings where the two analyses touch the same phenomena. That consistency across independent measurements is the strongest signal in this document.

The inspection infrastructure built for NanoLens makes these measurements accessible to anyone with the checkpoint and a prompt. What patterns emerge on different data, different architectures, and different scales is an open question.