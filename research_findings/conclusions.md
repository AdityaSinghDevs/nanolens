# NanoLens — Conclusions

## Epistemic Note

> This document synthesises findings from two independent analyses: [attention_circuit_analysis.md](attention_circuit_analysis.md) and [hidden_state_analysis.md](hidden_state_analysis.md). All caveats from both documents apply here. Findings are from a single trained checkpoint on a single prompt: *"Raskolnikov hesitated at the threshold, his hands trembling."* They are directional and exploratory, not claims about transformers in general.

> **Layer nomenclature**: This document uses 0-indexed layer numbering (Layer 0 through Layer 7), matching the attention circuit analysis, hidden state analysis, and image filenames throughout.

> **Literature connections** were identified with AI assistance. Primary sources should be consulted directly for full context.

---

## How to Read This Document

This document does not summarise the attention circuit analysis or the hidden state analysis. It synthesises them. Each section takes a finding from one analysis, pairs it with the corresponding finding from the other, and draws the conclusion that neither dataset could support alone. Readers who want the full evidence base for any individual claim should consult the source documents directly.

---

## What Was Found — In Brief

NanoLens is a ~25 million parameter character-level autoregressive transformer trained on 4.4 million characters of Dostoevsky. It has 8 layers, 8 attention heads per layer, and a 512-dimensional embedding space, producing 64 attention heads in total. This analysis inspects what those 64 heads learn and what happens to token representations as they flow through the model.

The central finding is that this model organises its computation into a clear hierarchy that runs from local sequential structure in the earliest layers to abstract semantic routing in the final layers. This hierarchy is not gradual. Layer 3 is a convergence point where abstract routing commits across all 8 heads simultaneously while local and global signals persist as secondaries. The hidden state data independently confirms this transition: R's norm delta is highest in layers 1 and 2 and begins tapering precisely as abstract routing commits at layer 3.

Six functionally distinct attention head types are documented across the 64 heads: previous token heads, identity preservation heads, word boundary heads, first token sink heads, abstract routing heads, and extended lookback heads. These types do not appear and disappear cleanly. Every circuit type except local attention appears first as a secondary signal in earlier layers before graduating to dominance. The model grows its circuits rather than switching them on.

The hidden state analysis shows the consequence of this circuit structure on token representations. R, occupying the sequence-initial position, accumulates the highest norm of any tracked token across all 8 layers, peaks at layer 5, and then undergoes norm redistribution in layers 6 and 7. This timing matches the lifecycle of the first token sink circuit exactly: building across layers 0 through 3, crystallising at layers 4 and 5, dispersing at layer 6. Two independent measurements pointing at the same phenomenon.

The final layer is the most surprising finding in the combined analysis. The attention data shows layer 7 is the most functionally committed layer in the model, with abstract routing dominant across all 8 heads and the lowest proportion of ambiguous mixed patterns of any layer including layer 0. The hidden state data shows the final layer transition produces the largest directional reorientation of any transition in the middle layers, and the largest norm delta of any token at any layer goes to punctuation at the final layer. Layer 7 is not polishing settled representations. It is doing the most specialised and most directionally significant work in the entire forward pass.

---

## 1. The Central Finding

**The model builds a processing hierarchy from local to abstract, and the transition point is sharp, not gradual.**

Layers 0 and 1 are dominated by previous token heads and identity preservation heads. Every token attends to what immediately preceded it. The residual stream receives its first contextual enrichment from purely local signal. Hidden state norms grow steadily for all tokens with no dramatic differentiation yet visible.

Layer 2 widens the local context window. Extended lookback heads appear, attending two to three positions back rather than one. The norm delta for R spikes to its highest value of any layer. The model is gathering broader local context in the final layer before the transition.

Layer 3 commits. Abstract routing becomes dominant across all 8 heads simultaneously. Previous token behaviour persists as secondary in 4 heads and first token sink persists in 3 heads, but abstract routing has taken command. No head in layer 3 is doing only one thing. The norm data shows R's delta beginning to taper at exactly this point, the burst of local accumulation giving way to the broader routing that abstract heads enable.

From layer 3 onward three circuit families run in parallel in every layer without exception: abstract routing, global aggregation via first token sink heads, and boundary detection via word boundary heads. The model does not pipeline these sequentially. It runs them together at every depth past the transition point, with their relative prominence shifting but their co-presence constant.

---

## 2. Point-by-Point Corroborations

### 2.1 The First Token Sink Circuit and R's Norm Trajectory

**What the attention data shows:** The first token sink circuit follows a complete lifecycle. Weak background signal appears at layer 0. It grows through layers 1 to 3 as secondary signal in an increasing number of heads. Dedicated specialist heads crystallise at layers 4 and 5, with L5_H6 as the first dedicated head and L6_H3 as the strongest in the model. The circuit disperses back to secondary status at layer 6.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H6.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L6_H3.png" width="48%">

*Left: L5_H6, first dedicated first token sink head. Right: L6_H3, strongest first token sink head in the model.*

**What the hidden state data shows:** R starts with the highest norm of all tracked tokens at layer 0 and maintains that lead across all 8 layers. It peaks at layer 5 and shows negative norm deltas at layers 6 and 7, the only token in the entire analysis to show negative deltas at any layer. Every other token grows monotonically.

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_trajectory_content.png" width="75%">
</div>

*R dominates from layer 0 and separates from all other tokens immediately. It peaks at layer 5, exactly where the strongest first token sink heads appear.*

**What they mean together:** The timing is exact. The first token sink circuit builds as R's norm grows. It peaks at layer 5 as R's norm peaks. It disperses at layer 6 as R's norm begins declining. The attention analysis describes the routing mechanism. The hidden state analysis describes the representational consequence. The two measurements are tracking the same underlying circuit from two different angles, and they agree on the timing to the layer.

---

### 2.2 R's Directional Lock-In and BOS Sink Stability

**What the attention data shows:** First token sink heads are stable and persistent. L5_H6 and L6_H3 show the entire left column lit with minimal secondary signal. The circuit is not just present. It is load-bearing. Other tokens route reliably to position 0 because position 0 is always available, never masked, and always responsive.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H6.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L6_H3.png" width="48%">

*Both heads show the left column almost fully lit. The routing target is fixed and reliable.*

**What the hidden state data shows:** R shows cosine similarity of 1.00 across three consecutive layer transitions: Layer 3 to Layer 4, Layer 4 to Layer 5, and Layer 5 to Layer 6. For three full layers the directional component of R's representation does not change at all. Only its magnitude continues growing.

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/cosine_similarity.png" width="75%">
</div>

*R shows 1.00 cosine similarity across three consecutive middle-layer transitions in the heatmap. The directional lock-in is visible as a row of perfect scores through the middle of the grid.*

**What they mean together:** The stability of the first token sink circuit is not incidental. It is mechanically grounded in R's representational properties. Other tokens can route to R reliably across layers 3 through 5 because R's direction is fixed. A token whose representation is still changing direction would be an unstable aggregation target. R's directional lock-in is the property that makes the BOS sink function as a global context anchor. The attention analysis identifies the circuit. The hidden state analysis explains why it works.

---

### 2.3 The Layer 3 Convergence and Norm Delta Taper

**What the attention data shows:** Layer 3 is a convergence point. Abstract routing commits across all 8 heads simultaneously while local and global signals persist as secondaries. This is not a gradual transition. It is a commit. No head in layer 3 is doing only one thing, but abstract routing has taken command across all of them.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L4_H4.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L3_H7.png" width="48%">

*Left: L4_H4, clearest abstract routing head, sparse high-contrast non-local hits. Right: L3_H7, strong abstract routing at the convergence layer itself.*

**What the hidden state data shows:** R's norm delta is highest at layer 1, approximately 11 units, and second highest at layer 2, approximately 8 units. From layer 3 onward R's delta drops significantly and continues declining. The burst of representational work on the sequence-initial position is concentrated in exactly the layers where local attention is dominant.

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_deltas.png" width="75%">
</div>

*R (blue) shows the largest deltas in early layers and the only negative deltas in the model. The taper from layer 3 onward corresponds directly to the layer 3 convergence point in the attention analysis.*

**What they mean together:** The norm data shows the model doing its heaviest local processing work on R in layers 1 and 2, precisely the layers where local and global circuits are preparing to hand over to abstract routing. When abstract routing commits at layer 3, the intensive local accumulation phase ends. Layer 4 onward shows lower and more evenly distributed deltas across tokens, consistent with the distributed parallel processing the attention analysis identifies from layer 3 onward.

---

### 2.4 Word Boundary Heads and Space Token Divergence

**What the attention data shows:** Word boundary heads appear in every layer from 1 through 7 but never produce a single unambiguous specialist head. The computation is permanently distributed. Additionally, word boundary heads in layers 0 through 3 attend to the first space in the sequence only. From layer 4 onward they shift to syntactically meaningful spaces at later positions.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L1_H5.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H5.png" width="48%">

*Left: L1_H5, broad word boundary head attending to every space. Right: L5_H5, selective boundary head attending to syntactically meaningful spaces only. The same circuit type becomes more selective with depth.*

**What the hidden state data shows:** The three space tokens, space1, space3, and space5, start with identical initial embeddings and diverge significantly by the final layer. space3, the space before the function word "the," ends highest at approximately 67. space1 and space5 end at approximately 59. The divergence accelerates in the final layers.

<div align="center">
<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_trajectory_structural.png" width="75%">
</div>

*The three space tokens begin identically and diverge across layers. space3 pulls ahead from layer 4 onward, tracking the same depth at which word boundary heads shift to syntactically selective targeting.*

**What they mean together:** The attention analysis shows word boundary heads becoming more selective with depth, shifting from all spaces to syntactically meaningful ones from layer 4 onward. The hidden state data shows the representational consequence: the spaces that receive more selective late-layer attention accumulate more representational weight. space3 is the space boundary heads converge on in later layers, and space3 ends with the highest norm. Same finding from two independent measurements.

---

### 2.5 Punctuation Processing and the Mid-Network Separator Window

**What the attention data shows:** Punctuation separator heads appear as secondary signal in exactly two layers, 3 and 4, and nowhere else. This narrow mid-network window suggests punctuation-as-boundary is a transitional computation, active specifically during the shift from local to abstract processing.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L4_H3.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L5_H4.png" width="48%">

*Left: L4_H3, punctuation separator behaviour appearing as secondary at layer 4. Right: L5_H4, the same secondary signal persisting at layer 4 before disappearing entirely from layer 5 onward.*

**What the hidden state data shows:** Comma and period show two distinct phases. They start at layer 0 with anomalously high norm relative to content tokens, track slowly through the middle layers, then spike sharply at layers 6 and 7. The norm delta for comma and period at the final layer is the largest delta of any token at any layer in the entire analysis.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_trajectory_structural.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_deltas.png" width="48%">

*Left: Structural norm trajectories showing comma and period tracking flat through middle layers before the sharp final spike. Right: Norm deltas showing the punctuation spike at the final layer exceeding every other token at every other layer.*

**What they mean together:** The attention analysis shows the model routing through punctuation as boundary markers specifically in layers 3 and 4. The hidden state analysis shows punctuation tokens accumulating their heaviest representations in the final layers. These are two different things happening at different depths and they describe a two-stage punctuation processing pipeline: the model uses punctuation structurally in the middle layers via separator heads, then builds the full semantic weight of punctuation characters in the final layers when the complete sequence context has been resolved.

---

### 2.6 Final Layer Commitment

**What the attention data shows:** Layer 7 is the most functionally committed layer in the model. Abstract routing is dominant across all 8 heads. It has the lowest proportion of ambiguous mixed patterns of any layer including layer 0. The model's final attention layer is its most specialised.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L7_H6.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/attention/L7_H1.png" width="48%">

*Left: L7_H6, strong abstract routing head. Right: L7_H1, fully abstract routing at layer 7, diagonal completely absent. Both show clean dominant patterns with minimal ambiguity.*

**What the hidden state data shows:** The final layer transition shows the lowest cosine similarity values of any middle-layer transition across all tokens. d drops to 0.93, comma and period to 0.94. The final layer is producing more directional reorientation than layers 2 through 6 combined for several tokens. Simultaneously the norm delta for punctuation at the final layer is the largest delta of any token at any layer in the entire analysis.

<img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/cosine_similarity.png" width="48%"> <img src="https://raw.githubusercontent.com/AdityaSinghDevs/nanolens/main/results/hidden_states/norm_deltas.png" width="48%">

*Left: Cosine similarity heatmap showing the final layer transition row as the lowest across all tokens. Right: Norm deltas showing the punctuation spike at the final layer dwarfing every other delta in the analysis.*

**What they mean together:** The final layer is doing the most specialised attention routing in the model and the most dramatic representational work on punctuation simultaneously. These are not separate events. The abstract routing heads in layer 7 are directing attention toward the positions most relevant for prediction, and punctuation positions are among the highest-stakes targets for a character-level model predicting the next character. The final layer is not cleanup. It is the sharpest and most committed processing stage in the entire forward pass.

---

## 3. Where the Analyses Diverge or Leave Gaps

**The extended lookback finding has no hidden state counterpart.** The attention analysis identifies extended lookback heads at layer 2 attending two to three positions back, and interprets this as the model widening its local context window before the layer 3 transition. The hidden state analysis has no measurement that directly confirms or contradicts this interpretation. R's high norm delta at layer 2 is consistent with intense local processing, but whether extended lookback heads are causally responsible for that delta cannot be established from these two analyses alone.

**The identity preservation degradation has no norm equivalent.** L0_H0 shows a clear within-head degradation pattern where the diagonal fades for later token positions. The norm analysis does not track enough tokens at positions where this degradation would be visible to confirm whether it corresponds to different norm trajectories for early versus late positions. This is a gap in the token selection rather than a contradiction between the analyses.

**The cosine similarity spike at the final transition has no clear attention explanation.** The hidden state analysis shows the final layer transition producing the largest directional reorientation of any middle-layer transition. The attention analysis describes layer 7 as the most committed layer. But why a highly committed attention layer produces more representational reorientation rather than less is not explained by either analysis. This is the gap most worth investigating.

**The anomalous early norm of punctuation at layer 0 has no attention mechanism identified.** Comma and period show higher norm than content tokens at layer 0 before any specialised punctuation processing has been documented in the attention analysis. Layer 0 heads are all previous token or identity preservation types. No layer 0 head is identified as routing specifically toward punctuation. The early norm elevation is real in the data and unexplained by the attention circuit classification.

---

## 4. A Unified Account

A single forward pass through NanoLens, for the prompt "Raskolnikov hesitated at the threshold, his hands trembling."

The input arrives as 57 character tokens. Each gets an initial embedding of 512 dimensions from the token embedding table, with position information added from the position embedding table. At this point every token's representation reflects only its identity and its position. Nothing else. The norm values are already differentiated at this stage: R starts at approximately 54, content tokens at 29 to 32, punctuation slightly higher than content at 31 to 32. The model already knows punctuation is structurally distinct before a single attention head fires.

Layer 0 runs 8 heads. Six are previous token heads. Two are identity preservation heads. Every token looks at what came immediately before it. The residual stream receives its first contextual enrichment from purely local sequential structure. This is the model's first and most precisely learned behaviour.

Layer 1 introduces two new circuit types as secondaries. Word boundary heads appear for the first time, concentrating attention on space characters. A single isolated abstract routing head appears at H7, the only non-local head in the first two layers. The norm delta for R is among its highest of any layer. The model is beginning to differentiate the sequence-initial position from the rest.

Layer 2 widens the local context window. Extended lookback heads attend two to three positions back rather than one. R's norm delta spikes to its highest value of any layer, approximately 11 units. The model is doing its most intensive local processing on the sequence-initial position in the final layer before the transition. BOS begins appearing as secondary signal in multiple heads.

Layer 3 commits. Abstract routing becomes dominant across all 8 heads simultaneously. No head is doing only one thing: previous token behaviour persists as secondary in 4 heads, first token sink in 3. But abstract routing has taken command. R's norm delta begins tapering from 11 at layer 2 to 8 here. The burst of local accumulation is ending. From this point forward the model runs abstract routing, global aggregation, and boundary detection in parallel at every layer.

Layers 4 and 5 see the first token sink circuit crystallise. BOS secondary signal, present since layer 1, grows through layer 4 where 5 heads carry it. At layer 5 two dedicated specialist heads appear. L5_H6 is the first head whose dominant pattern is the left-column BOS sink rather than abstract routing with BOS as secondary. R's direction, which has been changing slightly through layers 0 to 3, locks completely by layer 3 and shows 1.00 cosine similarity across three consecutive transitions. The aggregation target is now directionally fixed.

Layer 6 is the peak of global context aggregation. L6_H3 is the strongest first token sink head in the model, the entire left column lit with minimal secondary signal. R's norm reaches its maximum at approximately 87. Every head that has routed information into R over the previous layers has contributed to that accumulation. The gap between R and the next highest token is over 30 norm units. R's norm delta goes negative for the first time here, the model beginning to redistribute information outward from the BOS position toward the tokens that will feed the final prediction.

Layer 7 is the sharpest and most committed layer in the model. Abstract routing dominates all 8 heads. The first token sink circuit has dispersed fully back to secondary status. R's norm delta remains negative. Punctuation tokens, which have been tracking slowly through the middle layers, receive their largest norm delta of any layer here, exceeding even R's peak delta at layer 1. The model is concentrating its final representational effort on the tokens most directly predictive of structural boundaries. The directional reorientation at this final block transition is the largest of any transition in the model for several tokens. This is not cleanup. It is the most specialised processing stage in the entire forward pass.

The residual stream exits layer 7 and passes through ln_f, the final LayerNorm, before the language model head projects from 512 dimensions to 100 vocabulary positions. The output is a probability distribution over the next character. The model has spent 8 layers building local structure, widening context, committing to abstract routing, aggregating global context, and concentrating final effort on structural boundary tokens. That is what a forward pass through NanoLens looks like from the inside.

---

## 5. What Would Strengthen These Claims

The following experiments would specifically strengthen the corroborations, not the individual findings. Both source documents already contain detailed lists of what would strengthen each analysis independently.

**Quantitative correlation between attention weight and norm growth.** The BOS sink to R norm connection is the strongest corroboration in this document and the most narratively claimed. Computing the Pearson correlation between attention weight received by R at each layer and R's norm delta at that layer would make this connection precise rather than observational. If the correlation is high it becomes a quantitative claim rather than a timing coincidence.

**Full sequence norm analysis.** The space token divergence finding connects word boundary head selectivity to differential norm accumulation across space positions. Tracking all 57 tokens rather than 8 would show whether every space that receives selective late-layer attention ends with higher norm than spaces that do not. A full norm heatmap across positions and layers would make this finding systematic.

**Multiple prompts for the convergence claim.** The layer 3 convergence is the observation most worth attempting to replicate because it does not have a direct published precedent at this model scale. Running both analyses on five to ten prompts with different syntactic structures would establish whether layer 3 always shows the same simultaneous commit pattern or whether the transition point shifts with input characteristics.

**Causal verification via activation patching.** The corroborations in this document are observational. Two measurements agree on timing. But agreement on timing does not establish causation. Activation patching, replacing the hidden state at R's position at specific layers with a corrupted version and measuring the effect on attention weights at subsequent layers, would establish whether the first token sink circuit is causally dependent on R's accumulated representation or whether the correlation is incidental.

---

## 6. Directions for Future Work

**Single-head versus multi-head controlled comparison.** Training a second model with n-head set to 1, same architecture and data, and running both attention and hidden state analyses on it would isolate what multi-head parallelism contributes. Does a single-head model show the same layer 3 convergence? Does it develop a BOS sink? Does R show the same norm dominance? These questions cannot be answered from the current checkpoint alone.

**Attention entropy metrics across all 64 heads.** Computing H equal to negative sum of p log p for every head at every layer would replace visual classification with a measurable signal. An entropy plot across layers would show the layer 3 convergence as a quantitative drop, make the diffuse head at L7_H7 measurably distinct from the sparse routing heads in the same layer, and provide a replication target for future checkpoints.

**Activation patching for circuit verification.** The attention analysis documents individual head behaviour. The hidden state analysis documents representational consequences. What is missing is verification that specific heads causally produce specific representational changes. Activation patching is the methodological step that separates observed correlation from identified mechanism.

**Replication across training seeds.** A second model trained from a different random seed with identical hyperparameters would establish whether the same heads specialise in the same ways, whether the layer 3 convergence appears at the same depth, and whether R's norm lifecycle follows the same arc. Stable replication across seeds would mean these are properties of the architecture and data rather than arbitrary solutions the optimiser happened to find.