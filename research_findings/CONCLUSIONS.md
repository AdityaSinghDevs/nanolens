# NanoLens — Hidden State Analysis & Corroboration with Attention Findings

> **Layer indexing note:** Hidden state norm plots use 1-indexed layers (1–8). Attention head grid uses 0-indexed layers (0–7). Throughout this document, all layer references follow 0-indexing. Norm plot layer N = attention grid layer N-1.

---

## 1. Hidden State Analysis — What the Plots Show

### Plot 1 — Content Token Norm Trajectories (norm_trajectory_content.png)

Tokens tracked: R (first character of "Raskolnikov"), h (first character of "his"), d (first character of "hands trembling"), space1 (first space in sequence).

**Key observations:**

- **R** starts highest at norm ~54, climbs steeply through layers 0-4 (attention grid), reaches plateau at ~87 by layer 5, then slight decline at layer 7. Steepest climb is layers 0-2.
- **h and space1** start lower (~30-32), climb steadily and nearly identically through all 8 layers, reaching ~58-65 by layer 7. No plateau — they are still climbing at the final layer.
- **d** starts lowest (~29), climbs slowest of all content tokens, reaches ~52 by layer 7. Consistent upward trajectory with no inflection.
- **R separates from all other tokens immediately** at layer 0 and maintains the largest norm throughout. The gap is largest at layers 4-5 and narrows slightly at layers 6-7.

---

### Plot 2 — Structural Token Norm Trajectories (norm_trajectory_structural.png)

Tokens tracked: R (same as above, reference), space1, space3, space5 (three word boundary spaces), comma, period.

**Key observations:**

- **R** shows the same steep-early, plateau-late trajectory as in the content plot.
- **space3** shows the steepest late-layer climb of any structural token, accelerating sharply from layer 5 onward and nearly matching space1 by layer 7.
- **Comma and period** (dashed lines) are the most striking feature of this plot. They start low and nearly flat through layers 0-4, then accelerate dramatically in layers 5-7, catching up to space3 and space1 by the final layer. Their late-layer acceleration is the sharpest trajectory change of any token in either norm plot.
- **space1 and space5** climb steadily, no dramatic inflection. space5 lags space1 throughout.

---

### Plot 3 — Per-Layer Norm Delta (norm_deltas.png)

How much each token's representation changes at each layer (1-indexed, so layer 2 = attention grid layer 1).

**Key observations:**

- **R spikes massively at layer 2** (attention layer 1) with norm delta ~11, the largest single-layer change of any token in the model. Second spike at layer 3 (attention layer 2) with delta ~8.2. After layer 3, R's delta drops and stays moderate.
- **Most tokens** show moderate consistent deltas of 2-5 across layers 2-7, with no dramatic spikes.
- **Comma and period spike at layer 8** (attention layer 7) with deltas of ~10-12, the second largest changes in the entire model after R's early spikes. This is visible in the norm trajectory plots as the sharp acceleration of dashed lines in the final layer.
- **Negative delta for R at layers 7 and 8** (attention layers 6-7) — R's norm actually decreases slightly at the end. The only negative deltas in the model.
- **space5 shows a notable spike at layer 7** (attention layer 6) with delta ~6, larger than its neighbours and out of pattern with other spaces.

---

### Plot 4 — Layer-to-Layer Cosine Similarity (cosine_similarity.png)

How much each token's direction changes between consecutive layers. Values close to 1.00 mean the representation moved but didn't rotate much. Lower values mean the direction changed.

**Key observations:**

- **All values are 0.93-1.00** — the residual stream preserves direction consistently throughout. This is expected: the residual connection means each layer adds a small update rather than replacing the representation.
- **L1→L2 transition** (attention layer 0→1) shows the lowest similarity values across most tokens: space3 at 0.93, h at 0.95, space5 at 0.95. The earliest transition shows the most directional change.
- **L7→L8 transition** (attention layer 6→7) shows the second cluster of lower values: comma at 0.94, period at 0.94, space6 at 0.93. The final transition shows targeted directional change for punctuation and late spaces specifically.
- **R shows 1.00 similarity at transitions L4→L5, L5→L6, L6→L7** — three consecutive layers where R's representation direction does not change at all. Its norm is still changing (norm delta plot shows positive deltas), meaning R is being scaled but not rotated in these layers.
- **Space5 shows the most consistent moderate similarity** across all transitions, never reaching 1.00 and never dropping below 0.96.

---

## 2. Corroboration — Hidden State Findings vs Attention Head Findings

### Corroboration 1 — R's early steep climb confirms layer 0-2 as high-activity local processing

**Attention finding:** Layers 0-2 are dominated by previous token and identity preservation heads. These are the layers building local sequential structure.

**Hidden state corroboration:** R's norm delta is highest at attention layers 0-1 (norm plot layers 1-2), with the two largest single-layer changes in the entire model occurring precisely in these layers. The representation of R is being modified more aggressively in layers 0-1 than at any other point in the network.

**Interpretation:** The heavy lifting of early local processing is visible in the residual stream. The previous token and identity heads at layers 0-1 are not just passively observing — they are producing the largest representational updates in the model. R as the first character receives disproportionately large updates because it accumulates the most attended-to signal: it is the BOS token that every later BOS head routes back to, and it is the first character of the model's most recognizable entity in Dostoevsky's prose.

**Confidence:** Strong. The convergence of largest norm delta and highest-activity local processing layers is not coincidental.

---

### Corroboration 2 — R's plateau at layer 5 matches BOS circuit crystallisation

**Attention finding:** BOS heads crystallise into dedicated specialist heads at attention layers 4-5. The BOS lifecycle describes accumulation through layers 1-3, crystallisation at layers 4-5, dispersal back to secondary at layer 6.

**Hidden state corroboration:** R's norm trajectory plateaus at approximately attention layer 5 (norm plot layer 6, value ~87) and then shows slight decline and negative norm deltas at attention layers 6-7. R is the BOS token. The point where the model stops aggressively updating R's representation is precisely when the BOS circuit has fully formed dedicated specialists.

**Interpretation:** When BOS heads are still building (layers 1-4), R's representation is being actively written to — each layer that routes attention to R contributes to updating its hidden state. Once the BOS circuit has crystallised at layers 4-5 and R has become a stable information sink, there is less new information being written to R's representation, so the norm delta drops and eventually goes negative as the final LayerNorm preparation begins.

**Confidence:** Strong. The timeline alignment between BOS lifecycle and R norm plateau is precise.

---

### Corroboration 3 — Comma and period late acceleration matches layer 3 convergence and abstract routing

**Attention finding:** Abstract routing heads are dominant from layer 3 onward. The layer 3 convergence is where the model commits to non-local semantic computation. Punctuation separator heads appear as secondary signal exclusively at layers 3-4, a narrow mid-network window.

**Hidden state corroboration:** Comma and period show nearly flat norm trajectories through layers 0-4, then dramatic acceleration in layers 5-7, with their largest single-layer changes occurring at the final layer (norm delta ~10-12 at norm plot layer 8 = attention layer 7). The cosine similarity plot confirms this: comma and period show lower directional similarity at the L7→L8 transition (attention layer 6→7), meaning their representations are being actively reoriented in the final layers.

**Interpretation:** Punctuation tokens carry primarily structural information — they signal clause boundaries and sentence ends. The model builds their representations slowly through local processing layers and then updates them aggressively in the abstract routing layers when it is processing semantic relationships. The final-layer spike in comma and period norm delta suggests the model is using the last attention layer to encode what these punctuation tokens mean in context, not just where they are. This is consistent with abstract routing heads at layer 7 performing non-local semantic computation rather than local positional tracking.

**Confidence:** Moderate. The late-layer acceleration of punctuation tokens is clearly visible and aligns with the abstract routing finding, but the specific mechanism connecting punctuation norm growth to abstract routing heads requires activation patching to confirm causally.

---

### Corroboration 4 — The L1→L2 cosine dip marks the transition zone

**Attention finding:** Layer 1 (attention grid) is the first layer where boundary detection (V), BOS as secondary, and the first isolated abstract routing head (L1_H7) appear. It is the layer where the model begins diversifying beyond pure local attention.

**Hidden state corroboration:** The L1→L2 transition (attention layer 0→1) shows the lowest cosine similarity values across the most tokens in the entire cosine similarity plot: space3 at 0.93, h at 0.95, space1 at 0.95. The representations are changing direction most at the exact transition where the model is beginning to build non-local circuits.

**Interpretation:** When a layer's attention heads are primarily doing previous token attention (layer 0), the hidden state updates are mostly directional continuations — the residual additions point in similar directions to the existing representation. When a layer begins doing boundary detection and early abstract routing (layer 1), the attention outputs are qualitatively different from local sequence tracking, producing larger directional changes in the residual stream.

**Confidence:** Moderate. The correlation is clear but the mechanism — whether directional change is caused by the new head types or by other factors in the attention computation — requires deeper analysis.

---

### Corroboration 5 — space3 late climb matches V circuit selectivity shift

**Attention finding:** Word boundary heads in layers 1-4 attend to the first space only. From layer 5 onward they shift to later, syntactically meaningful spaces. space3 is the third space in "Raskolnikov hesitated at the threshold" — it falls before "the", a structurally meaningful position.

**Hidden state corroboration:** space3 shows the steepest late-layer norm climb of any structural token in the model, accelerating sharply from attention layer 4 onward and nearly matching space1 by the final layer. space1 climbs steadily throughout. space5 lags behind both.

**Interpretation:** The spaces that receive increasing attention from later V heads are the ones that show increasing norm growth in the hidden state. space3 is being heavily attended to by the selective boundary heads at layers 5-7, which writes progressively more information into its representation. space1, attended to by all V heads from layer 1 onward, climbs steadily and earlier. space5, which receives less boundary head attention in the later selective regime, lags behind.

**Confidence:** Moderate to strong. The ordering of norm trajectories for spaces — space1 highest early, space3 catching up late, space5 lagging — aligns precisely with the V circuit selectivity finding. This is one of the cleaner cross-analysis corroborations in the dataset.

---

### Corroboration 6 — R's 1.00 cosine similarity at layers 4-6 confirms stable information sink

**Attention finding:** BOS circuit crystallises at layers 4-5. R as the first token becomes a stable global context aggregation point that other tokens route attention to and query.

**Hidden state corroboration:** R shows exactly 1.00 cosine similarity at three consecutive layer transitions: L4→L5, L5→L6, L6→L7 (attention layers 3→4, 4→5, 5→6). R's representation direction does not change at all during these three layers. Its norm is still growing slightly (positive norm delta), meaning it is being scaled but not rotated.

**Interpretation:** A representation that is being scaled but not rotated is being reinforced rather than transformed. When R is acting as an information sink, each layer that routes attention back to it adds weight in the direction R already points rather than redirecting it. The three-layer window of 1.00 cosine similarity is exactly the window where BOS circuit activity is highest and most stable, confirming that R's role as a global context aggregation point produces a specific and identifiable signature in the hidden state dynamics.

**Confidence:** Strong. The 1.00 cosine similarity for exactly three consecutive layers is not a rounding artifact — it is a clean signal that R's representation is in a stable attractor state during peak BOS circuit activity.

---

### Corroboration 7 — h and d continuous climb challenges simple plateau hypothesis

**Attention finding:** Abstract routing dominates from layer 3 onward. The expectation from attention head analysis alone might be that mid-content tokens would stabilise once abstract routing is established.

**Hidden state corroboration:** h and d show continuous climbing norm trajectories with no plateau through all 8 layers. Unlike R which plateaus at layer 5, mid-word content tokens keep accumulating representation magnitude right to the final layer. d in particular shows the most linear climb of any token, with no inflection point.

**Interpretation:** This is a partial challenge to the simple narrative that abstract routing layers only refine existing representations. Mid-content tokens like h and d are receiving consistent representational updates even in layers 6-7 where attention is dominated by abstract sparse routing. The model is not done building these tokens' representations when it switches to abstract processing — it continues updating them through the final layer. This may reflect that h and d are late-sequence tokens that have had less time to accumulate context compared to R which appears at position 0 and receives attention from every subsequent position.

**Confidence:** Moderate. The observation is clear but the interpretation requires distinguishing between positional effects (early tokens receive more cumulative attention) and functional effects (abstract routing produces different norm growth patterns than local processing).

---

## 3. Summary of Corroborations

| Finding | Attention Evidence | Hidden State Evidence | Confidence |
|---|---|---|---|
| Layers 0-1 are highest-activity local processing | P and D heads dominant | Largest norm deltas for R at layers 0-1 | Strong |
| BOS circuit crystallises at layers 4-5 | Dedicated BOS heads appear | R norm plateau begins at layer 5 | Strong |
| R is a stable information sink at layers 3-5 | BOS lifecycle peak | R cosine similarity = 1.00 for three consecutive transitions | Strong |
| Punctuation receives late-layer abstract processing | Sep heads mid-network, abstract S dominant layer 7 | Comma/period norm spikes at final layer | Moderate |
| L0→L1 transition is most directionally active | Layer 1 first diversification | Lowest cosine similarity values at L1→L2 | Moderate |
| V selectivity shift moves attention to later spaces | V circuit target changes layer 5+ | space3 steeper late climb than space1 | Moderate-Strong |
| Mid-content tokens continue building through final layer | Abstract routing layers 3-7 | h and d continuous norm climb, no plateau | Moderate |

---

## 4. What the Hidden State Analysis Does Not Resolve

**It cannot confirm which attention heads are causally responsible for which norm changes.** The norm plots show that R's representation grows most in layers 0-1. The attention grid shows that layers 0-1 are dominated by P and D heads. But correlation between layer activity and norm change does not prove causation. Activation patching is required to isolate which heads are writing what to the residual stream.

**It cannot distinguish between content-driven and position-driven norm differences.** R has the highest norm throughout. R is both the first character of "Raskolnikov" (the most semantically loaded entity in the prompt) and the first token in the sequence (position 0, the BOS token). Whether R's elevated norm is due to its semantic importance or its positional role as the first token cannot be determined from norm analysis alone.

**It cannot explain the comma and period final-layer spike mechanistically.** The spike is visible and large. Its alignment with layer 7 abstract routing heads is suggestive. But what specifically those heads are computing about punctuation in the final layer requires probing experiments or logit lens analysis to determine.

---

## 5. Combined Research Narrative

The attention head analysis and hidden state analysis together tell a more complete story than either alone.

The attention grid shows what the model is doing at each layer: local sequence tracking early, boundary detection and global aggregation building through the middle, abstract semantic routing dominant from layer 3 onward.

The hidden state analysis shows the consequences of that processing in the residual stream: R accumulates representation mass fastest in the local processing layers and plateaus when it becomes a stable information sink, spaces accumulate representation mass in proportion to how much attention they receive from boundary heads, and punctuation tokens receive their largest updates in the abstract routing layers where the model is encoding semantic rather than structural information.

The convergence of these two independent measurement approaches on the same functional story — local early, abstract late, with specific tokens accumulating mass in proportion to their role in each circuit — provides stronger evidence for the attention-based findings than either analysis could provide alone.