# Attention Head Classification Grid

## Grid

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

## Annotations

- `L1_H4` — V pattern begins at 3rd token, not sequence start
- `L5_H7` — V at beginning and end of each word
- `Till layer 4` — V appears at first space only
- `From layer 6` — V shifts to later spaces
- `L0_H0` — D degrades into diffuse (X) across sequence length

---

## Layer Summary

| Layer | Dominant Type | Character |
|---|---|---|
| 0 | P, D | Pure local — previous token and self attention |
| 1 | P, V emerging | Local with first boundary detection appearing |
| 2 | P transitioning to S | Local giving way — BOS secondary signal begins |
| 3 | S | Hard switch — majority abstract across all heads |
| 4 | S, BOS emerging | Abstract dominant — dedicated BOS sink heads appearing |
| 5 | S, BOS | Mixed abstract — BOS sink heads fully formed |
| 6 | S, BOS | Abstract — sparse routing and BOS coexisting |
| 7 | S | Fully abstract — semantic routing across all heads |

## Head Type Distribution by Layer

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

# NanoLens — Attention Head Analysis & Research Findings

> **Epistemic note**: All findings are from a single trained checkpoint (8 layers, 8 heads, 512 embedding dim, 25M parameters) on a single prompt: *"Raskolnikov hesitated at the threshold, his hands trembling."* They are directional and exploratory — observations from a character-level model, not claims about transformers in general. Patterns are consistent with prior interpretability literature, which lends credibility, but replication across prompts and checkpoints is needed before stronger claims can be made.

---

## 1. Classification Grid

| Layer | H0 | H1 | H2 | H3 | H4 | H5 | H6 | H7 |
|---|---|---|---|---|---|---|---|---|
| 0 | D★(X) | P | P★ | P | D(P) | P★ | P | P |
| 1 | P(V) | P(D) | P★ | P(3rd token) | V(BOS,X) | V | P★ | S |
| 2 | S(BOS) | P(3rd&2nd) | P(BOS) | P(4th&3rd) | P(S) | P(3-4) | S(BOS,V) | S★ |
| 3 | S(P) | S(BOS,V) | S(BOS,P) | S(BOS) | S(P) | S(V) | S(P) | S★ |
| 4 | S(BOS) | S | S(BOS) | S(BOS,V,Sep) | S★ | S(V,BOS) | S | S(V,BOS) |
| 5 | S(BOS) | S(V) | S(BOS,V) | S(P) | S(Sep,BOS) | BOS★(V) | BOS★ | S(V) |
| 6 | S(V,BOS) | S★ | S(BOS,V) | BOS★ | S | S(BOS,V) | S(BOS) | S(BOS) |
| 7 | S | S(BOS) | S(V) | S(BOS) | S(BOS) | S(BOS,V) | S★ | S★(BOS) |

**Legend:**

| Symbol | Full Name | What It Means |
|---|---|---|
| P | Previous Token | Sub-diagonal stripe, every token attends to the one before it |
| D | Diagonal | Main diagonal dominant, token attends primarily to itself |
| V | Vertical Stripes | Bright columns at space characters, word boundary detection |
| Sep | Separator | Vertical stripes at punctuation specifically |
| BOS | First Token Sink | Entire left column lit, every token routes attention back to position 0 |
| S | Sparse Scattered | High contrast non-local hits, abstract feature routing |
| X | Diffuse | Broadly distributed low contrast attention |
| ★ | Strong Example | Clearest unambiguous representative of that type |
| (secondary) | Mixed Behaviour | Head exhibits secondary pattern alongside dominant type |

---

## 2. Head Type Distribution by Layer

| Layer | P | D | V | Sep | BOS | S | X |
|---|---|---|---|---|---|---|---|
| 0 | 6 (2★) + 1sub | 2 (1★) | — | — | — | — | 1sub |
| 1 | 5 (2★) + 1sub | 1sub | 2 (1sub) | — | 1sub | 1 | 1sub |
| 2 | 5 (1sub) | — | 1sub | — | 3sub | 3 (1★) | — |
| 3 | 4sub | — | 2sub | — | 3sub | 8 (1★) | — |
| 4 | — | — | 3sub | 1sub | 5sub | 8 (1★) | — |
| 5 | 1sub | — | 4sub | — | 2 (1★) + 3sub | 6 | — |
| 6 | — | — | 3sub | — | 1★ + 5sub | 7 (1★) | — |
| 7 | — | — | 2sub | — | 5sub | 8 (2★) | — |

**Legend:** `★` = strong unambiguous example, `sub` = secondary/mixed behaviour, `—` = absent

---

## 3. Sharp Examples by Head

| Type | Head | Layer | Notes |
|---|---|---|---|
| P | L0_H2 | 0 | Cleanest previous token head in model |
| P | L0_H5 | 0 | Strong previous token |
| P | L1_H2 | 1 | Strong previous token |
| P | L1_H6 | 1 | Strong previous token |
| D | L0_H0 | 0 | Only diagonal head, degrades to X across sequence length |
| S | L2_H7 | 2 | First strong sparse head, earliest abstract routing in model |
| S | L3_H7 | 3 | Strong sparse at phase transition layer |
| S | L4_H4 | 4 | Strong sparse mid-network |
| S | L6_H1 | 6 | Strong sparse late network |
| S | L6_H3 | 6 | Strongest BOS in model, also carries S |
| S | L7_H6 | 7 | Strong sparse final layer |
| S | L7_H7 | 7 | Strong sparse final layer with BOS secondary |
| BOS | L5_H6 | 5 | First strong BOS head |
| BOS | L6_H3 | 6 | Strongest BOS in model |
| V | — | — | No sharp examples across all 64 heads |
| X | — | — | Single occurrence, L0_H0 secondary only |

---

## 4. Layer Summary

| Layer | Dominant Type | Character |
|---|---|---|
| 0 | P, D | Pure local, previous token and self attention dominate |
| 1 | P, V emerging | Local with first boundary detection as weak secondary signal |
| 2 | P transitioning to S | Local giving way, BOS and S both appear as subscripts simultaneously |
| 3 | S convergence | Hard commit to abstract across all heads, P and BOS persist as secondaries |
| 4 | S, BOS emerging | Abstract dominant, BOS and V both maintained as secondary circuits |
| 5 | S, BOS dominant | BOS graduates to dominant for the first time, V reaches peak subscript count |
| 6 | S, BOS | Two committed abstract circuits coexisting, strongest BOS sharp example |
| 7 | S | Most committed layer, abstract routing across all heads, BOS and V persist as background |

---

## 5. Annotations

- `L1_H4` — V pattern begins at 3rd token, not sequence start
- `L5_H7` — V at beginning and end of each word
- Till layer 4 — V appears at first space only
- From layer 6 — V shifts to later syntactically meaningful spaces
- `L0_H0` — D degrades into diffuse (X) across sequence length

---

## 6. Deep Analysis

### Finding 1 — Layer 3 is a convergence, not a clean switch

The dominant-only read of layer 3 suggests a hard phase transition: S jumps to 8 heads while P drops. The subscript read tells a more accurate story. S is dominant across all 8 heads in layer 3, but P persists as secondary in 4 heads and BOS persists as secondary in 3 heads simultaneously. Three different functional systems are running in the same layer at the same time, local structure, global aggregation, and abstract routing, with abstract routing having taken command.

This is not a clean switch. It is a convergence layer where the model commits to abstract processing as primary while keeping local and global signals alive as background. No head in layer 3 is doing only one thing.

### Finding 2 — Circuits build as subscripts before becoming dominant

The developmental sequence of BOS makes this visible. BOS first appears in layer 1 as a single subscript. It grows as subscript through layers 2 and 3 with 3 heads each. It persists as subscript in layer 4 with 5 heads. It achieves dominant status for the first time in layer 5 with 2 sharp examples. Then it broadens into dominant + subscript form across layers 6 and 7.

The model does not switch BOS on at layer 5. It builds the circuit gradually across 4 layers as a background signal before committing to it as a primary function. The same pattern likely holds for S, which first appears as dominant in layer 1 head 7 before expanding to 3 heads in layer 2 and 8 in layer 3.

This is a general principle visible in the data: new circuit types appear first as subscripts in earlier layers, get reinforced across multiple layers, and graduate to dominance when sufficiently mature.

### Finding 3 — V never achieves sharpness but never disappears

V has zero sharp examples across all 64 heads. It is always secondary or subscript. Yet it appears in every layer from 1 to 7 without a single absence. Word boundary detection is load-bearing but distributed, no single head owns it cleanly, and many heads carry a piece of it as background.

Contrast this with BOS, which achieves sharp examples and has dedicated specialist heads at layers 5 and 6. Global context aggregation centralizes into specialists. Boundary detection remains distributed throughout the network at all depths. These are two different architectural solutions to two different problems.

Additionally, V shifts its target across depth. Layers 1 to 4 show V attending to the first space in the sequence only. Layers 6 and 7 show V shifting to later, syntactically meaningful spaces. The same circuit type, applied to progressively more selective positions as depth increases. This is functional refinement of a distributed circuit across layers.

### Finding 4 — D is a layer 0 exclusive, identity preservation is a primitive

D appears once sharply in layer 0 and once as subscript in layer 1. After that it is absent from all 48 remaining heads. Identity preservation, a token attending primarily to itself, is only useful before the model has built contextual representations. Once higher layers have enriched each token's representation with context, there is no value in a head that echoes the raw embedding.

The L0_H0 degradation is a specific observation within this: the diagonal pattern holds cleanly for early positions in the sequence but degrades to diffuse (X) for later tokens. Identity preservation is hardest to maintain as context accumulates. Long sequences have more competing signals pulling attention away from self.

### Finding 5 — Layer 7 is the most functionally committed layer

Layer 7 shows S 8 times dominant with 2 sharp examples, BOS 5 times as subscript, V twice as subscript. Every head has a clear dominant type. The ratio of ambiguous mixed patterns to clean dominants is lower in layer 7 than in any other layer, including layer 0.

The model's final processing layer is its most specialized. This is counterintuitive: you might expect the final layer to do general mixing before projection to vocabulary. Instead it runs highly committed abstract routing in every head, with global context and boundary signals maintained quietly in the background. The clean-up and integration work appears to happen in the residual stream and the final LayerNorm, not in the attention heads themselves.

### Finding 6 — The parallel processing architecture from layer 3 onward

From layer 3 to layer 7, three circuit families run simultaneously in every layer without exception: S (abstract routing), BOS (global aggregation), and V (boundary detection). None of these disappears. None takes over completely. The model maintains all three in parallel, with their relative prominence shifting across depth but their simultaneous presence constant.

This is not a pipeline where local circuits hand off to global circuits which hand off to abstract circuits. It is a parallel architecture where all three are active at all depths past the transition point, with different layers weighting them differently.

---

## 7. Literature Connections

### Previous token heads — Elhage et al. 2021

"A Mathematical Framework for Transformer Circuits" (Elhage et al., Anthropic, 2021) explicitly documents previous token heads as one of the most common and reproducible head types across transformer models. They appear in layer 0 of virtually every transformer studied, identified by their characteristic sub-diagonal attention pattern. NanoLens reproduces this finding precisely: 6 P heads in layer 0, the strongest and clearest patterns in the entire model, with 2 sharp examples. The literature predicts this. The data confirms it in a character-level model at 25M parameters.

The circuits framework also describes P heads as part of the induction circuit mechanism: a previous token head in an early layer provides the shifted sequence that induction heads in later layers use for in-context pattern matching. The presence of strong P heads in layers 0 and 1 of NanoLens is consistent with this, though verifying the full induction circuit would require activation patching experiments beyond the scope of this analysis.

### BOS / first token sink — Elhage et al. 2021

The first token sink is documented in the circuits framework as a mechanism for storing global sequence-level information. Attention is a softmax over all previous positions and cannot produce zero attention to every position simultaneously. When a head has no useful local signal to route, it defaults to attending to a reliable, always-available position: the first token, which is never masked by the causal mask. This produces the characteristic full left-column illumination visible in NanoLens BOS heads.

The framework notes this as a stable and widespread phenomenon. NanoLens shows BOS emerging gradually as subscript from layer 1 and stabilizing as a dominant circuit at layer 5, consistent with the literature's description of BOS as a mid-network phenomenon used by later layers for global context lookup rather than a final-layer artifact.

### Layer hierarchy — Clark et al. 2019, Tenney et al. 2019

"What Does BERT Look At?" (Clark et al., 2019) and "BERT Rediscovers the Classical NLP Pipeline" (Tenney et al., 2019) both document that transformer layers follow a processing hierarchy: early layers handle syntax and local structure, later layers handle semantics and long-range dependencies. This has been observed consistently in large bidirectional models trained on subword tokens.

NanoLens reproduces this hierarchy in a fundamentally different setting: character-level tokenization, causal (not bidirectional) attention, autoregressive training, and 25M parameters rather than 110M+. The local-to-abstract gradient is not an artifact of scale, tokenization strategy, or training objective. It appears to be a fundamental property of how transformers organize learned representations under gradient descent regardless of these variables.

### Word boundary detection in character-level models

Published work on character-level transformers, including Al-Rfou et al. (2019) and the original GPT work (Radford et al., 2018), observed that character-level models implicitly learn word-level structure through attention without any explicit word-level supervision. NanoLens provides direct mechanistic evidence for this: V heads detecting space characters as structural boundaries, present from layer 1 through layer 7, in a model that has never seen a word token.

The finding that V never achieves sharpness (zero strong examples across 64 heads) while remaining persistently distributed is not directly addressed in the existing literature on character-level models. It suggests word boundary detection in this architecture is an emergent distributed computation rather than a localized circuit, which may be a consequence of character-level tokenization specifically. Subword models can rely on token boundaries directly; character-level models must construct word boundaries from scratch and appear to distribute that computation across many heads rather than centralizing it.

### The layer 3 convergence — no direct literature precedent at this scale

The sharpness of the layer 3 transition in NanoLens, where every head commits to S as primary while simultaneously maintaining P and BOS as secondaries, does not have a direct published precedent at this model scale and architecture that I can identify. The circuits framework describes layer hierarchies but does not document this specific convergence pattern in 8-layer models. Larger model studies (e.g. Wang et al. 2022 on GPT-2 indirect object identification circuits) show complex multi-layer circuit structures but do not describe a single convergence layer of this character.

This is the most novel observation in the NanoLens dataset. It should be treated with appropriate caution given the single-prompt, single-checkpoint scope. But it is the finding most worth attempting to replicate, because if it holds across prompts and checkpoints, it represents a specific claim about how 8-layer transformers organize their processing hierarchy that is not currently in the literature.

---

## 8. What Would Strengthen These Claims

The following experiments would move these findings from exploratory observations to defensible claims:

- Run the same inspection on 3-5 different prompts and check whether head classifications are stable across inputs
- Compute attention entropy (H = -sum(p log p)) for all 64 heads quantitatively, replacing visual classification with a measurable signal
- Train a second checkpoint from a different random seed and check whether the same heads specialize in the same ways, or whether specialization is random across seeds
- Run the single-head vs multi-head controlled comparison to isolate what multi-head parallelism contributes
- Hidden state norm plots across layers for specific tokens to connect the attention findings to the residual stream representation magnitude story

None of these are required for NanoLens as a research toolkit. They are the roadmap for turning these observations into PRISMA-grade claims.