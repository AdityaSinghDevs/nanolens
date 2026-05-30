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