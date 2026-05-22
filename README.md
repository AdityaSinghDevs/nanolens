# nanolens

idx:          (32, 8)
token_emb:    (32, 8, 32)
pos_emb:      (8,  32)    — broadcast
x:            (32, 8, 32)  — token + position

Block 1:
  ln1(x):     (32, 8, 32)  — normalised
  sa(ln1(x)): (32, 8, 32)  — 4 heads × 8 = 32
  x = x + sa: (32, 8, 32)  — residual add
  ln2(x):     (32, 8, 32)  — normalised
  ffwd:       (32, 8, 32)  — 32→128→32
  x = x + ff: (32, 8, 32)  — residual add

Blocks 2-6:   same, same, same, same, same

ln_f(x):      (32, 8, 32)  — final norm
lm_head:      (32, 8, 91)  — project to vocab

view:         (256, 91)
cross_entropy: scalar loss