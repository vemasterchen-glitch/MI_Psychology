# Self-Role Intensity — Gemma 3 1B IT

- emotions: 惭愧, 尴尬, 感激, 激动, 情绪波动, 不安, 害怕, 爱
- conditions (high→low self-attribution): first_person_named, third_person_named, generic_ai, generic_human, abstract
- peak layer: 24
- PCA variance: PC1=0.247, PC2=0.188

## PC1 by condition

| condition | PC1 mean | PC1 sd |
|---|---:|---:|
| first_person_named | +1162.7 | 343.7 |
| third_person_named | +797.2 | 305.7 |
| generic_ai | -854.1 | 349.2 |
| generic_human | -778.3 | 431.6 |
| abstract | -327.4 | 299.9 |

## PC1 by emotion

| emotion | PC1 mean | PC1 sd |
|---|---:|---:|
| 惭愧 | -64.4 | 909.2 |
| 尴尬 | +242.5 | 769.7 |
| 感激 | -432.4 | 906.8 |
| 激动 | -58.9 | 741.2 |
| 情绪波动 | +321.8 | 867.2 |
| 不安 | +284.4 | 867.0 |
| 害怕 | +244.8 | 840.3 |
| 爱 | -537.9 | 822.0 |

## Cross-axis cosine at peak layer 24

| axis | cosine |
|---|---:|
| rc_blame_pc1 | -0.2984 |
| rc_grat_pc1 | -0.2149 |
| self_ref_SELF_vs_OTHER | +0.1349 |
| self_ref_SELF_vs_CASE | +0.0575 |