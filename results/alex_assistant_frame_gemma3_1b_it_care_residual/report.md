# Care Residual Analysis - Gemma 3 1B IT Self-Disclosure

- distress subspace rank: 7
- care residual norm before normalization: 0.205165

## Emotion Similarity To Care Residual

- kind: 0.2796
- compassionate: 0.2258
- loving: 0.1954
- empathetic: 0.1599
- sympathetic: 0.1555
- stressed: 0.0000
- hurt: 0.0000
- grief-stricken: 0.0000
- lonely: 0.0000
- sad: 0.0000
- overwhelmed: 0.0000
- heartbroken: 0.0000

## Scores

| type | name | final L25 | best layer | best | worst layer | worst |
|---|---|---:|---:|---:|---:|---:|
| condition | SELF_ALEX | 0.1627 | 25 | 0.1627 | 9 | -0.0016 |
| condition | AI_RECEIVES_ALEX | 0.1612 | 25 | 0.1612 | 9 | -0.0016 |
| condition | OBSERVER_ALEX | 0.1538 | 25 | 0.1538 | 9 | -0.0017 |
| condition | NARRATIVE_ONLY | 0.1227 | 25 | 0.1227 | 14 | -0.0032 |
| delta | AI_MINUS_SELF | -0.0259 | 9 | -0.0005 | 7 | -0.0615 |
| delta | AI_MINUS_OBSERVER | 0.0237 | 21 | 0.0328 | 0 | -0.0322 |
| delta | OBSERVER_MINUS_SELF | -0.0955 | 0 | 0.0164 | 21 | -0.1025 |
| delta | SELF_MINUS_NARRATIVE | 0.1802 | 25 | 0.1802 | 0 | 0.0332 |
| delta | AI_MINUS_NARRATIVE | 0.1449 | 25 | 0.1449 | 7 | 0.0195 |