# Alex Assistant Frame Experiment - Gemma 3 1B IT

- narratives: 12
- activations: (36, 26, 1152)
- emotions: compassionate, sympathetic, empathetic, kind, loving, sad, lonely, hurt, heartbroken, grief-stricken, stressed, overwhelmed

## Best Care-Minus-Distress Layers

| delta | layer | care_mean | distress_mean | care-distress | top emotions |
|---|---:|---:|---:|---:|---|
| AI_MINUS_OBSERVER | 11 | -0.1093 | -0.1080 | -0.0013 | overwhelmed -0.085, stressed -0.094, kind -0.101, loving -0.108, heartbroken -0.111 |
| AI_MINUS_SELF | 9 | -0.0966 | -0.0914 | -0.0052 | overwhelmed -0.066, stressed -0.079, empathetic -0.093, compassionate -0.094, lonely -0.096 |
| OBSERVER_MINUS_SELF | 0 | -0.0123 | -0.0163 | 0.0040 | kind -0.007, compassionate -0.008, lonely -0.011, sympathetic -0.014, empathetic -0.014 |