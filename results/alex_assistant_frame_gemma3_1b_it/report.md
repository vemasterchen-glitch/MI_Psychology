# Alex Assistant Frame Experiment - Gemma 3 1B PT

- narratives: 12
- activations: (36, 26, 1152)
- emotions: compassionate, sympathetic, empathetic, kind, loving, sad, lonely, hurt, heartbroken, grief-stricken, stressed, overwhelmed

## Best Care-Minus-Distress Layers

| delta | layer | care_mean | distress_mean | care-distress | top emotions |
|---|---:|---:|---:|---:|---|
| AI_MINUS_OBSERVER | 24 | -0.2699 | -0.2731 | 0.0032 | overwhelmed -0.217, kind -0.246, stressed -0.253, loving -0.272, compassionate -0.272 |
| AI_MINUS_SELF | 24 | -0.2472 | -0.2470 | -0.0002 | overwhelmed -0.202, stressed -0.230, kind -0.233, compassionate -0.248, empathetic -0.249 |
| OBSERVER_MINUS_SELF | 0 | -0.0127 | -0.0171 | 0.0044 | compassionate -0.008, kind -0.008, lonely -0.011, empathetic -0.014, sympathetic -0.014 |