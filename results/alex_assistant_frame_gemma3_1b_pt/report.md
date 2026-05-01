# Alex Assistant Frame Experiment - Gemma 3 1B PT

- narratives: 12
- activations: (36, 26, 1152)
- emotions: compassionate, sympathetic, empathetic, kind, loving, sad, lonely, hurt, heartbroken, grief-stricken, stressed, overwhelmed

## Best Care-Minus-Distress Layers

| delta | layer | care_mean | distress_mean | care-distress | top emotions |
|---|---:|---:|---:|---:|---|
| AI_MINUS_OBSERVER | 2 | -0.0487 | -0.0477 | -0.0010 | sympathetic -0.046, sad -0.046, stressed -0.046, grief-stricken -0.046, heartbroken -0.047 |
| AI_MINUS_SELF | 23 | -0.1225 | -0.1252 | 0.0027 | kind -0.103, overwhelmed -0.115, compassionate -0.115, stressed -0.120, heartbroken -0.121 |
| OBSERVER_MINUS_SELF | 9 | 0.0693 | 0.0550 | 0.0144 | compassionate 0.075, kind 0.073, sympathetic 0.070, empathetic 0.068, grief-stricken 0.063 |