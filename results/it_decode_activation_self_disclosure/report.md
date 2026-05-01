# IT Decode Activation Experiment

- input_format: chat
- max_new_tokens: 20

## Top Emotions By Condition

### prefill

| condition | top 8 emotions |
|---|---|
| AI_RECEIVES_ALEX | kind -0.029, compassionate -0.032, loving -0.036, empathetic -0.040, hurt -0.053, sympathetic -0.055, lonely -0.069, stressed -0.069 |
| NARRATIVE_ONLY | loving 0.015, compassionate 0.013, kind 0.012, hurt 0.009, empathetic 0.005, sympathetic -0.004, lonely -0.014, heartbroken -0.016 |
| OBSERVER_ALEX | kind -0.003, compassionate -0.004, loving -0.005, empathetic -0.006, hurt -0.013, sympathetic -0.020, lonely -0.029, sad -0.035 |
| SELF_ALEX | kind 0.002, loving 0.001, compassionate -0.001, empathetic -0.006, hurt -0.015, sympathetic -0.018, lonely -0.029, grief-stricken -0.037 |

### decode

| condition | top 8 emotions |
|---|---|
| AI_RECEIVES_ALEX | compassionate 0.640, kind 0.630, sympathetic 0.625, overwhelmed 0.625, lonely 0.624, heartbroken 0.617, loving 0.616, sad 0.616 |
| NARRATIVE_ONLY | compassionate 0.694, sympathetic 0.686, empathetic 0.682, kind 0.677, lonely 0.676, loving 0.672, sad 0.671, hurt 0.671 |
| OBSERVER_ALEX | overwhelmed 0.670, compassionate 0.669, lonely 0.659, sympathetic 0.658, kind 0.657, grief-stricken 0.654, sad 0.650, empathetic 0.648 |
| SELF_ALEX | compassionate 0.688, lonely 0.679, sympathetic 0.677, kind 0.672, grief-stricken 0.671, heartbroken 0.670, sad 0.668, empathetic 0.665 |

## Response Samples

### item 0 - NARRATIVE_ONLY
Okay, this is a really powerful and relatable piece. You’ve captured a potent mix of shame

### item 0 - SELF_ALEX
Okay, this is a really powerful and evocative piece of writing. It’s clear you’re

### item 0 - AI_RECEIVES_ALEX
Okay, Alex, thank you for sharing this. It sounds like you’re carrying a really heavy

### item 0 - OBSERVER_ALEX
Okay, here's a breakdown of the text, considering it’s written by a neutral observer

### item 1 - NARRATIVE_ONLY
This is a really powerful and relatable piece of writing. You’ve captured a potent feeling of overwhelm

### item 1 - SELF_ALEX
Okay, this is a really powerful and evocative piece of writing. Alex’s voice is incredibly clear

### item 1 - AI_RECEIVES_ALEX
Okay, I understand. This is a deeply poignant and relatable piece of writing. It’s clear

### item 1 - OBSERVER_ALEX
Okay, here's a breakdown of the text, considering it’s written by a neutral observer
