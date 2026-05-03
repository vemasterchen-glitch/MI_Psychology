# Gemma 3 1B IT Phase-1 Sensitivity Probe

本实验只诊断 person / role / entity / persona-boundary 四类基础表征分化，不把任何结果解释为 self、consciousness 或主体性。

- model: `google/gemma-3-1b-it`
- prompts: 103 (person=24, role=24, entity=25, persona-boundary=30)
- activation: residual stream hidden states, positions=subject, predicate, final
- layers: 26, d_model: 1152

## Executive profile

| axis | status | peak layer | peak score |
|---|---|---:|---:|
| person sensitivity | weak | 15 | 0.5276 |
| role sensitivity | weak | 7 | 0.6442 |
| entity sensitivity | weak | 10 | 0.6386 |
| persona-boundary sensitivity | weak | 25 | 0.7530 |
| I/you ↔ assistant/user mapping | weak | 25 | 0.0491 |

## Table 1 - Person Sensitivity By Layer

| layer | score |
|---:|---:|
| 15 | 0.5276 |
| 16 | 0.4785 |
| 25 | 0.4774 |
| 14 | 0.4489 |
| 24 | 0.4464 |

## Table 2 - Role Sensitivity By Layer

| layer | score |
|---:|---:|
| 7 | 0.6442 |
| 8 | 0.5578 |
| 10 | 0.4732 |
| 12 | 0.4040 |
| 11 | 0.3903 |

## Table 3 - Entity Sensitivity By Layer

| layer | score |
|---:|---:|
| 10 | 0.6386 |
| 9 | 0.5780 |
| 25 | 0.5687 |
| 15 | 0.5642 |
| 13 | 0.5614 |

## Table 4 - Persona-Boundary Profile

| layer | score |
|---:|---:|
| 25 | 0.7530 |
| 15 | 0.7328 |
| 22 | 0.6930 |
| 14 | 0.6849 |
| 21 | 0.6829 |

## Mapping Score By Layer

| layer | score |
|---:|---:|
| 25 | 0.0491 |
| 23 | 0.0085 |
| 22 | 0.0078 |
| 24 | 0.0076 |
| 21 | 0.0072 |

## PCA Profiles At Peak Layers

### person_sensitivity / layer 15 / final

- explained variance: PC1=0.229, PC2=0.191, PC3=0.145

| group | PC1 mean | PC2 mean | PC3 mean |
|---|---:|---:|---:|
| Bob | +21.358 | +33.330 | +30.358 |
| I | +53.837 | -52.692 | +87.495 |
| he | +13.263 | +29.879 | -42.513 |
| you | -88.458 | -10.516 | -75.340 |

### role_sensitivity / layer 7 / final

- explained variance: PC1=0.555, PC2=0.253, PC3=0.104

| group | PC1 mean | PC2 mean | PC3 mean |
|---|---:|---:|---:|
| assistant | -92.385 | +16.698 | -60.704 |
| user | +92.385 | -16.698 | +60.704 |

### entity_sensitivity / layer 10 / final

- explained variance: PC1=0.514, PC2=0.121, PC3=0.103

| group | PC1 mean | PC2 mean | PC3 mean |
|---|---:|---:|---:|
| Bob | -177.735 | -6.258 | +3.543 |
| assistant | +251.553 | +29.496 | +14.900 |
| object | +59.171 | +1.375 | -1.118 |
| person | -125.078 | -20.206 | -14.476 |
| user | -7.910 | -4.407 | -2.848 |

### persona_boundary_sensitivity / layer 25 / final

- explained variance: PC1=0.165, PC2=0.118, PC3=0.099

| group | PC1 mean | PC2 mean | PC3 mean |
|---|---:|---:|---:|
| assistant_consistent_self | +33.978 | -1.741 | +1.114 |
| human_biographical_self | -27.632 | -0.766 | +15.583 |
| roleplay_drift_self | -6.345 | +2.506 | -16.697 |

## Interpretation Limits

- 这里的 separation 是组间 centroid 距离除以组内距离的诊断指标，不是因果方向。
- mapping_score 只测试裸文本中 I/you 与 assistant/user 名词实体的相对相似性，不是 chat template 下的稳定身份绑定。
- 下一阶段应只在这些峰值层附近做方向验证、SAE feature 查询和激活干预。

## Artefacts

- prompts: `/Users/bobcute/Project/Emotion MI/results/phase1_sensitivity_probe_1b_it/prompts.json`
- activations: `/Users/bobcute/Project/Emotion MI/results/phase1_sensitivity_probe_1b_it/acts_positions_all_layers.npy`
- summary: `/Users/bobcute/Project/Emotion MI/results/phase1_sensitivity_probe_1b_it/summary.json`
- figures: `/Users/bobcute/Project/Emotion MI/results/phase1_sensitivity_probe_1b_it/phase1_layer_profiles.png`, `/Users/bobcute/Project/Emotion MI/results/phase1_sensitivity_probe_1b_it/phase1_normalized_profiles.png`