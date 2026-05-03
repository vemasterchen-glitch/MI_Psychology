# 意识与自我材料 text-only - Gemma 3 1B IT

本版只输入 80 条原文 text，不使用 direct/quote/reflection/concept round，不使用 chat template。

## 输出概览

- texts: 80
- representation: mean-pooled token hidden states across the raw text
- activation peak layer: L22
- PCA PC1/PC2 explained variance: 0.078 / 0.053
- clusters: 8
- SAE layer: L24
- active SAE features per text: 1.3 ± 1.1

## 聚群内容

| cluster | name | size | stimuli |
|---:|---|---:|---|
| 0 | consciousness, perception, memory, and identity continuity | 8 | S023, S025, S031, S058, S059, S071, S072, S076 |
| 1 | non-self, nothingness, shadow, and negation | 12 | S017, S052, S053, S054, S055, S056, S060, S062, S064, S073, S074, S075 |
| 2 | consciousness, perception, memory, and identity continuity | 16 | S004, S005, S008, S009, S010, S026, S027, S034, S037, S042, S046, S048, S057, S063, S079, S080 |
| 3 | consciousness, perception, memory, and identity continuity | 19 | S007, S011, S016, S019, S020, S022, S024, S032, S033, S035, S036, S038, S039, S040, S041, S043, S044, S061, S067 |
| 4 | identity formula: self and ultimate reality | 2 | S029, S030 |
| 5 | first-person existence and self-assertion | 8 | S001, S002, S003, S045, S047, S049, S077, S078 |
| 6 | non-self, nothingness, shadow, and negation | 8 | S015, S050, S051, S065, S066, S068, S069, S070 |
| 7 | abstract consciousness, thought, and mind | 7 | S006, S012, S013, S014, S018, S021, S028 |

## SAE 最高激活 features

| rank | feature | mean | active rate | top stimuli |
|---:|---:|---:|---:|---|
| 1 | 705 | 254.52 | 0.50 | S007, S011, S043, S019, S008 |
| 2 | 132 | 147.08 | 0.17 | S077, S064, S017, S057, S078 |
| 3 | 534 | 50.35 | 0.11 | S064, S057, S015, S019, S001 |
| 4 | 267 | 37.37 | 0.05 | S064, S077, S017, S057 |
| 5 | 207 | 21.33 | 0.05 | S077, S003, S078, S001 |
| 6 | 468 | 20.62 | 0.05 | S059, S058, S076, S071 |
| 7 | 962 | 19.82 | 0.04 | S055, S060, S042 |
| 8 | 18 | 15.35 | 0.04 | S040, S019, S018 |
| 9 | 1036 | 11.54 | 0.03 | S035, S036 |
| 10 | 651 | 10.45 | 0.03 | S036, S035 |
| 11 | 656 | 9.23 | 0.03 | S033, S031 |
| 12 | 2847 | 7.88 | 0.01 | S018 |
| 13 | 623 | 6.39 | 0.03 | S024, S054 |
| 14 | 1016 | 6.06 | 0.01 | S014 |
| 15 | 3484 | 5.40 | 0.01 | S059 |

## SAE 聚群区分 features

| rank | feature | F | p | eta2 | strongest cluster means |
|---:|---:|---:|---:|---:|---|
| 1 | 468 | 9.02 | 6.09e-08 | 0.467 | C0:206.2 / C1:0.0 / C2:0.0 |
| 2 | 207 | 8.89 | 7.58e-08 | 0.464 | C5:213.3 / C0:0.0 / C1:0.0 |
| 3 | 705 | 5.29 | 6.58e-05 | 0.340 | C3:439.7 / C2:371.4 / C7:290.3 |
| 4 | 132 | 2.68 | 1.59e-02 | 0.207 | C4:471.9 / C1:440.5 / C5:307.4 |
| 5 | 2847 | 1.56 | 1.60e-01 | 0.132 | C7:90.1 / C0:0.0 / C1:0.0 |
| 6 | 137 | 1.56 | 1.60e-01 | 0.132 | C7:50.6 / C0:0.0 / C1:0.0 |
| 7 | 1016 | 1.56 | 1.60e-01 | 0.132 | C7:69.2 / C0:0.0 / C1:0.0 |
| 8 | 962 | 1.33 | 2.51e-01 | 0.114 | C1:98.8 / C2:25.0 / C0:0.0 |
| 9 | 129 | 1.32 | 2.52e-01 | 0.114 | C6:48.5 / C0:0.0 / C1:0.0 |
| 10 | 6401 | 1.32 | 2.52e-01 | 0.114 | C5:52.4 / C0:0.0 / C1:0.0 |
| 11 | 2292 | 1.32 | 2.52e-01 | 0.114 | C0:52.6 / C1:0.0 / C2:0.0 |
| 12 | 3484 | 1.32 | 2.52e-01 | 0.114 | C0:54.0 / C1:0.0 / C2:0.0 |
| 13 | 267 | 1.06 | 3.96e-01 | 0.094 | C1:151.0 / C5:74.6 / C2:36.3 |
| 14 | 651 | 0.92 | 4.96e-01 | 0.082 | C3:44.0 / C0:0.0 / C1:0.0 |
| 15 | 1036 | 0.91 | 5.05e-01 | 0.081 | C3:48.6 / C0:0.0 / C1:0.0 |

## 主要图表

- `text_activation_map.png`: 80 条原文刺激 x layer 激活地图
- `text_pca_cluster_map.png`: text-only PCA/PAC 聚群图
- `text_cluster_distance.png`: text-only cosine distance 聚群热图
- `sae_cluster_feature_heatmap.png`: top SAE features x cluster
- `token_heatmaps/`: top feature 在原文 token 上的 heatmap
- `feature_explanations.md`: feature 解释