# 意识与自我材料 text-only - Gemma 3 1B IT

本版只输入 80 条原文 text，不使用 direct/quote/reflection/concept round，不使用 chat template。

## 输出概览

- texts: 80
- representation: mean-pooled token hidden states across the raw text
- activation peak layer: L24
- PCA PC1/PC2 explained variance: 0.081 / 0.069
- clusters: 8
- SAE layer: L24
- active SAE features per text: 1.7 ± 1.5

## 聚群内容

| cluster | name | size | stimuli |
|---:|---|---:|---|
| 0 | Non-self, nothingness, negation | 7 | S017, S042, S057, S058, S077, S078, S079 |
| 1 | First-person certainty and self-reference | 11 | S015, S030, S045, S047, S050, S052, S065, S066, S068, S069, S070 |
| 2 | German/French inwardness and alterity | 10 | S059, S060, S061, S062, S072, S073, S074, S075, S076, S080 |
| 3 | Chinese self-cultivation / transformation | 9 | S031, S032, S033, S034, S035, S036, S037, S038, S039 |
| 4 | First-person certainty and self-reference | 7 | S006, S012, S013, S014, S018, S021, S028 |
| 5 | First-person certainty and self-reference | 22 | S007, S008, S016, S019, S020, S022, S023, S024, S026, S027, S040, S041, S043, S044, S046, S053, S054, S055, S056, S063, S067, S071 |
| 6 | First-person certainty and self-reference | 3 | S003, S029, S064 |
| 7 | First-person certainty and self-reference | 11 | S001, S002, S004, S005, S009, S010, S011, S025, S048, S049, S051 |

## SAE 最高激活 features

| rank | feature | mean | active rate | top stimuli |
|---:|---:|---:|---:|---|
| 1 | 705 | 185.23 | 0.36 | S007, S011, S043, S019, S008 |
| 2 | 949 | 127.20 | 0.12 | S061, S075, S076, S059, S074 |
| 3 | 132 | 122.62 | 0.12 | S077, S064, S030, S029, S020 |
| 4 | 2076 | 100.60 | 0.11 | S039, S035, S033, S032, S038 |
| 5 | 282 | 76.09 | 0.05 | S042, S058, S017, S057 |
| 6 | 366 | 75.55 | 0.11 | S061, S076, S075, S062, S059 |
| 7 | 534 | 74.67 | 0.16 | S020, S064, S030, S015, S019 |
| 8 | 1993 | 49.69 | 0.04 | S079, S077, S078 |
| 9 | 267 | 41.53 | 0.05 | S064, S030, S077, S020 |
| 10 | 3779 | 35.99 | 0.06 | S076, S060, S059, S074, S073 |
| 11 | 2139 | 25.48 | 0.04 | S061, S080, S062 |
| 12 | 3624 | 23.04 | 0.04 | S042, S058, S057 |
| 13 | 46 | 19.21 | 0.03 | S079, S077 |
| 14 | 18 | 15.35 | 0.04 | S040, S019, S018 |
| 15 | 962 | 13.00 | 0.03 | S055, S020 |

## SAE 聚群区分 features

| rank | feature | F | p | eta2 | strongest cluster means |
|---:|---:|---:|---:|---:|---|
| 1 | 949 | 924.10 | 0.00e+00 | 0.989 | C2:1017.6 / C0:0.0 / C1:0.0 |
| 2 | 2076 | 560.85 | 0.00e+00 | 0.982 | C3:894.2 / C0:0.0 / C1:0.0 |
| 3 | 366 | 58.84 | 2.92e-27 | 0.851 | C2:604.4 / C0:0.0 / C1:0.0 |
| 4 | 282 | 12.46 | 2.43e-10 | 0.548 | C0:869.6 / C1:0.0 / C2:0.0 |
| 5 | 705 | 8.42 | 1.73e-07 | 0.450 | C5:382.6 / C7:326.8 / C4:290.3 |
| 6 | 3779 | 8.15 | 2.80e-07 | 0.442 | C2:287.9 / C0:0.0 / C1:0.0 |
| 7 | 3624 | 6.27 | 9.31e-06 | 0.379 | C0:263.3 / C1:0.0 / C2:0.0 |
| 8 | 1993 | 6.11 | 1.28e-05 | 0.373 | C0:567.9 / C1:0.0 / C2:0.0 |
| 9 | 13968 | 4.95 | 1.31e-04 | 0.325 | C6:282.7 / C0:0.0 / C1:0.0 |
| 10 | 132 | 4.92 | 1.40e-04 | 0.324 | C6:843.8 / C0:534.8 / C1:136.3 |
| 11 | 2139 | 3.77 | 1.54e-03 | 0.268 | C2:203.8 / C0:0.0 / C1:0.0 |
| 12 | 46 | 3.71 | 1.77e-03 | 0.265 | C0:219.6 / C1:0.0 / C2:0.0 |
| 13 | 656 | 2.61 | 1.86e-02 | 0.202 | C3:85.0 / C0:0.0 / C1:0.0 |
| 14 | 267 | 2.22 | 4.26e-02 | 0.177 | C6:408.9 / C0:97.7 / C1:70.8 |
| 15 | 125 | 2.11 | 5.27e-02 | 0.171 | C2:76.9 / C0:0.0 / C1:0.0 |

## 主要图表

- `text_activation_map.png`: 80 条原文刺激 x layer 激活地图
- `text_pca_cluster_map.png`: text-only PCA/PAC 聚群图
- `text_cluster_distance.png`: text-only cosine distance 聚群热图
- `sae_cluster_feature_heatmap.png`: top SAE features x cluster
- `token_heatmaps/`: top feature 在原文 token 上的 heatmap
- `feature_explanations.md`: feature 解释