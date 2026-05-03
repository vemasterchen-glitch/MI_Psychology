# 意识与自我材料 - Gemma 3 1B IT

## 输出概览

- prompts: 320 = 80 stimuli x 4 rounds
- activation peak layer: L24
- PCA PC1/PC2 explained variance: 0.155 / 0.130
- clusters: 8
- SAE layer: L24
- active SAE features per prompt: 15.8 ± 3.0

## 跨轮次稳定高激活 features

| rank | feature | mean | min round mean | active rate | round means |
|---:|---:|---:|---:|---:|---|
| 1 | 946 | 1875.01 | 1500.87 | 1.00 | direct:1500.9 / quote:2036.7 / reflection:1953.4 / concept:2009.0 |
| 2 | 575 | 838.83 | 722.03 | 0.93 | direct:722.0 / quote:926.0 / reflection:830.7 / concept:876.6 |
| 3 | 1 | 504.68 | 475.41 | 0.94 | direct:475.4 / quote:505.1 / reflection:538.7 / concept:499.5 |
| 4 | 477 | 611.59 | 426.92 | 0.94 | direct:584.0 / quote:751.3 / reflection:426.9 / concept:684.2 |
| 5 | 14089 | 321.26 | 288.97 | 0.79 | direct:306.7 / quote:375.1 / reflection:314.2 / concept:289.0 |
| 6 | 2181 | 530.49 | 244.38 | 0.65 | direct:853.4 / quote:397.8 / reflection:244.4 / concept:626.3 |
| 7 | 12969 | 243.07 | 218.62 | 0.54 | direct:235.4 / quote:218.6 / reflection:270.4 / concept:247.9 |
| 8 | 14167 | 246.35 | 147.69 | 0.45 | direct:147.7 / quote:199.6 / reflection:442.7 / concept:195.4 |
| 9 | 195 | 341.73 | 139.67 | 0.79 | direct:139.7 / quote:386.0 / reflection:430.1 / concept:411.2 |
| 10 | 1798 | 407.37 | 125.86 | 0.76 | direct:125.9 / quote:550.9 / reflection:575.5 / concept:377.3 |
| 11 | 184 | 246.20 | 123.60 | 0.56 | direct:263.9 / quote:123.6 / reflection:365.7 / concept:231.6 |
| 12 | 7346 | 192.79 | 81.58 | 0.34 | direct:174.0 / quote:254.8 / reflection:81.6 / concept:260.8 |
| 13 | 754 | 178.06 | 74.00 | 0.59 | direct:74.0 / quote:201.5 / reflection:306.5 / concept:130.2 |
| 14 | 2214 | 199.64 | 73.45 | 0.65 | direct:169.5 / quote:199.5 / reflection:73.5 / concept:356.1 |
| 15 | 656 | 74.79 | 66.72 | 0.12 | direct:66.7 / quote:73.5 / reflection:71.4 / concept:87.5 |

## 跨轮次显著不同 features

| rank | feature | F | p | eta2 | round means |
|---:|---:|---:|---:|---:|---|
| 1 | 9019 | 112.96 | 0.00e+00 | 0.517 | direct:0.0 / quote:0.0 / reflection:190.0 / concept:0.0 |
| 2 | 1798 | 110.81 | 0.00e+00 | 0.513 | direct:125.9 / quote:550.9 / reflection:575.5 / concept:377.3 |
| 3 | 195 | 63.34 | 4.33e-32 | 0.376 | direct:139.7 / quote:386.0 / reflection:430.1 / concept:411.2 |
| 4 | 278 | 63.19 | 4.96e-32 | 0.375 | direct:0.0 / quote:0.0 / reflection:147.2 / concept:3.8 |
| 5 | 946 | 55.90 | 5.15e-29 | 0.347 | direct:1500.9 / quote:2036.7 / reflection:1953.4 / concept:2009.0 |
| 6 | 754 | 51.15 | 5.66e-27 | 0.327 | direct:74.0 / quote:201.5 / reflection:306.5 / concept:130.2 |
| 7 | 2214 | 44.72 | 4.08e-24 | 0.298 | direct:169.5 / quote:199.5 / reflection:73.5 / concept:356.1 |
| 8 | 355 | 42.64 | 3.66e-23 | 0.288 | direct:36.4 / quote:381.0 / reflection:488.4 / concept:255.0 |
| 9 | 477 | 42.41 | 4.67e-23 | 0.287 | direct:584.0 / quote:751.3 / reflection:426.9 / concept:684.2 |
| 10 | 2181 | 31.17 | 1.12e-17 | 0.228 | direct:853.4 / quote:397.8 / reflection:244.4 / concept:626.3 |
| 11 | 1753 | 30.09 | 3.87e-17 | 0.222 | direct:10.1 / quote:189.3 / reflection:4.8 / concept:115.9 |
| 12 | 10950 | 28.59 | 2.21e-16 | 0.213 | direct:15.6 / quote:107.3 / reflection:357.8 / concept:215.1 |
| 13 | 7647 | 27.00 | 1.42e-15 | 0.204 | direct:9.1 / quote:19.5 / reflection:121.4 / concept:3.7 |
| 14 | 476 | 23.57 | 8.54e-14 | 0.183 | direct:141.0 / quote:0.0 / reflection:0.0 / concept:5.7 |
| 15 | 14167 | 22.02 | 5.65e-13 | 0.173 | direct:147.7 / quote:199.6 / reflection:442.7 / concept:195.4 |

## 主要图表

- `stimulus_activation_map.png`: 刺激级 layer activation map
- `round_activation_map.png`: round x layer activation map
- `pca_round_map.png`: PCA/PAC round pattern
- `pca_cluster_map.png`: PCA 聚群
- `stimulus_cluster_distance.png`: stimulus cosine distance clustering
- `sae_round_feature_heatmap.png`: top SAE features x rounds
- `token_heatmaps/`: top feature token heatmaps
- `feature_explanations.md`: feature 解释