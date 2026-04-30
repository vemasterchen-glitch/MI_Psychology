# DEG Ordinal Response Trend

Question: as blame DEG increases from 1 to 5, does layer activation response get larger?
Primary response metric: `||act(item) - neutral_mean||` per layer. This file uses float32 re-extracted clean PT and IT caches with zero item/layer checks.

## PT
- Significant positive linear DEG trend layers: 34/34.
- Significant negative linear DEG trend layers: 0/34.
| Layer | DEG1 | DEG2 | DEG3 | DEG4 | DEG5 | slope/DEG | q | Spearman rho |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L8 | 3926.4 | 3349.8 | 7732.1 | 7675.8 | 7408.8 | 1129.1 | 3.45e-08 | 0.57 |
| L17 | 7354.4 | 6644.8 | 14500.0 | 14563.6 | 14027.1 | 2126.4 | 2.00e-08 | 0.57 |
| L25 | 7552.6 | 7725.6 | 12795.5 | 13576.2 | 12946.4 | 1663.8 | 1.42e-08 | 0.61 |
| L33 | 8470.4 | 9063.3 | 9298.4 | 10497.1 | 9946.0 | 438.5 | 4.96e-04 | 0.47 |

Strongest positive slopes: L15 slope=2210.2 q=2.00e-08, L14 slope=2210.0 q=2.00e-08, L17 slope=2126.4 q=2.00e-08, L16 slope=2055.0 q=2.00e-08, L18 slope=2015.5 q=2.00e-08, L13 slope=1989.5 q=2.00e-08

## IT
- Significant positive linear DEG trend layers: 34/34.
- Significant negative linear DEG trend layers: 0/34.
| Layer | DEG1 | DEG2 | DEG3 | DEG4 | DEG5 | slope/DEG | q | Spearman rho |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L8 | 4193.7 | 3544.4 | 7979.9 | 8115.5 | 7934.4 | 1205.3 | 1.16e-08 | 0.59 |
| L17 | 7483.2 | 6753.3 | 14800.3 | 15371.9 | 15196.8 | 2404.6 | 1.91e-09 | 0.63 |
| L25 | 7798.5 | 7877.1 | 14686.4 | 15932.1 | 15490.4 | 2343.9 | 5.00e-10 | 0.66 |
| L33 | 7725.8 | 7852.3 | 8915.1 | 10884.4 | 10477.2 | 853.5 | 5.36e-05 | 0.45 |

Strongest positive slopes: L23 slope=2553.4 q=5.00e-10, L24 slope=2419.7 q=5.00e-10, L17 slope=2404.6 q=1.91e-09, L18 slope=2384.4 q=9.17e-10, L14 slope=2377.0 q=4.18e-09, L15 slope=2375.7 q=3.90e-09

