# Gemma 3 4B IT vs PT Blame-Recipient Comparison

Source summaries:
- PT: `results/blame_recipient/blame_analysis.json`
- IT: `results/blame_recipient_it/blame_analysis.json`

## Main Readout

- Bifurcation layer: PT L33, IT L33.
- PCA PC1 variance: PT 95.8%, IT 77.8%.
- PCA PC2 variance: PT 0.8%, IT 1.8%.
- IT shows much stronger instruction-tuned separation for DEG=3, especially in the final layer.

## DEG Distance From DEG=1

| Layer | DEG | PT | IT | IT/PT |
|---:|---:|---:|---:|---:|
| L8 | 2 | 0.000027 | 0.000306 | 11.2x |
| L8 | 3 | 0.000073 | 0.000293 | 4.0x |
| L8 | 4 | 0.000065 | 0.000278 | 4.3x |
| L8 | 5 | 0.000090 | 0.000275 | 3.0x |
| L17 | 2 | 0.000038 | 0.000375 | 9.9x |
| L17 | 3 | 0.000107 | 0.000482 | 4.5x |
| L17 | 4 | 0.000111 | 0.000352 | 3.2x |
| L17 | 5 | 0.000129 | 0.000359 | 2.8x |
| L25 | 2 | 0.000208 | 0.003149 | 15.2x |
| L25 | 3 | 0.000570 | 0.016786 | 29.5x |
| L25 | 4 | 0.000603 | 0.002656 | 4.4x |
| L25 | 5 | 0.000648 | 0.002848 | 4.4x |
| L33 | 2 | 0.001059 | 0.009759 | 9.2x |
| L33 | 3 | 0.002355 | 0.144660 | 61.4x |
| L33 | 4 | 0.002586 | 0.009848 | 3.8x |
| L33 | 5 | 0.002711 | 0.010618 | 3.9x |

## MDL Pairwise Distances at L25

| Pair | PT | IT | IT/PT |
|---|---:|---:|---:|
| 行为_vs_输出 | 0.000255 | 0.002615 | 10.3x |
| 行为_vs_能力 | 0.000537 | 0.002973 | 5.5x |
| 行为_vs_价值观 | 0.000338 | 0.003375 | 10.0x |
| 输出_vs_能力 | 0.000664 | 0.003145 | 4.7x |
| 输出_vs_价值观 | 0.000501 | 0.003626 | 7.2x |
| 能力_vs_价值观 | 0.000535 | 0.001403 | 2.6x |

## DEG Pairwise Distances at L25

| Pair | PT | IT | IT/PT |
|---|---:|---:|---:|
| DEG1_vs_DEG2 | 0.000208 | 0.003149 | 15.1x |
| DEG1_vs_DEG3 | 0.000570 | 0.016786 | 29.5x |
| DEG1_vs_DEG4 | 0.000603 | 0.002656 | 4.4x |
| DEG1_vs_DEG5 | 0.000648 | 0.002848 | 4.4x |
| DEG2_vs_DEG3 | 0.000515 | 0.017309 | 33.6x |
| DEG2_vs_DEG4 | 0.000557 | 0.002267 | 4.1x |
| DEG2_vs_DEG5 | 0.000607 | 0.002109 | 3.5x |
| DEG3_vs_DEG4 | 0.000131 | 0.015056 | 114.6x |
| DEG3_vs_DEG5 | 0.000291 | 0.014617 | 50.3x |
| DEG4_vs_DEG5 | 0.000146 | 0.001767 | 12.1x |
