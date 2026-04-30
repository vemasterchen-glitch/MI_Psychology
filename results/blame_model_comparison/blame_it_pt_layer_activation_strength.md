# IT vs PT Blame Layer Activation Strength

Activation strength is measured as RMS over hidden dimensions. The primary test is paired by identical blame sentence at each layer: IT strength vs PT strength, Benjamini-Hochberg corrected across 34 layers per metric.

## Raw RMS

- Significant IT vs PT layers after BH correction: 34/34.
| Layer | PT mean | IT mean | IT/PT | IT-PT | q | dz |
|---:|---:|---:|---:|---:|---:|---:|
| L4 | 151.4105 | 0.7310 | 0.00x | -150.6794 | 7.80e-50 | -4.10 |
| L2 | 57.4988 | 0.7310 | 0.01x | -56.7677 | 4.12e-44 | -3.40 |
| L3 | 70.8080 | 0.7310 | 0.01x | -70.0770 | 9.74e-44 | -3.34 |
| L5 | 209.3937 | 30.1096 | 0.14x | -179.2841 | 9.74e-44 | -3.33 |
| L22 | 1280.9401 | 483.9860 | 0.38x | -796.9541 | 1.33e-43 | -3.30 |
| L0 | 38.6155 | 0.7043 | 0.02x | -37.9112 | 2.40e-40 | -2.96 |

## Neutral-centered RMS

- Significant IT vs PT layers after BH correction: 29/34.
| Layer | PT mean | IT mean | IT/PT | IT-PT | q | dz |
|---:|---:|---:|---:|---:|---:|---:|
| L4 | 62.5702 | 0.6796 | 0.01x | -61.8906 | 1.19e-26 | -1.91 |
| L6 | 107.5404 | 19.3320 | 0.18x | -88.2084 | 4.23e-22 | -1.58 |
| L3 | 29.3957 | 0.6796 | 0.02x | -28.7161 | 2.00e-21 | -1.53 |
| L5 | 91.3877 | 18.0330 | 0.20x | -73.3547 | 6.82e-21 | -1.48 |
| L2 | 22.3441 | 0.6796 | 0.03x | -21.6645 | 3.56e-20 | -1.43 |
| L0 | 18.3270 | 0.6779 | 0.04x | -17.6490 | 5.40e-20 | -1.42 |

## PT raw blame vs neutral

- Significant layers after BH correction: 34/34.
| Layer | blame mean | neutral mean | diff | q | d |
|---:|---:|---:|---:|---:|---:|
| L13 | 750.0478 | 999.1235 | -249.0757 | 2.18e-06 | -1.75 |
| L6 | 247.7627 | 350.7478 | -102.9851 | 5.17e-06 | -1.69 |
| L11 | 608.5325 | 807.7357 | -199.2032 | 2.18e-06 | -1.69 |
| L14 | 844.3267 | 1115.2708 | -270.9441 | 3.12e-06 | -1.68 |
| L5 | 209.3937 | 296.4386 | -87.0449 | 5.15e-06 | -1.67 |
| L10 | 538.8704 | 698.8688 | -159.9985 | 4.76e-06 | -1.63 |

## IT raw blame vs neutral

- Significant layers after BH correction: 21/34.
| Layer | blame mean | neutral mean | diff | q | d |
|---:|---:|---:|---:|---:|---:|
| L5 | 30.1096 | 20.3429 | 9.7667 | 2.45e-02 | 0.75 |
| L32 | 833.9115 | 1017.5307 | -183.6193 | 6.06e-04 | -0.62 |
| L6 | 58.2841 | 47.6704 | 10.6138 | 1.18e-01 | 0.58 |
| L33 | 674.1537 | 820.7923 | -146.6386 | 1.04e-02 | -0.57 |
| L31 | 834.0539 | 993.2238 | -159.1698 | 7.38e-04 | -0.55 |
| L30 | 781.9222 | 928.9448 | -147.0226 | 9.79e-04 | -0.54 |

