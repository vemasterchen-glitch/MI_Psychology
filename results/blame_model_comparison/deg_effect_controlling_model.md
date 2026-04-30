# DEG Effect Controlling for Model

Metric: baseline-centered activation strength, `RMS(blame item) - mean RMS(neutral)` within each model/layer.
Main test per layer: OLS nested F-test `activation ~ model + C(DEG)` vs `activation ~ model`. BH correction is applied across 34 layers per test family.

## DEG categorical effect controlling PT/IT
- Significant layers: 8/34.
| Layer | F | q | partial eta2 | slope |
|---:|---:|---:|---:|---:|
| L21 | 7.874 | 2.11e-04 | 0.170 |  |
| L19 | 7.627 | 2.11e-04 | 0.165 |  |
| L18 | 7.199 | 2.76e-04 | 0.158 |  |
| L33 | 5.533 | 2.93e-03 | 0.126 |  |
| L29 | 4.962 | 5.88e-03 | 0.114 |  |
| L25 | 3.932 | 2.30e-02 | 0.093 |  |

## DEG linear trend controlling PT/IT
- Significant layers: 7/34.
| Layer | F | q | partial eta2 | slope |
|---:|---:|---:|---:|---:|
| L21 | 15.712 | 2.66e-03 | 0.091 | 55.313 |
| L19 | 15.016 | 2.66e-03 | 0.087 | 52.245 |
| L18 | 14.147 | 2.70e-03 | 0.083 | 49.524 |
| L27 | 13.184 | 2.70e-03 | 0.077 | 55.324 |
| L29 | 13.069 | 2.70e-03 | 0.077 | 57.331 |
| L33 | 12.736 | 2.70e-03 | 0.075 | 50.199 |

## PT/IT x DEG interaction
- Significant layers: 15/34.
| Layer | F | q | partial eta2 | slope |
|---:|---:|---:|---:|---:|
| L12 | 7.563 | 4.79e-04 | 0.168 |  |
| L32 | 6.500 | 1.28e-03 | 0.148 |  |
| L24 | 6.048 | 1.39e-03 | 0.139 |  |
| L31 | 6.012 | 1.39e-03 | 0.138 |  |
| L33 | 5.617 | 2.08e-03 | 0.130 |  |
| L30 | 4.956 | 4.31e-03 | 0.117 |  |

## PT DEG categorical
- Significant layers: 2/34.
| Layer | F | q | partial eta2 | slope |
|---:|---:|---:|---:|---:|
| L12 | 4.876 | 4.39e-02 | 0.206 |  |
| L8 | 4.502 | 4.39e-02 | 0.194 |  |
| L21 | 3.593 | 1.05e-01 | 0.161 |  |
| L19 | 3.439 | 1.05e-01 | 0.155 |  |
| L18 | 3.247 | 1.11e-01 | 0.148 |  |
| L24 | 3.018 | 1.30e-01 | 0.139 |  |

## IT DEG categorical
- Significant layers: 28/34.
| Layer | F | q | partial eta2 | slope |
|---:|---:|---:|---:|---:|
| L32 | 11.902 | 5.23e-06 | 0.388 |  |
| L33 | 10.840 | 7.19e-06 | 0.366 |  |
| L31 | 10.669 | 7.19e-06 | 0.363 |  |
| L30 | 10.101 | 7.19e-06 | 0.350 |  |
| L15 | 9.840 | 7.19e-06 | 0.344 |  |
| L9 | 9.773 | 7.19e-06 | 0.343 |  |

## DEG means at report layers
| Model | Layer | DEG1 | DEG2 | DEG3 | DEG4 | DEG5 |
|---|---:|---:|---:|---:|---:|---:|
| PT | L8 | -126.352 | -91.811 | -241.091 | -144.453 | -235.096 |
| PT | L17 | -304.493 | -269.678 | -285.754 | -282.125 | -271.384 |
| PT | L25 | -307.692 | -263.748 | -229.780 | -245.074 | -230.835 |
| PT | L33 | -272.934 | -233.980 | -81.797 | -129.812 | -78.994 |
| IT | L8 | -34.752 | 22.197 | 3.427 | 20.451 | 18.498 |
| IT | L17 | -170.397 | -1.989 | -15.410 | -9.071 | 2.787 |
| IT | L25 | -259.191 | 0.462 | -277.484 | 0.215 | 5.587 |
| IT | L33 | -334.023 | 28.475 | -347.730 | -32.190 | -47.724 |
