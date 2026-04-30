# Blame Activation Strength Relative to Neutral

Metric: RMS activation over hidden dimensions. For interpretation across PT and IT, the main curve is model-internal baseline-normalized strength: `mean RMS(blame) - mean RMS(neutral)` per layer. Significance is Welch t-test for blame vs neutral within each model, BH-corrected across 34 layers.

## PT
- Significant blame-vs-neutral layers: 34/34.
| Layer | blame RMS | neutral RMS | delta | delta % | q | d |
|---:|---:|---:|---:|---:|---:|---:|
| L0 | 38.616 | 55.484 | -16.869 | -30.4% | 2.18e-06 | -1.35 |
| L8 | 341.794 | 509.555 | -167.761 | -32.9% | 3.34e-07 | -1.28 |
| L17 | 1017.346 | 1300.033 | -282.687 | -21.7% | 3.12e-06 | -1.38 |
| L25 | 1471.669 | 1727.095 | -255.426 | -14.8% | 4.76e-06 | -1.08 |
| L33 | 1473.579 | 1633.083 | -159.504 | -9.8% | 1.69e-04 | -0.66 |

Strongest absolute effects: L13 d=-1.75, L6 d=-1.69, L11 d=-1.69, L14 d=-1.68, L5 d=-1.67

## IT
- Significant blame-vs-neutral layers: 21/34.
| Layer | blame RMS | neutral RMS | delta | delta % | q | d |
|---:|---:|---:|---:|---:|---:|---:|
| L0 | 0.704 | 0.462 | 0.242 | 52.3% | 3.47e-01 | 0.12 |
| L8 | 134.249 | 128.285 | 5.964 | 4.6% | 5.23e-01 | 0.14 |
| L17 | 402.830 | 441.646 | -38.816 | -8.8% | 2.31e-02 | -0.35 |
| L25 | 573.686 | 679.768 | -106.082 | -15.6% | 9.44e-04 | -0.48 |
| L33 | 674.154 | 820.792 | -146.639 | -17.9% | 1.04e-02 | -0.57 |

Strongest absolute effects: L5 d=0.75, L32 d=-0.62, L6 d=0.58, L33 d=-0.57, L31 d=-0.55

## IT vs PT on baseline-centered blame
- Significant layers: 31/34.
| Layer | PT delta | IT delta | IT-PT delta | q | dz |
|---:|---:|---:|---:|---:|---:|
| L0 | -16.869 | 0.242 | 17.111 | 1.34e-18 | 1.34 |
| L8 | -167.761 | 5.964 | 173.725 | 1.37e-16 | 1.20 |
| L17 | -282.687 | -38.816 | 243.871 | 1.13e-13 | 1.02 |
| L25 | -255.426 | -106.082 | 149.344 | 1.86e-04 | 0.44 |
| L33 | -159.504 | -146.639 | 12.865 | 7.56e-01 | 0.03 |
