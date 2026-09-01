# P2 Soft-Gating Network Experiment Report

| Configuration | Overall MAE | Normal MAE | Shock MAE | Spike MAE (29-30) | Directional Acc (%) | Interval Coverage (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard Gate (Baseline) | 168.57 | 169.96 | 149.38 | 473.41 | 87.10 | 78.37 |
| Soft Gate (k=5, Smooth) | 226.12 | 225.86 | 229.68 | 606.96 | 84.72 | 24.60 |
| Soft Gate (k=10, Medium) | 210.80 | 209.30 | 231.41 | 549.87 | 85.71 | 30.56 |
| Soft Gate (k=20, Steep) | 192.16 | 189.08 | 234.81 | 487.52 | 86.51 | 39.29 |

*Decision Rule:* Keep the hard gate unless a soft gate improves both Overall MAE below ₹168.57 and Spike MAE below ₹473.41.
Recommendation: keep the hard gate because the soft gate did not beat the required thresholds. Best soft gate: Soft Gate (k=20, Steep).
