# Experiment Log

| # | Name | val_bpb | Delta vs prev | Delta vs baseline | Date |
|---|------|---------|---------------|-------------------|------|
| 00 | Baseline (stock, 1xA100) | 1.3712 | -- | -- | 2026-03-22 |
| 01 | + Sliding window eval (stride=64) | 1.3373 | -0.0339 | -0.0339 | 2026-03-22 |
| 02 | Int6 + 11L + 3x MLP + WD + zstd | 1.5244 | +0.1871 | +0.1532 | 2026-04-02 |
| 01r | Exp01 rerun @ 2000 steps | 1.2634 | -- | -0.1078 | 2026-04-02 |
| 02r | Exp02 rerun @ 2000 steps | 1.2425 | -0.0209 | -0.1287 | 2026-04-02 |
| 03 | + EMA weights (decay=0.997) | 1.2523 | +0.0098 | -0.1189 | 2026-04-02 |
