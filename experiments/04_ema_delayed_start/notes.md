# Experiment 04: Delayed-Start EMA

## Changes
- Delayed EMA start: only begins tracking at 50% of training (step 1000 of 2000)
- Decay warmup: `decay = min(0.997, (1 + ema_steps) / (10 + ema_steps))` ramps up gradually
- Avoids polluting the average with early noisy weights from steps 0-999
- EMA tracks only the last ~1000 steps where the model is more converged
- All other settings identical to exp02/03 (11L, 3x MLP, int6, WD=0.04, zstd-22, sliding eval stride=64)

## Results
| Metric | Exp02r (no EMA) | Exp03 (EMA from step 0) | Exp04 (delayed EMA) | Delta vs exp02r |
|--------|----------------|------------------------|---------------------|-----------------|
| val_bpb (standard, last step) | 1.2673 | 1.2669 | pending | pending |
| val_bpb (post-EMA, sliding) | -- | 1.2471 | pending | pending |
| val_bpb (int6 roundtrip, sliding) | 1.2425 | 1.2523 | pending | pending |

## Takeaway
Pending.
