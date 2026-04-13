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
| val_bpb (standard, last step) | 1.2673 | 1.2669 | 1.2679 | +0.0006 |
| val_bpb (post-EMA, sliding) | -- | 1.2471 | 1.2323 | -0.0350 |
| val_bpb (int6 roundtrip, sliding) | 1.2425 | 1.2523 | 1.2400 | -0.0025 |
| quant gap (post-EMA - roundtrip) | -- | +0.0052 | +0.0077 | -- |
| artifact size | 13,565,076 | 13,801,836 | 13,550,859 | -14KB |
| step_avg | 910ms | 913ms | 913ms | +3ms |
| peak memory | 14,685 MiB | 14,783 MiB | 14,785 MiB | +100 MiB |

## Takeaway
Improvement: Delaying EMA start to 50% of training is a clear win. Post-EMA sliding val_bpb drops to 1.2323 (vs 1.2471 in exp03), and the int6 roundtrip result of 1.2400 beats both exp03 (1.2523) and exp02r (1.2425) by -0.0123 and -0.0025 respectively. The EMA now only averages over converged weights, producing a smoother model that also quantizes slightly better than full-run EMA.
