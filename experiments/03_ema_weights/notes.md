# Experiment 03: EMA Weight Averaging

## Changes
- Added EMA (Exponential Moving Average) weight averaging with decay=0.997
- EMA shadow state maintained in fp32 every training step
- EMA weights loaded into model after training, before quantization
- Diagnostic eval added to measure post-EMA val_bpb before int6 roundtrip
- All other settings identical to exp02 (11L, 3x MLP, int6, WD=0.04, zstd-22, sliding eval stride=64)

## Results
| Metric | Exp02r (no EMA) | Exp03 (EMA 0.997) | Delta |
|--------|----------------|--------------------|-------|
| val_bpb (standard, last step) | 1.2673 | 1.2669 | -0.0004 |
| val_bpb (post-EMA, sliding) | -- | 1.2471 | -- |
| val_bpb (int6 roundtrip, sliding) | 1.2425 | 1.2523 | +0.0098 |
| quant gap (post-EMA - roundtrip) | -- | +0.0052 | -- |
| artifact size | 13,565,076 | 13,801,836 | +237KB |
| step_avg | 910ms | 913ms | +3ms |
| peak memory | 14,685 MiB | 14,783 MiB | +98 MiB |

## Takeaway
Regression: EMA improved pre-quantization val_bpb (1.2471 vs 1.2669 standard), but the int6 roundtrip result (1.2523) is worse than exp02r's (1.2425) by +0.0098 bpb. The EMA-smoothed weights may have different value distributions that quantize less cleanly at int6 precision. The memory and speed costs are negligible as expected (~98 MiB, ~3ms/step).
