# Experiment 06: HP tuning (SP1024 stack)

## Changes

- `MUON_WEIGHT_DECAY`: `0.04` -> `0.085` (lower end of 0.085–0.095 SOTA band)
- `QK_GAIN_INIT`: `1.5` -> `4.0` (learnable per-head query scaling; monotonic wins on the ladder)
- `WARMDOWN_ITERS`: `300` -> `1440` (72% of 2000-step run = warmdown=0.72)

All other settings identical to Exp 05: SP1024, 11L, MLP 3x, int6, EMA (decay=0.997, start 50%), XSA last 4 layers, sliding eval stride=64, 2000 iters, seed=1337.

## Results

| Metric | Exp05 (XSA last-4) | Exp06 (HP tuning) | Delta |
|--------|--------------------|------------------|------|
| val_bpb (standard, last step) | 1.2629 | 1.2531 | -0.0098 |
| val_bpb (post-EMA, sliding) | 1.2273 | 1.2187 | -0.0086 |
| val_bpb (int6 roundtrip, sliding) | 1.2347 | 1.2589 | +0.0242 |
| int6+zstd-22 payload (bytes) | 13,521,163 | 10,007,476 | -3,513,687 |
| step_avg / peak memory | 922ms / 14,798 MiB | 919ms / 14,795 MiB | -3ms / -3 MiB |

## Takeaway

WD 0.085, QK gain 4.0, and 72% warmdown clearly improve **pre-quant** BPB (standard and post-EMA sliding by ~0.01), and the int6 **payload compresses more** (smaller bytes on disk) — consistent with the roadmap’s “higher WD shrinks weight magnitudes” story. The **int6 roundtrip val_bpb regresses** by +0.0242 vs Exp 05, so the primary submission metric is worse on this stack: quant roundtrip is not monotonic with the float metrics here. This matches the plan’s “regression at SP1024/MLP 3x” branch: next step is Exp 07 (SP4096 + 4x MLP) where WD–quantization synergy is expected to pay, possibly with an A/B on `MUON_WEIGHT_DECAY` (0.04 vs 0.085) if int6 remains sensitive.
