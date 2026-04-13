# Experiment 05: XSA (Exclusive Self Attention)

## Changes
- Added Exclusive Self Attention (XSA) on the last 4 layers (layers 7-10 of 11)
- XSA subtracts the self-value projection from attention output, forcing cross-position information mixing
- Zero new parameters, estimated ~2ms/step overhead
- Efficient GQA-aware implementation using reshape (no repeat_interleave)
- All other settings identical to exp04 (11L, 3x MLP, int6, WD=0.04, EMA delayed start 50%, sliding eval stride=64)

## Reference
- arXiv:2603.09078 (Exclusive Self Attention)
- Record submissions show ~0.002 BPB improvement from XSA on similar stacks

## Results
| Metric | Exp04 (no XSA) | Exp05 (XSA last 4) | Delta |
|--------|----------------|---------------------|-------|
| val_bpb (standard, last step) | 1.2679 | 1.2629 | -0.0050 |
| val_bpb (post-EMA, sliding) | 1.2323 | 1.2273 | -0.0050 |
| val_bpb (int6 roundtrip, sliding) | 1.2400 | 1.2347 | -0.0053 |
| artifact size | 13,550,859 | 13,521,163 | -30KB |
| step_avg | 913ms | 922ms | +9ms |
| peak memory | 14,785 MiB | 14,798 MiB | +13 MiB |

## Takeaway
XSA on the last 4 layers is a clear win: -0.0053 BPB on the final int6 roundtrip metric (1.2400 -> 1.2347) with zero new parameters and only +9ms/step overhead. The improvement is consistent across all evaluation modes (standard, post-EMA, int6 roundtrip). Artifact size is slightly smaller. This confirms the record submissions' finding that XSA reliably improves results at minimal cost.
