# Experiment 02: Int6 Quantization + 11L + 3x MLP

## Changes
- Increased model depth from 9 to 11 layers (`NUM_LAYERS=11`)
- Increased MLP expansion from 2x to 3x (`MLP_MULT=3`), total params 26.5M
- Introduced int6 quantization (range [-31, 31] in int8 containers) for most weight matrices
- Embeddings (`tok_emb`) kept at full int8 ([-127, 127]) since they lack STE protection
- Added fake int6 quantization with STE in `CastedLinear.forward` during training (quantization-aware training)
- Switched compression from zlib-9 to zstd-22 (exploits sparse high bits of int6-in-int8)
- Added Muon weight decay (`MUON_WEIGHT_DECAY=0.04`)
- Sliding window eval (stride=64) carried over from exp01

## Results
| Seed | val_bpb (pre-quant) | val_bpb (int6 roundtrip) | Delta vs baseline (1.3712) |
|------|---------------------|--------------------------|---------------------------|
| 1337 | 1.3865              | 1.5244                   | +0.1532                   |

- Stopped early at step 659/20000 due to 600s wallclock cap (bigger model = slower steps, ~911ms/step)
- int6+zstd-22 artifact size: 8,558,303 bytes (payload ratio 3.94x)
- Peak memory: 14,685 MiB

## Takeaway
Regression: the larger model (11L, 3x MLP) only trained for 659 steps under the wallclock cap, far too few to converge. The int6 quantization gap is also very large (+0.138 bpb), suggesting the model hasn't trained long enough for STE to properly condition the weights. Needs either more wall-clock time or a smaller/faster architecture to be viable.

## 2000-step rerun (1xA100, 2000 steps, ~30 min)

Fair comparison with exp01 at equal step budget (no wallclock cap).

| Metric | 600s cap (659 steps) | 2000 steps | Delta |
|--------|---------------------|------------|-------|
| val_bpb (standard, last step) | 1.3865 | 1.2673 | -0.1192 |
| val_bpb (int6+zstd roundtrip, sliding) | 1.5244 | 1.2425 | -0.2819 |
| artifact size | 8,558,303 | 13,565,076 | +5.0MB |
| step_avg | 911ms | 910ms | -- |
| peak memory | 14,685 MiB | 14,685 MiB | -- |

### Head-to-head vs exp01 baseline (both at 2000 steps)

| Metric | Exp01 (9L/2x/int8) | Exp02 (11L/3x/int6) | Delta |
|--------|--------------------|--------------------|-------|
| val_bpb (standard) | 1.2953 | 1.2673 | **-0.0280** |
| val_bpb (roundtrip, sliding) | 1.2634 | 1.2425 | **-0.0209** |
| quant gap (standard - roundtrip) | +0.0319 | +0.0248 | -0.0071 |
| artifact size | 15.5MB | 13.6MB | **-1.9MB** |

The int6+11L+3x MLP model wins on all metrics when given enough training steps.
