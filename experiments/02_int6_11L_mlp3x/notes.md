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
