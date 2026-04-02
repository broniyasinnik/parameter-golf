# Experiment 01: Sliding Window Eval

## Changes
- Added `eval_stride` (default 64) and `eval_batch_seqs` (default 32) hyperparameters
- Added `GPT.forward_logits()` method that returns raw `(batch, seq_len, vocab)` logits without computing loss
- Added `eval_val_sliding()` function: overlapping windows advance by `stride` tokens, only the last `stride` tokens per window (with 960+ tokens of context) are scored
- Wired sliding window into the final post-quantization eval call site; periodic training validation unchanged
- No training changes -- eval-only improvement

## Results (1xA100, 915 steps in 10 min)
| Metric | Baseline (standard eval) | Exp 01 (sliding window) | Delta |
|--------|-------------------------|------------------------|-------|
| val_bpb | 1.3712 | 1.3373 | -0.0339 |
| val_loss | 2.3152 | 2.2580 | -0.0572 |
| eval_time | 22s | 998s | +976s |
| steps | 909 | 915 | +6 |

Note: Both runs on 1xA100 (not 8xH100), so absolute bpb is higher than the leaderboard.
The baseline comparison is `logs/baseline_sp1024.txt` (standard eval, same hardware).
On 8xH100 the expected improvement is ~0.032 bpb (1.2244 -> ~1.192).

## Takeaway
Sliding window eval works as expected: -0.034 bpb for free. No training changes. Eval time is much longer on 1xGPU (998s vs ~70s on 8xH100) but still within the 10-min eval budget on the submission hardware.

## 2000-step rerun (1xA100, 2000 steps, ~22 min)

Fair comparison with exp02 at equal step budget (no wallclock cap).

| Metric | 600s cap (915 steps) | 2000 steps | Delta |
|--------|---------------------|------------|-------|
| val_bpb (standard, last step) | 1.3683 | 1.2953 | -0.0730 |
| val_bpb (int8+zlib roundtrip, sliding) | 1.3373 | 1.2634 | -0.0739 |
| artifact size | 12,031,507 | 15,496,641 | +3.5MB |
| step_avg | 656ms | 655ms | -- |
| peak memory | 10,889 MiB | 10,889 MiB | -- |
