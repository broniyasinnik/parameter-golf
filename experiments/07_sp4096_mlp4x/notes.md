# Experiment 07: SP4096 + MLP 4x (A/B on Muon weight decay)

## Changes vs Exp 06

- `VOCAB_SIZE`: `1024` -> `4096` (SP4096 BPE; tokenizer + dataset rebuilt locally from FineWeb docs since HF only publishes SP1024)
- `DATA_PATH`: `./data/datasets/fineweb10B_sp1024/` -> `./data/datasets/fineweb10B_sp4096/`
- `TOKENIZER_PATH`: SP1024 model -> `./data/tokenizers/fineweb_4096_bpe.model`
- `MLP_MULT`: `3` -> `4` (added body capacity to absorb the larger vocab)
- A/B on `MUON_WEIGHT_DECAY`:
  - **07A** = `0.085` (Exp 06 setting)
  - **07B** = `0.04` (Exp 05 / pre-Exp 06 setting)

All other settings copied from Exp 06: 11L, XSA last 4, EMA (decay=0.997, start 50%), `QK_GAIN_INIT=4.0`, `WARMDOWN_ITERS=1440`, `ITERATIONS=2000`, `SEED=1337`, sliding eval stride=64.

Model is meaningfully larger: `model_params=33,841,752` (vs 26,501,720 in Exp 06), peak GPU memory ~16,756 MiB (vs 14,795 MiB).

## Results


| Metric                                | Exp06 (SP1024, WD0.085) | Exp07A (SP4096, WD0.085) | Exp07B (SP4096, WD0.04) |
| ------------------------------------- | ----------------------- | ------------------------ | ----------------------- |
| val_bpb (standard, last step)         | 1.2531                  | 1.2110                   | 1.2100                  |
| val_bpb (post-EMA, sliding)           | 1.2187                  | 1.1839                   | 1.1833                  |
| **val_bpb (int6 roundtrip, sliding)** | **1.2589**              | **1.2414**               | **1.2019**              |
| int6+zstd-22 payload (bytes)          | 10,007,476              | 13,108,281               | 16,576,588              |
| step_avg / peak memory                | 919ms / 14,795 MiB      | 1,043ms / 16,756 MiB     | 1,042ms / 16,756 MiB    |


## Takeaway

SP4096 + 4x MLP is a clear win across the board: pre-quant, post-EMA, and int6 roundtrip all improve vs Exp 06. The A/B on Muon weight decay is decisive on the int6 metric — **WD=0.04 (07B) wins by -0.0395 vs WD=0.085 (07A)**, taking int6 val_bpb from **1.2589 → 1.2019** (-0.0570 vs Exp 06). Pre-quant metrics are essentially tied between A and B (07B is ~0.001 better), so the WD effect lives almost entirely in the quantization/compression interaction:

- WD=0.085 produced a smaller int6+zstd payload (~~13.1MB) than WD=0.04 (~~16.6MB), confirming the "WD shrinks weight magnitudes -> better Brotli/zstd compression" intuition.
- But on this larger model, the smaller payload comes with a markedly worse int6 roundtrip val_bpb. Higher WD is over-shrinking weights for the quantizer in this regime, and the float improvement does not survive the int6 round trip.

This inverts the Exp 06 finding (where WD=0.085 helped float metrics but hurt int6 on the smaller stack). At SP4096+4x, the principled choice is **WD=0.04**: it gives the best int6 BPB at the cost of ~3.5MB more payload. Going forward (Exp 08 / SDClip work), keep WD at 0.04 unless a quantization-aware path explicitly wants the smaller payload.

Step time grew from 919ms to ~1042ms (+13%); peak memory +13%. Both are expected from the wider MLP and larger embedding table, well within the A100 budget.