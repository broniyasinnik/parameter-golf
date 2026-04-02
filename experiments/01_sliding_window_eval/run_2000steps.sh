#!/bin/bash
# Rerun with 2000-step cap (no wallclock limit) for fair comparison with exp02
PYTHONUNBUFFERED=1 \
RUN_ID=exp01_sliding_window_eval_2000steps \
SEED=${SEED:-1337} \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=2000 \
WARMDOWN_ITERS=300 \
MAX_WALLCLOCK_SECONDS=0 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=32 \
uv run python3 experiments/01_sliding_window_eval/train_gpt.py \
  2>&1 | tee experiments/01_sliding_window_eval/train_2000steps_seed${SEED:-1337}.log
