#!/bin/bash
# Exp03: EMA weight averaging (decay=0.997), otherwise identical to exp02
PYTHONUNBUFFERED=1 \
RUN_ID=exp03_ema_weights_2000steps \
SEED=${SEED:-1337} \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MLP_MULT=3 \
MUON_WEIGHT_DECAY=0.04 \
EMA_DECAY=0.997 \
ITERATIONS=2000 \
WARMDOWN_ITERS=300 \
MAX_WALLCLOCK_SECONDS=0 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=32 \
uv run python3 train_gpt.py \
  2>&1 | tee experiments/03_ema_weights/train_2000steps_seed${SEED:-1337}.log
