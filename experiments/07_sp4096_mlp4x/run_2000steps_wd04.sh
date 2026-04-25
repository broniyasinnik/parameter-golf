#!/bin/bash
# Exp07B: SP4096 + MLP 4x, MUON WD=0.04; else same as 07A
cd "$(dirname "$0")/../.." || exit 1
PYTHONUNBUFFERED=1 \
RUN_ID=exp07B_sp4096_mlp4x_wd04 \
SEED=${SEED:-1337} \
DATA_PATH=./data/datasets/fineweb10B_sp4096/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_4096_bpe.model \
VOCAB_SIZE=4096 \
NUM_LAYERS=11 \
MLP_MULT=4 \
MUON_WEIGHT_DECAY=0.04 \
QK_GAIN_INIT=4.0 \
EMA_DECAY=0.997 \
EMA_START_FRACTION=0.5 \
XSA_LAST_N=4 \
ITERATIONS=2000 \
WARMDOWN_ITERS=1440 \
MAX_WALLCLOCK_SECONDS=0 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=32 \
uv run python3 train_gpt.py \
  2>&1 | tee experiments/07_sp4096_mlp4x/train_2000steps_07B_wd04_seed${SEED:-1337}.log
