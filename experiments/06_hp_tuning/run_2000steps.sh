#!/bin/bash
# Exp06: HP tuning (WD 0.04->0.085, QK_GAIN 1.5->4.0, warmdown 300->1440), otherwise identical to exp05
cd "$(dirname "$0")/../.." || exit 1
PYTHONUNBUFFERED=1 \
RUN_ID=exp06_hp_tuning_2000steps \
SEED=${SEED:-1337} \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MLP_MULT=3 \
MUON_WEIGHT_DECAY=0.085 \
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
  2>&1 | tee experiments/06_hp_tuning/train_2000steps_seed${SEED:-1337}.log
