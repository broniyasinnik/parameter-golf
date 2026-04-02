#!/bin/bash
PYTHONUNBUFFERED=1 \
RUN_ID=exp02_int6_11L_mlp3x \
SEED=${SEED:-1337} \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MLP_MULT=3 \
MUON_WEIGHT_DECAY=0.04 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=32 \
uv run python3 train_gpt.py \
  2>&1 | tee experiments/02_int6_11L_mlp3x/train_seed${SEED:-1337}.log
