#!/bin/bash
PYTHONUNBUFFERED=1 \
RUN_ID=exp01_sliding_window_eval \
SEED=${SEED:-1337} \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=32 \
python3 train_gpt.py \
  2>&1 | tee experiments/01_sliding_window_eval/train_seed${SEED:-1337}.log
