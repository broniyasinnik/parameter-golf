#!/bin/bash
# Build fineweb10B_sp4096 from HuggingFace docs (CPU-heavy, hours). Prereq for Exp 7 runs.
# Does not modify root data/ until the export succeeds; then copies into data/datasets and data/tokenizers.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
OUT="${OUT:-$REPO_ROOT/data/_sp4096_build_wip}"
SPEC="$REPO_ROOT/data/tokenizer_specs_sp4096.json"
mkdir -p "$OUT"
export MATCHED_FINEWEB_TOKENIZER_THREADS="${MATCHED_FINEWEB_TOKENIZER_THREADS:-8}"
export MATCHED_FINEWEB_SP_BATCH_SIZE="${MATCHED_FINEWEB_SP_BATCH_SIZE:-2048}"

uv run python3 data/download_hf_docs_and_tokenize.py \
  --repo-id willdepueoai/parameter-golf \
  --remote-root datasets \
  --output-root "$OUT" \
  --tokenizer-config "$SPEC"

DS_SRC="$OUT/datasets/fineweb10B_sp4096"
TOK_MOD="$OUT/tokenizers/fineweb_4096_bpe.model"
TOK_VOC="$OUT/tokenizers/fineweb_4096_bpe.vocab"
if [[ ! -d "$DS_SRC" || ! -f "$TOK_MOD" ]]; then
  echo "Build failed: missing $DS_SRC or $TOK_MOD" >&2
  exit 1
fi

mkdir -p data/datasets data/tokenizers
rm -rf data/datasets/fineweb10B_sp4096
cp -a "$DS_SRC" data/datasets/fineweb10B_sp4096
cp -a "$TOK_MOD" data/tokenizers/
if [[ -f "$TOK_VOC" ]]; then cp -a "$TOK_VOC" data/tokenizers/; fi
echo "SP4096 dataset and tokenizer installed under data/"
