#!/bin/bash
# Chains Exp 07A then 07B sequentially. Used inside a tmux session so the runs
# survive disconnects. Exits non-zero if 07A fails so 07B is skipped.
set -e
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
echo "[exp07] starting 07A WD=0.085 at $(date -Iseconds)"
bash experiments/07_sp4096_mlp4x/run_2000steps_wd085.sh
echo "[exp07] 07A done at $(date -Iseconds); starting 07B WD=0.04"
bash experiments/07_sp4096_mlp4x/run_2000steps_wd04.sh
echo "[exp07] 07B done at $(date -Iseconds)"
