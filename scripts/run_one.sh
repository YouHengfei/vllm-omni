#!/usr/bin/env bash
# Full-dataset single backend+locale on one idle GPU: B1→B4.
# Use for parallel A/B across GPUs (e.g. Omni-EN gpu0, Omni-ZH gpu2, HF-EN gpu3, HF-ZH gpu4).
# Run from repo root. Env: .venv-test (Omni/eval), .venv-test-hf (HF gen).
# Usage: local_tests/vibevoice_pr1/run_one.sh <gpu> <backend:omni|hf> <locale:en|zh> [items] [dataset_dir]
set -euo pipefail
GPU=$1; BACKEND=$2; LOCALE=$3; ITEMS=${4:-0}; DATASET=${5:-}
cd "$(dirname "$0")/../.."  # repo root
export HF_HOME=/SharedData/youhf/cache/huggingface
export SEED_TTS_SIM_EVAL=1
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=$GPU
MODEL=/SharedData/youhf/models/vibevoice_VibeVoice-1.5B-hf
OMNI=.venv-test/bin/python
HF=.venv-test-hf/bin/python
[ -z "$DATASET" ] && DATASET=/SharedData/youhf/dataset/Seed-TTS-Eval/$LOCALE
[ "$ITEMS" -eq 0 ] && ITEMS=$(wc -l < "$DATASET/meta.lst")
TS=$(date +%Y%m%d-%H%M%S)
BASE=/SharedData/youhf/tmp/vv-idle-${LOCALE}-${BACKEND}-gpu${GPU}-${TS}
mkdir -p "$BASE"
echo "=== ${BACKEND} ${LOCALE} full on GPU$GPU, $ITEMS items, log $BASE ===" | tee -a "$BASE/run.log"

run_omni() {
  local b=$1
  echo "[$(date +%H:%M:%S)] Omni B$b start" | tee -a "$BASE/run.log"
  $OMNI -B local_tests/vibevoice_pr1/eval_snapshot_seedtts.py \
    --model "$MODEL" --dataset "$DATASET" \
    --items "$ITEMS" --batch-size "$b" --rounds 1 --warmup 1 --max-tokens 2048 \
    --output "$BASE/omni-b$b" >>"$BASE/omni-b$b.log" 2>&1
  echo "[$(date +%H:%M:%S)] Omni B$b done" | tee -a "$BASE/run.log"
}
run_hf() {
  local b=$1
  echo "[$(date +%H:%M:%S)] HF B$b gen start" | tee -a "$BASE/run.log"
  $HF -B local_tests/vibevoice_pr1/gen_hf_seedtts.py \
    --model "$MODEL" --dataset "$DATASET" \
    --items "$ITEMS" --batch-size "$b" --warmup 1 --max-tokens 2048 \
    --output "$BASE/hf-b$b-wavs" >>"$BASE/hf-b$b-gen.log" 2>&1
  echo "[$(date +%H:%M:%S)] HF B$b eval start" | tee -a "$BASE/run.log"
  $OMNI -B local_tests/vibevoice_pr1/eval_snapshot_seedtts.py \
    --dataset "$DATASET" --items "$ITEMS" --seed 0 \
    --eval-wavs "$BASE/hf-b$b-wavs" \
    --output "$BASE/hf-b$b" >>"$BASE/hf-b$b-eval.log" 2>&1
  echo "[$(date +%H:%M:%S)] HF B$b done" | tee -a "$BASE/run.log"
}

if [ "$BACKEND" = "omni" ]; then run_omni 1; run_omni 4; else run_hf 1; run_hf 4; fi
echo "[$(date +%H:%M:%S)] ALL DONE ${BACKEND} ${LOCALE} gpu$GPU -> $BASE" | tee -a "$BASE/run.log"
