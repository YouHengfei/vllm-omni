#!/usr/bin/env bash
# Pinned full-dataset re-test on one GPU. Auto-records manifest, saves both-end WAVs,
# records natural-stop / ASR-failure separately. No tuning changes.
# Usage: run_pinned.sh <gpu> <backend:locale> [<backend:locale> ...]   (each runs B1->B4 serially)
set -uo pipefail   # no -e: keep going and record status per job
cd "$(dirname "$0")/../.."  # repo root
GPU=$1; shift
export HF_HOME=/SharedData/youhf/cache/huggingface
export SEED_TTS_SIM_EVAL=1
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=$GPU
export PATH="$PWD/.venv-test/bin:$PATH"  # engine-core subprocess needs bare ninja/python
MODEL=/SharedData/youhf/models/vibevoice_VibeVoice-1.5B-hf
OMNI=.venv-test/bin/python
HF=.venv-test-hf/bin/python
TS=$(date +%Y%m%d-%H%M%S)
BASE=/SharedData/youhf/tmp/vv-pinned-gpu${GPU}-${TS}
mkdir -p "$BASE"

# --- provenance (pinned versions) ---
HEAD=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain -uno | wc -l)
GPUINFO=$(nvidia-smi --query-gpu=index,uuid,name,driver_version --format=csv,noheader | grep "^$GPU,")
EVAL_SNAP=$(sha256sum local_tests/vibevoice_pr1/eval_snapshot_seedtts.py | cut -d' ' -f1)
GENHF_SNAP=$(sha256sum local_tests/vibevoice_pr1/gen_hf_seedtts.py | cut -d' ' -f1)
RUNNER_SNAP=$(sha256sum local_tests/vibevoice_pr1/run_pinned.sh | cut -d' ' -f1)
CKPT_CFG=$(sha256sum /SharedData/youhf/models/vibevoice_VibeVoice-1.5B-hf/config.json | cut -d' ' -f1)
EN_META=$(sha256sum /SharedData/youhf/dataset/Seed-TTS-Eval/en/meta.lst | cut -d' ' -f1)
ZH_META=$(sha256sum /SharedData/youhf/dataset/Seed-TTS-Eval/zh/meta.lst | cut -d' ' -f1)
python3 - "$BASE/provenance.json" "$HEAD" "$DIRTY" "$GPUINFO" "$EVAL_SNAP" "$GENHF_SNAP" "$RUNNER_SNAP" "$CKPT_CFG" "$EN_META" "$ZH_META" <<'PY'
import json,sys,os
p,head,dirty,gpu,es,gs,rs,cfg,enm,zhm=sys.argv[1:11]
idx,uuid,name,drv=[x.strip() for x in gpu.split(',')]
json.dump({
  "source_head": head, "tracked_dirty_files": int(dirty), "branch": "feat/vibevoice-eager",
  "merge_base_upstream": os.popen("git merge-base HEAD upstream/main").read().strip(),
  "scripts": {"eval_snapshot_seedtts.py": es, "gen_hf_seedtts.py": gs, "run_pinned.sh": rs},
  "checkpoint_config_sha256": cfg,
  "dataset_meta_sha256": {"en": enm, "zh": zhm},
  "gpu": {"index": idx, "uuid": uuid, "name": name, "driver": drv},
  "env": {"HF_HOME": os.environ["HF_HOME"], "SEED_TTS_SIM_EVAL": "1", "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": str(idx)},
  "generation_params": {"max_tokens":2048,"warmup":1,"rounds":1,"seed":0,"guidance_scale":1.3,"num_diffusion_steps":10,"dtype":"bfloat16","sampling":"greedy"},
  "timing_boundary": "both backends: prompt-proc + generate + sync + D2H; eval after engine close"
}, open(p,"w"), indent=2)
PY
.venv-test/bin/python -m pip freeze 2>/dev/null | sort > "$BASE/freeze_omni.txt"
HTTPS_PROXY=http://127.0.0.1:17892 uv --system-certs pip list --python .venv-test-hf/bin/python 2>/dev/null | sort > "$BASE/freeze_hf.txt"

run_job() {
  local backend=$1 locale=$2 b=$3
  local tag="${locale}_${backend}_b${b}"
  local t0=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$(date +%H:%M:%S)] $tag start" | tee -a "$BASE/run.log"
  if [ "$backend" = "omni" ]; then
    $OMNI -B local_tests/vibevoice_pr1/eval_snapshot_seedtts.py \
      --model "$MODEL" --dataset "/SharedData/youhf/dataset/Seed-TTS-Eval/$locale" \
      --items $([ "$locale" = en ] && echo 1088 || echo 2020) --batch-size "$b" \
      --rounds 1 --warmup 1 --max-tokens 2048 --save-wavs "$BASE/wavs_${tag}" \
      --output "$BASE/${tag}" >>"$BASE/${tag}.log" 2>&1
  else
    $HF -B local_tests/vibevoice_pr1/gen_hf_seedtts.py \
      --model "$MODEL" --dataset "/SharedData/youhf/dataset/Seed-TTS-Eval/$locale" \
      --items $([ "$locale" = en ] && echo 1088 || echo 2020) --batch-size "$b" \
      --warmup 1 --max-tokens 2048 --output "$BASE/wavs_${tag}" >>"$BASE/${tag}_gen.log" 2>&1
    $OMNI -B local_tests/vibevoice_pr1/eval_snapshot_seedtts.py \
      --dataset "/SharedData/youhf/dataset/Seed-TTS-Eval/$locale" \
      --items $([ "$locale" = en ] && echo 1088 || echo 2020) --seed 0 \
      --eval-wavs "$BASE/wavs_${tag}" --output "$BASE/${tag}" >>"$BASE/${tag}_eval.log" 2>&1
  fi
  local rc=$?
  local t1=$(date '+%Y-%m-%d %H:%M:%S')
  local rpt="$BASE/${tag}/report.json"
  python3 - "$BASE/${tag}.meta.json" "$tag" "$backend" "$locale" "$b" "$t0" "$t1" "$rc" "$rpt" <<'PY'
import json,sys,os
tag,backend,locale,b,t0,t1,rc,rpt=sys.argv[1:9]
m={"tag":tag,"backend":backend,"locale":locale,"batch":int(b),"window":[t0,t1],"exit":int(rc)}
if os.path.isfile(rpt):
    r=json.load(open(rpt)); mm=r.get("rounds_metrics",[r])[0] if "rounds_metrics" in r else r
    m.update(report_sha256=__import__("hashlib").sha256(open(rpt,'rb').read()).hexdigest(),
             wer=mm.get("seed_tts_content_error_mean"), sim=mm.get("seed_tts_sim_mean"),
             n_eval=mm.get("seed_tts_content_evaluated"), asr_failed=mm.get("seed_tts_asr_failed"),
             natural_stop=r.get("natural_stop_rate"), rtf=r.get("aggregate_rtf"))
else: m["report"]="missing"
json.dump(m,open(sys.argv[1],"w"),indent=2)
PY
  echo "[$(date +%H:%M:%S)] $tag done rc=$rc" | tee -a "$BASE/run.log"
}

for task in "$@"; do
  backend=${task%%:*}; locale=${task##*:}
  run_job "$backend" "$locale" 1
  run_job "$backend" "$locale" 4
done

# assemble manifest
python3 - "$BASE" <<'PY'
import json,sys,glob,os
base=sys.argv[1]; prov=json.load(open(f"{base}/provenance.json"))
runs=[json.load(open(f)) for f in sorted(glob.glob(f"{base}/*.meta.json"))]
json.dump({"run_id":os.path.basename(base),"provenance":prov,"runs":runs,
           "status":"all succeeded" if all(r.get("exit",1)==0 and "report_sha256" in r for r in runs) else "check failures"},
          open(f"{base}/manifest.json","w"),indent=2,ensure_ascii=False)
print(f"manifest -> {base}/manifest.json  runs={len(runs)}")
PY
echo "[$(date +%H:%M:%S)] ALL DONE gpu$GPU -> $BASE" | tee -a "$BASE/run.log"
