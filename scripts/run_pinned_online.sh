#!/usr/bin/env bash
# Standard pinned online + HF VibeVoice Seed-TTS-Eval run on one free GPU.
# Follows notes/vibevoice_test_provenance.md: pin -> run -> append -> seal.
#
# Per locale runs two backends under the SAME config (CFG 1.3, 10 steps, BF16,
# greedy, max_tokens 2048, seed 0, first N items):
#   * online : eval_online_seedtts.py (nonstreaming WAV + streaming PCM,
#               WER/SIM/RTF/TTFP via /v1/audio/speech)
#   * hf     : gen_hf_seedtts.py (HF generate, WER/SIM/RTF; no TTFP — HF does
#               not stream audio, see README) + eval_snapshot_seedtts.py --eval-wavs
#
#   - writes results/<run_id>/manifest.json with status=pending + pinned inputs
#     (git head, script SHA256, checkpoint/dataset SHA, env freeze, GPU start
#      snapshot, params, planned_jobs) BEFORE generation;
#   - appends outputs read from report.json / gen log (never hand-typed);
#   - captures GPU end snapshot, flags contaminated per job;
#   - seals status=complete.
#
# Usage:
#   run_pinned_online.sh <gpu> <items_per_locale> [en|zh|...]
# Example (100-item en with HF baseline):
#   CUDA_VISIBLE_DEVICES=4 ./local_tests/vibevoice_pr2/run_pinned_online.sh 4 100 en
set -uo pipefail   # no -e: keep going, record status per job
cd "$(dirname "$0")/../.."   # repo root

GPU=${1:?usage: run_pinned_online.sh <gpu> <items> [locale ...]}
ITEMS=${2:?items per locale}
shift 2
LOCALES=("$@")
[ ${#LOCALES[@]} -eq 0 ] && LOCALES=(en zh)

export HF_HOME=/SharedData/youhf/cache/huggingface
export VLLM_USE_FLASHINFER_SAMPLER=0
export SEED_TTS_SIM_EVAL=1
export SEED_TTS_SIM_DEVICE=cpu
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$PWD"  # tests.helpers.runtime import in eval scripts
export PATH="$PWD/.venv-test/bin:$PWD/.venv/bin:$PATH"  # engine-core needs ninja
unset HTTPS_PROXY HTTP_PROXY https_proxy http_proxy ALL_PROXY all_proxy

MODEL=vibevoice/VibeVoice-1.5B-hf
HF_MODEL=/SharedData/youhf/models/vibevoice_VibeVoice-1.5B-hf
ONLINE_SCRIPT=local_tests/vibevoice_pr2/eval_online_seedtts.py
GEN_HF_SCRIPT=local_tests/vibevoice_pr1/gen_hf_seedtts.py
EVAL_HF_SCRIPT=local_tests/vibevoice_pr1/eval_snapshot_seedtts.py
PY=.venv-test/bin/python          # omni env: serving + seed-tts eval (WER/SIM)
HF_PY=.venv-test-hf/bin/python    # HF env: transformers 5.19 generate only
TS=$(date +%Y%m%d-%H%M%S)
RUN_ID="vv-online-gpu${GPU}-${TS}"
BASE=local_tests/vibevoice_pr2/results/${RUN_ID}
mkdir -p "$BASE"

DS=/SharedData/youhf/dataset/Seed-TTS-Eval

# ---------- 1. pin inputs (status=pending) ----------
HEAD=$(git rev-parse HEAD)
DIRTY=$(git status --porcelain -uno | wc -l)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
MERGE_BASE=$(git merge-base HEAD upstream/main 2>/dev/null || echo "")
ONLINE_SHA=$(sha256sum "$ONLINE_SCRIPT" | cut -d' ' -f1)
GENHF_SHA=$(sha256sum "$GEN_HF_SCRIPT" | cut -d' ' -f1)
EVALHF_SHA=$(sha256sum "$EVAL_HF_SCRIPT" | cut -d' ' -f1)
RUNNER_SHA=$(sha256sum local_tests/vibevoice_pr2/run_pinned_online.sh | cut -d' ' -f1)
CKPT_CFG=$(sha256sum "$HF_MODEL/config.json" 2>/dev/null | cut -d' ' -f1 || echo "")
EN_META=$(sha256sum "$DS/en/meta.lst" | cut -d' ' -f1)
ZH_META=$(sha256sum "$DS/zh/meta.lst" | cut -d' ' -f1)

nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu \
  --format=csv,noheader -i "$GPU" > "$BASE/gpu_start_snapshot.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
  > "$BASE/gpu_start_compute_apps.csv" 2>/dev/null || true

$PY -m pip freeze 2>/dev/null | sort > "$BASE/freeze_omni.txt"
$HF_PY -m pip freeze 2>/dev/null | sort > "$BASE/freeze_hf.txt"

# planned jobs: online + hf per locale
declare -a PLANNED
i=0
for loc in "${LOCALES[@]}"; do
  PLANNED[i++]="{\"gpu\":$GPU,\"locale\":\"$loc\",\"items\":$ITEMS,\"backend\":\"online\",\"modes\":[\"nonstreaming\",\"streaming\"]}"
  PLANNED[i++]="{\"gpu\":$GPU,\"locale\":\"$loc\",\"items\":$ITEMS,\"backend\":\"hf\",\"modes\":[\"generate+eval\"]}"
done
JOBS_CSV=$(IFS=,; echo "${PLANNED[*]}")

$PY - "$BASE/manifest.json" <<PYEOF
import json, os
base="$BASE"
with open(f"{base}/freeze_omni.txt") as f:
    omni=[l.strip() for l in f if l.strip()]
with open(f"{base}/freeze_hf.txt") as f:
    hf=[l.strip() for l in f if l.strip()]
manifest={
  "run_id":"$RUN_ID",
  "status":"pending",
  "standard_process":"pin->run->append->seal; no hand-edit after seal",
  "inputs":{
    "source":{"head":"$HEAD","branch":"$BRANCH","tracked_dirty_files":int("$DIRTY"),
               "merge_base_upstream":"$MERGE_BASE"},
    "scripts":{"eval_online_seedtts.py":{"sha256":"$ONLINE_SHA"},
               "gen_hf_seedtts.py":{"sha256":"$GENHF_SHA"},
               "eval_snapshot_seedtts.py":{"sha256":"$EVALHF_SHA"},
               "run_pinned_online.sh":{"sha256":"$RUNNER_SHA"}},
    "checkpoint":{"model_id":"$MODEL","hf_path":"$HF_MODEL","config_json_sha256":"$CKPT_CFG"},
    "dataset":{"root":"$DS",
              "en":{"meta_lst_sha256":"$EN_META"},
              "zh":{"meta_lst_sha256":"$ZH_META"},
              "selection":"SeedTTSDataset random_seed=0 first N rows (same for online+hf)"},
    "environments":{
      "omni":{"freeze_file":"freeze_omni.txt","sample":omni[:8]},
      "hf":{"freeze_file":"freeze_hf.txt","sample":hf[:8]}},
    "hardware":{"startup_snapshot":"gpu_start_snapshot.csv","exclusivity_declared":True},
    "parameters":{"max_tokens":2048,"warmup":3,"seed":0,"items_per_locale":$ITEMS,
                  "guidance_scale":1.3,"num_diffusion_steps":10,"dtype":"bfloat16",
                  "sampling":"greedy","sim_device":"cpu",
                  "timing_boundary":"online: POST->full body / first chunk; hf: prompt-proc+generate+sync+D2H; eval after engine close",
                  "env_vars":{"VLLM_USE_FLASHINFER_SAMPLER":"0","SEED_TTS_SIM_EVAL":"1","SEED_TTS_SIM_DEVICE":"cpu","OMP_NUM_THREADS":"2","CUDA_VISIBLE_DEVICES":"$GPU"},
                  "hf_note":"HF generate() does not stream audio (returns at end); TTFP not measurable for hf backend"},
    "planned_jobs":[$JOBS_CSV]
  },
  "outputs":[]
}
json.dump(manifest, open(f"{base}/manifest.json","w"), indent=2, ensure_ascii=False)
print(f"pinned manifest -> {base}/manifest.json  status=pending")
PYEOF

# ---------- 2. launch + 3. append outputs ----------
append_job() {
  # append_job <tag> <backend> <locale> <items> <t0> <t1> <rc> <report_path> <apps_end>
  $PY - "$BASE/manifest.json" "$BASE/gpu_start_snapshot.csv" "$@" <<'PY'
import json,sys,os,re
manifest_path,start_snap,tag,backend,loc,items,t0,t1,rc,rpt,apps_end=sys.argv[1:12]
m=json.load(open(manifest_path))
job={"tag":tag,"backend":backend,"locale":loc,"items":int(items),"window":[t0,t1],"exit":int(rc)}
target_uuid=None
try:
    with open(start_snap) as f:
        for line in f:
            parts=line.split(',')
            if len(parts)>=2: target_uuid=parts[1].strip(); break
except OSError: pass
contaminated=False
if target_uuid:
    try:
        with open(apps_end) as f:
            for line in f:
                line=line.strip()
                if not line or 'no compute' in line.lower(): continue
                parts=line.split(',')
                if parts and parts[0].strip()==target_uuid: contaminated=True; break
    except OSError: pass
job["contaminated"]=contaminated
if os.path.isfile(rpt):
    r=json.load(open(rpt))
    job["report_path"]=rpt
    job["report_sha256"]=__import__("hashlib").sha256(open(rpt,'rb').read()).hexdigest()
    mm=r.get("rounds_metrics",[r])[0] if "rounds_metrics" in r else r
    job["wer_mean"]=mm.get("seed_tts_content_error_mean")
    job["wer_median"]=mm.get("seed_tts_content_error_median")
    job["sim_mean"]=mm.get("seed_tts_sim_mean")
    job["n_eval"]=mm.get("seed_tts_content_evaluated")
    if backend=="online":
        for mode,d in r.get("modes",{}).items():
            p=d.get("perf",{}); w=d.get("wer_sim",{})
            job.setdefault("modes",{})[mode]={
              "wer_mean":w.get("seed_tts_content_error_mean"),
              "wer_median":w.get("seed_tts_content_error_median"),
              "sim_mean":w.get("seed_tts_sim_mean"),
              "rtf":p.get("aggregate_rtf"),
              "ttfp_p50_ms":p.get("p50_ttfp_ms"),
              "ttfp_p99_ms":p.get("p99_ttfp_ms"),
              "n_eval":w.get("seed_tts_content_evaluated")}
    else:  # hf: rtf from gen log parsed by caller into rpt? no — from gen log
        job["rtf"]=r.get("aggregate_rtf")
else:
    job["report"]="missing"
m["outputs"].append(job)
json.dump(m,open(manifest_path,"w"),indent=2,ensure_ascii=False)
print(f"appended {tag} backend={backend} rc={rc} contaminated={contaminated}")
PY
}

snapshot_end() {
  nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu \
    --format=csv,noheader -i "$GPU" > "$BASE/$1_gpu_end.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
    > "$BASE/$1_compute_apps_end.csv" 2>/dev/null || true
}

run_online_job() {
  local loc=$1 items=$2
  local tag="${loc}_${items}_online"
  local t0=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$(date +%H:%M:%S)] $tag start" | tee -a "$BASE/run.log"
  $PY -B "$ONLINE_SCRIPT" --model "$MODEL" \
    --dataset "$DS/$loc" --items "$items" --warmup 3 --max-tokens 2048 \
    --output "$BASE/${tag}" >>"$BASE/${tag}.log" 2>&1
  local rc=$?
  local t1=$(date '+%Y-%m-%d %H:%M:%S')
  snapshot_end "$tag"
  append_job "$tag" online "$loc" "$items" "$t0" "$t1" "$rc" \
    "$BASE/${tag}/report.json" "$BASE/${tag}_compute_apps_end.csv"
  echo "[$(date +%H:%M:%S)] $tag done rc=$rc" | tee -a "$BASE/run.log"
}

run_hf_job() {
  local loc=$1 items=$2
  local tag="${loc}_${items}_hf"
  local wav_dir="$BASE/wavs_${tag}"
  local t0=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$(date +%H:%M:%S)] $tag start (gen+eval)" | tee -a "$BASE/run.log"
  # 1. HF generate under .venv-test-hf (transformers 5.19)
  $HF_PY -B "$GEN_HF_SCRIPT" --model "$HF_MODEL" \
    --dataset "$DS/$loc" --items "$items" --warmup 1 --max-tokens 2048 \
    --output "$wav_dir" >>"$BASE/${tag}_gen.log" 2>&1
  local gen_rc=$?
  local t1=$(date '+%Y-%m-%d %H:%M:%S')
  snapshot_end "$tag"
  # 2. eval WAVs under .venv-test (same WER/SIM harness as online)
  local eval_rc=0
  if [ $gen_rc -eq 0 ]; then
    $PY -B "$EVAL_HF_SCRIPT" --dataset "$DS/$loc" --items "$items" --seed 0 \
      --eval-wavs "$wav_dir" --output "$BASE/${tag}" >>"$BASE/${tag}_eval.log" 2>&1
    eval_rc=$?
  fi
  local rc=$(( gen_rc != 0 ? gen_rc : eval_rc ))
  # Parse RTF from gen log (HF gen prints: "HF gen done: wall=Xs audio=Ys rtf=Z ->")
  $PY - "$BASE/manifest.json" "$BASE/gpu_start_snapshot.csv" "$tag" hf "$loc" "$items" "$t0" "$t1" "$rc" "$BASE/${tag}/report.json" "$BASE/${tag}_compute_apps_end.csv" "$BASE/${tag}_gen.log" <<'PY'
import json,sys,os,re
manifest_path,start_snap,tag,backend,loc,items,t0,t1,rc,rpt,apps_end,gen_log=sys.argv[1:13]
m=json.load(open(manifest_path))
job={"tag":tag,"backend":backend,"locale":loc,"items":int(items),"window":[t0,t1],"exit":int(rc)}
target_uuid=None
try:
    with open(start_snap) as f:
        for line in f:
            parts=line.split(',')
            if len(parts)>=2: target_uuid=parts[1].strip(); break
except OSError: pass
contaminated=False
if target_uuid:
    try:
        with open(apps_end) as f:
            for line in f:
                line=line.strip()
                if not line or 'no compute' in line.lower(): continue
                parts=line.split(',')
                if parts and parts[0].strip()==target_uuid: contaminated=True; break
    except OSError: pass
job["contaminated"]=contaminated
# rtf from gen log
rtf=None; gen_wall=None; gen_audio=None; natural_stop=None
try:
    log=open(gen_log).read()
    mt=re.search(r"HF gen done: wall=([\d.]+)s audio=([\d.]+)s rtf=([\d.]+) stop=(\d+)/(\d+)",log)
    if mt:
        gen_wall=float(mt.group(1)); gen_audio=float(mt.group(2)); rtf=float(mt.group(3))
        natural_stop=int(mt.group(4))/int(mt.group(5))
except OSError: pass
job["rtf"]=rtf; job["natural_stop_rate"]=natural_stop
if os.path.isfile(rpt):
    r=json.load(open(rpt))
    job["report_path"]=rpt
    job["report_sha256"]=__import__("hashlib").sha256(open(rpt,'rb').read()).hexdigest()
    job["wer_mean"]=r.get("seed_tts_content_error_mean")
    job["wer_median"]=r.get("seed_tts_content_error_median")
    job["sim_mean"]=r.get("seed_tts_sim_mean")
    job["n_eval"]=r.get("seed_tts_content_evaluated")
else:
    job["report"]="missing"
m["outputs"].append(job)
json.dump(m,open(manifest_path,"w"),indent=2,ensure_ascii=False)
print(f"appended {tag} backend={backend} rc={rc} contaminated={contaminated} rtf={rtf}")
PY
  echo "[$(date +%H:%M:%S)] $tag done rc=$rc gen_rc=$gen_rc eval_rc=$eval_rc" | tee -a "$BASE/run.log"
}

for loc in "${LOCALES[@]}"; do
  run_online_job "$loc" "$ITEMS"
  run_hf_job "$loc" "$ITEMS"
done

# ---------- 4. seal ----------
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "$GPU" \
  > "$BASE/gpu_end_snapshot.csv"
$PY - "$BASE/manifest.json" <<'PY'
import json,sys
p=sys.argv[1]; m=json.load(open(p))
ok=all(j.get("exit",1)==0 and not j.get("contaminated") for j in m["outputs"])
contam=any(j.get("contaminated") for j in m["outputs"])
m["status"]="complete" if ok and not contam else ("complete_with_contamination" if ok else "failed")
m["gpu_end_snapshot"]="gpu_end_snapshot.csv"
m["comparison"]={}
# build online-vs-hf per-locale WER/RTF delta if both present
for loc in sorted({j["locale"] for j in m["outputs"]}):
    on=[j for j in m["outputs"] if j.get("backend")=="online" and j["locale"]==loc]
    hf=[j for j in m["outputs"] if j.get("backend")=="hf" and j["locale"]==loc]
    if on and hf:
        onj=on[0]; hfj=hf[0]
        # online: aggregate across modes (use nonstreaming for WER/SIM parity)
        ns=onj.get("modes",{}).get("nonstreaming",{})
        m["comparison"][loc]={
          "online_nonstreaming":{"wer_mean":ns.get("wer_mean"),"sim_mean":ns.get("sim_mean"),"rtf":ns.get("rtf")},
          "online_streaming":{k:v for k,v in (onj.get("modes",{}).get("streaming",{})).items()},
          "hf":{"wer_mean":hfj.get("wer_mean"),"sim_mean":hfj.get("sim_mean"),"rtf":hfj.get("rtf")},
          "wer_delta_online_minus_hf":(ns.get("wer_mean")-hfj.get("wer_mean")) if ns.get("wer_mean") is not None and hfj.get("wer_mean") is not None else None,
          "rtf_ratio_hf_over_online":(hfj.get("rtf")/ns.get("rtf")) if ns.get("rtf") and hfj.get("rtf") else None}
json.dump(m,open(p,"w"),indent=2,ensure_ascii=False)
print(f"sealed {p}  status={m['status']}  jobs={len(m['outputs'])}")
PY

echo "[$(date +%H:%M:%S)] ALL DONE gpu$GPU items=$ITEMS -> $BASE" | tee -a "$BASE/run.log"
