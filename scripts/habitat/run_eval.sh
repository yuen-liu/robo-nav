#!/usr/bin/env bash
# Run a GOAT-Bench / RxR-CE / R2R-CE eval: one goat_env.py server (habitat env) + K goat_nav clients
# (LightNav-0's venv), then one resume pass for episodes that ended in planner_error (API errors).
# Rerunning the same command resumes: goat_nav skips results already in <out>.
#
#   bash scripts/habitat/run_eval.sh <goat|rxr|r2r> <out_dir> [options] [-- <goat_nav args>]
#
# What to run (one of):
#   --goals FILE          GOAT single-subtask goals with fixed starts (e.g. evals/goat/far_goals_10m.json)
#   --episode_list FILE   JSON list of {"index": i, ...} (e.g. evals/rxr/rxr_far_12m.json)
#   --episodes "0 1 2"    episode indices in the split (GOAT: whole lifelong episodes)
# Options:
#   --split S             default val_unseen
#   --clients K           goat_nav clients sharing one server (default 1; K > 1 uses serve --shared)
#   --env_gpu G           GPU for habitat-sim (default 0)
#   --nav_gpu G           GPU for LightNav-0 (vLLM, in-process; needed by zeroshot/landmark modes).
#                         Default: none -- enough for the VLM-only modes vlm / vlm_map / landmark_vlm.
#   --port P              default 5560
#   --maps DIR            landmark maps (default evals/goat/maps or evals/rxr/maps)
# Everything after -- goes to robo_nav.habitat.goat_nav (--modes, --budget_factor, --floors, ...).
#
# Env: ROOT (default ~/robonav), ROBO_NAV_DATA (default $ROOT/data), ROBO_NAV_VLM (claude|gemini),
# ROBO_NAV_GEMINI_MODEL, ANTHROPIC_API_KEY / GEMINI_API_KEY; ROBO_NAV_ENV_FILE, if set, is sourced
# first (a file of KEY=value lines, e.g. ~/.secrets/anthropic.env).
# Logs: <out>/serve_<pass>.log, <out>/client<k>_<pass>.log
set -uo pipefail

ROOT=${ROOT:-$HOME/robonav}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LV=$ROOT/LightNav-0/.venv/bin/python
CKPT=$ROOT/LightNav-0/checkpoints/LightNav-0
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
MM=${MICROMAMBA:-$(command -v micromamba || echo "$HOME/.local/bin/micromamba")}
export ROBO_NAV_DATA=${ROBO_NAV_DATA:-$ROOT/data}
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

DS=${1:?usage: run_eval.sh <goat|rxr|r2r> <out_dir> [options] [-- goat_nav args]}
OUT=${2:?usage: run_eval.sh <goat|rxr|r2r> <out_dir> [options] [-- goat_nav args]}
shift 2
case $DS in goat|rxr|r2r) ;; *) echo "dataset must be goat, rxr or r2r"; exit 2;; esac
SPLIT=val_unseen; K=1; ENV_GPU=0; NAV_GPU=; PORT=5560; GOALS=; LIST=; EPISODES=
MAPS=evals/$DS/maps; [ $DS = r2r ] && MAPS=
while [ $# -gt 0 ]; do
  case $1 in
    --goals) GOALS=$2;; --episode_list) LIST=$2;; --episodes) EPISODES=$2;;
    --split) SPLIT=$2;; --clients) K=$2;; --env_gpu) ENV_GPU=$2;; --nav_gpu) NAV_GPU=$2;;
    --port) PORT=$2;; --maps) MAPS=$2;;
    --) shift; break;;
    *) echo "unknown option $1 (goat_nav args go after --)"; exit 2;;
  esac
  shift 2
done
NAV_ARGS=("$@")
[ $(( ${GOALS:+1} + ${LIST:+1} + ${EPISODES:+1} )) = 1 ] || { echo "give exactly one of --goals / --episode_list / --episodes"; exit 2; }
[ -n "$GOALS" ] && [ $DS != goat ] && { echo "--goals is GOAT only"; exit 2; }

[ -n "${ROBO_NAV_ENV_FILE:-}" ] && { set -a; source "$ROBO_NAV_ENV_FILE"; set +a; }
# google-genai prefers GOOGLE_API_KEY when both are set; keep the one the user picked.
[ -n "${GEMINI_API_KEY:-}" ] && unset GOOGLE_API_KEY
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

if [ $DS = goat ]; then SERVE=(--split "$SPLIT" --navmesh_cell_height 0.05)   # HM3D v0.2 navmesh re-bake
else SERVE=(--dataset "$DS" --split "$SPLIT"); fi
[ "$K" -gt 1 ] && SERVE=(--shared "${SERVE[@]}")

cd "$REPO"; mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }

selection() {  # goat_nav selection args for client $1 of $K
  if [ -n "$GOALS" ]; then echo "--goals $GOALS --shard $1 $K"
  else
    local ids
    if [ -n "$LIST" ]; then ids=$("$LV" -c "import json,sys; print(' '.join(str(r['index']) for r in json.load(open(sys.argv[1]))))" "$LIST")
    else ids=$EPISODES; fi
    # same order on every client, so a shared server stays on one scene as long as possible
    echo "--episodes $(echo $ids | tr ' ' '\n' | awk -v i=$1 -v k=$K '(NR-1)%k==i' | tr '\n' ' ')"
  fi
}

run_pass() {
  local tag=$1 slog=$OUT/serve_$1.log
  : > "$slog"
  CUDA_VISIBLE_DEVICES=$ENV_GPU "$MM" run -n habitat python -I robo_nav/habitat/goat_env.py serve \
    "${SERVE[@]}" --port "$PORT" >> "$slog" 2>&1 &
  local spid=$!
  for _ in $(seq 90); do grep -q "GOAT env on" "$slog" && break; kill -0 $spid 2>/dev/null || break; sleep 2; done
  grep -q "GOAT env on" "$slog" || { log "env server not up, see $slog"; kill $spid 2>/dev/null; return 1; }
  local pids=() timeout=600
  [ "$K" -gt 1 ] && timeout=10800
  for i in $(seq 0 $((K-1))); do
    local sel; sel=$(selection $i)
    [ "$sel" = "--episodes " ] && continue
    CUDA_VISIBLE_DEVICES=$NAV_GPU "$LV" -m robo_nav.habitat.goat_nav --model_path "$CKPT" \
      --env tcp://127.0.0.1:$PORT --env_timeout_s $timeout ${MAPS:+--maps "$MAPS"} --out "$OUT" \
      $sel "${NAV_ARGS[@]}" > "$OUT/client${i}_$tag.log" 2>&1 &
    pids+=($!)
  done
  log "$tag pass: $K client(s), logs $OUT/client*_$tag.log"
  wait "${pids[@]}"
  kill $spid 2>/dev/null; wait $spid 2>/dev/null
}

n_err() { grep -l '"ended": "planner_error"' "$OUT"/*.json 2>/dev/null | wc -l; }

log "run_eval $DS $SPLIT -> $OUT (env GPU $ENV_GPU, nav GPU ${NAV_GPU:-none}, port $PORT): ${NAV_ARGS[*]}"
run_pass main || exit 1
log "main pass done: $(ls "$OUT"/*.json 2>/dev/null | wc -l) result files"
if [ "$(n_err)" -gt 0 ]; then log "$(n_err) planner_error results, resume pass"; run_pass resume; fi
log "done; planner_error left: $(n_err)"
log "report: $LV -m robo_nav.habitat.goal_report --runs $OUT"
