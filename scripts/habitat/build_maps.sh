#!/usr/bin/env bash
# Rebuild the landmark maps the landmark / landmark_vlm / vlm_map modes read (evals/<ds>/maps has the
# ones the reported runs used -- only needed for new scenes or a new annotator model):
#   1. render each scene's walkthrough tour (goat_env.py tour, habitat env) -> $TOURS/<scene>/frames
#   2. annotate it with the VLM (robo_nav.landmark.annotate, .venv-loc)    -> <maps>/<scene>/landmark_map.md
#   3. floor labels from the tour poses (robo_nav.habitat.floors)          -> <maps>/<scene>/floors.json
# Skips scenes that already have frames / a map. Annotation costs API money (~one call per 30 keyframes).
#
#   bash scripts/habitat/build_maps.sh <goat|rxr> [--split S] [--maps DIR] [--scenes "id ..."] [--jobs N]
#
# Defaults: --split val_seen for goat (the GOAT val splits share their HM3D scenes), val_unseen for rxr;
# --maps sim_data/<ds>/maps (copy into evals/ to keep). Env as in run_eval.sh (ROBO_NAV_VLM,
# ROBO_NAV_GEMINI_MODEL -- the shipped maps used gemini-4-argon -- API keys, ROBO_NAV_ENV_FILE).
# Logs: sim_data/<ds>/maps_logs/
set -uo pipefail

ROOT=${ROOT:-$HOME/robonav}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
MM=${MICROMAMBA:-$(command -v micromamba || echo "$HOME/.local/bin/micromamba")}
export ROBO_NAV_DATA=${ROBO_NAV_DATA:-$ROOT/data}
PY=$REPO/.venv-loc/bin/python

DS=${1:?usage: build_maps.sh <goat|rxr> [options]}; shift
case $DS in goat) SPLIT=val_seen; TOURS=$ROBO_NAV_DATA/goat_tours; EXTRA=(--navmesh_cell_height 0.05);;
            rxr) SPLIT=val_unseen; TOURS=$ROBO_NAV_DATA/vlnce_tours; EXTRA=(--dataset rxr);;
            *) echo "dataset must be goat or rxr"; exit 2;; esac
MAPS=sim_data/$DS/maps; SCENES=; JOBS=3
while [ $# -gt 0 ]; do
  case $1 in --split) SPLIT=$2;; --maps) MAPS=$2;; --scenes) SCENES=$2;; --jobs) JOBS=$2;;
             *) echo "unknown option $1"; exit 2;; esac
  shift 2
done
[ -n "${ROBO_NAV_ENV_FILE:-}" ] && { set -a; source "$ROBO_NAV_ENV_FILE"; set +a; }
[ -n "${GEMINI_API_KEY:-}" ] && unset GOOGLE_API_KEY
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"; LOGS=sim_data/$DS/maps_logs; mkdir -p "$LOGS" "$MAPS" "$TOURS"
log() { echo "[$(date '+%F %T')] $*"; }

has_frames() { [ -n "$(ls -A "$TOURS/$1/frames" 2>/dev/null)" ]; }
TODO=()
if [ -n "$SCENES" ]; then TODO=($SCENES); else
  # all scenes of the split; goat_env tour with no --scenes does the same, but we want to skip done ones
  mapfile -t ALL < <("$MM" run -n habitat python -I -c "
import sys; sys.path.insert(0, 'robo_nav/habitat'); import goat_env as g, os
eps = g.load_episodes('$SPLIT') if '$DS' == 'goat' else g.load_vlnce('$DS', '$SPLIT')
print('\n'.join(sorted({os.path.basename(os.path.normpath(e['scene_id'])).split('.')[0] for e in eps})))")
  TODO=("${ALL[@]}")
fi
NEED=(); for s in "${TODO[@]}"; do has_frames "$s" || NEED+=("$s"); done
log "${#TODO[@]} scenes, ${#NEED[@]} need tours"
if [ ${#NEED[@]} -gt 0 ]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$MM" run -n habitat python -I robo_nav/habitat/goat_env.py tour \
    --split "$SPLIT" "${EXTRA[@]}" --out "$TOURS" --scenes "${NEED[@]}" > "$LOGS/tours.log" 2>&1
  log "tours exit $? (log $LOGS/tours.log)"
fi

annotate() {
  local s=$1
  for try in 1 2 3; do
    "$PY" -m robo_nav.landmark.annotate --frames "$TOURS/$s/frames" --every 10 --out "$MAPS/$s" >> "$LOGS/$s.log" 2>&1
    [ -s "$MAPS/$s/landmark_map.md" ] && { log "map $s ok (try $try)"; return; }
    log "map $s failed try $try, see $LOGS/$s.log"; sleep 120   # usually 503 / rate limits
  done
}
for s in "${TODO[@]}"; do
  [ -s "$MAPS/$s/landmark_map.md" ] && continue
  has_frames "$s" || { log "map $s: no tour frames, skipped"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 10; done
  annotate "$s" &
done
wait
"$PY" -m robo_nav.habitat.floors --maps "$MAPS" --tours "$TOURS" > "$LOGS/floors.log" 2>&1 \
  && log "floors.json written" || log "floors failed, see $LOGS/floors.log"
log "done: $(ls "$MAPS"/*/landmark_map.md 2>/dev/null | wc -l) maps in $MAPS"
