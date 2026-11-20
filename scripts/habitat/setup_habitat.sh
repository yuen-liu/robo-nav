#!/usr/bin/env bash
# One-time setup for the Habitat evals (GOAT-Bench on HM3D, RxR-CE / R2R-CE on MP3D).
# Run scripts/setup_cluster.sh first (it builds LightNav-0/.venv, which runs the controller).
#
#   bash scripts/habitat/setup_habitat.sh <step> [<step> ...]
#
# Steps:
#   env     micromamba env `habitat` (python 3.7, habitat-sim 0.2.3 headless) for goat_env.py, plus the
#           controller's extra deps (anthropic, google-genai, pyzmq) in LightNav-0/.venv
#   hm3d    HM3D v0.2 val scenes + semantics        needs MATTERPORT_TOKEN_ID / MATTERPORT_TOKEN_SECRET
#   goat    GOAT-Bench episodes (hm3d/v1)            Google Drive, via gdown
#   mp3d    Matterport3D habitat scenes              needs MP_DOWNLOAD=/path/to/download_mp.py (personal),
#                                                    or MP3D_ZIP=/path/to/mp3d_habitat.zip already downloaded
#   rxr     RxR-CE episodes (RxR_VLNCE_v0)           Google Drive, via gdown
#   r2r     R2R-CE episodes (R2R_VLNCE_v1-3_preprocessed)
#   check   verify the layout + a habitat-sim smoke render
#   goat-all = env hm3d goat check      rxr-all = env mp3d rxr check
#
# Env vars: ROOT (default ~/robonav; LightNav-0 + robo-nav live here), DATA (default $ROOT/data, the
# path goat_env.py reads), MAMBA_ROOT_PREFIX (default ~/micromamba).
#
# Licenses: HM3D and MP3D are gated by Matterport's terms -- request access first:
#   HM3D: https://matterport.com/habitat-matterport-3d-research-dataset (then create an API token
#         at my.matterport.com/settings/account/devtools -> MATTERPORT_TOKEN_ID / _SECRET)
#   MP3D: https://niessner.github.io/Matterport/ (sign the ToU, you get download_mp.py by email;
#         never commit it -- it is in .gitignore)
set -euo pipefail

ROOT=${ROOT:-$HOME/robonav}
DATA=${DATA:-$ROOT/data}
export ROBO_NAV_DATA=$DATA  # read by goat_env.py / smoke.py
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LV=$ROOT/LightNav-0/.venv/bin/python
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
MM=${MICROMAMBA:-$(command -v micromamba || echo "$HOME/.local/bin/micromamba")}
DL=$DATA/_dl

log() { echo "== $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
hab() { "$MM" run -n habitat "$@"; }

gdown_() {  # gdown <file id> <out>, installed into LightNav-0's venv on first use
  [ -x "$LV" ] || die "no $LV -- run scripts/setup_cluster.sh first"
  "$LV" -c "import gdown" 2>/dev/null || uv pip install -q --python "$LV" gdown
  "$LV" -m gdown "$1" -O "$2"
}

# unzip <zip> and move the directory that contains <marker> (a relative path) to <dest>
unzip_to() {
  local zip=$1 marker=$2 dest=$3 tmp
  tmp=$(mktemp -d "$DL/unzip.XXXX")
  unzip -q "$zip" -d "$tmp"
  local hit; hit=$(find "$tmp" -path "*/$marker" -print -quit)
  [ -n "$hit" ] || die "$zip has no $marker"
  local top=${hit%/"$marker"}
  mkdir -p "$(dirname "$dest")"
  [ -e "$dest" ] && die "$dest already exists (delete it to re-extract)"
  mv "$top" "$dest"
  rm -rf "$tmp"
}

step_env() {
  if [ ! -x "$MM" ]; then
    log "installing micromamba to ~/.local/bin"
    mkdir -p "$HOME/.local/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME/.local" bin/micromamba
    MM=$HOME/.local/bin/micromamba
  fi
  if "$MM" env list | grep -qE '^\s*habitat\s'; then
    log "micromamba env habitat exists"
  else
    log "creating micromamba env habitat"
    # The combination the published runs used. goat_env.py only needs habitat-sim, numpy, pyzmq, pillow.
    "$MM" create -y -n habitat -c aihabitat -c conda-forge \
      python=3.7 habitat-sim=0.2.3 headless pillow pyzmq pip
  fi
  hab python -c "import habitat_sim, zmq; print('habitat-sim', habitat_sim.__version__)"

  log "controller deps in LightNav-0/.venv"
  [ -x "$LV" ] || die "no $LV -- run scripts/setup_cluster.sh first"
  uv pip install -q --python "$LV" anthropic google-genai pyzmq
}

step_hm3d() {
  [ -n "${MATTERPORT_TOKEN_ID:-}" ] && [ -n "${MATTERPORT_TOKEN_SECRET:-}" ] \
    || die "set MATTERPORT_TOKEN_ID and MATTERPORT_TOKEN_SECRET (Matterport API token, see header)"
  if [ -f "$DATA/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json" ]; then
    log "HM3D already at $DATA/scene_datasets/hm3d"; return
  fi
  log "HM3D v0.2 val (habitat + configs + semantics, ~10 GB)"
  # Lands in $DATA/versioned_data/ with a symlink at $DATA/scene_datasets/hm3d.
  hab python -m habitat_sim.utils.datasets_download --data-path "$DATA/" --uids hm3d_val_v0.2 \
    --username "$MATTERPORT_TOKEN_ID" --password "$MATTERPORT_TOKEN_SECRET"
}

step_goat() {
  local dest=$DATA/datasets/goat_bench/hm3d/v1
  [ -d "$dest/val_unseen/content" ] && { log "GOAT-Bench already at $dest"; return; }
  log "GOAT-Bench episodes (github.com/Ram81/goat-bench)"
  mkdir -p "$DL/goat"
  [ -s "$DL/goat/goat_bench.zip" ] || gdown_ 1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe "$DL/goat/goat_bench.zip"
  unzip_to "$DL/goat/goat_bench.zip" val_unseen/val_unseen.json.gz "$dest"
}

step_mp3d() {
  [ -f "$DATA/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb" ] && { log "MP3D already at $DATA/scene_datasets/mp3d"; return; }
  local zip=${MP3D_ZIP:-}
  if [ -z "$zip" ]; then
    [ -f "${MP_DOWNLOAD:-}" ] || die "set MP_DOWNLOAD=/path/to/download_mp.py or MP3D_ZIP=/path/to/mp3d_habitat.zip (see header)"
    log "MP3D habitat scenes (~15 GB)"
    mkdir -p "$DL/mp3d"
    zip=$(find "$DL/mp3d" -name mp3d_habitat.zip -size +1G -print -quit)
    if [ -z "$zip" ]; then
      # The official script is python 2.7 and asks to accept the ToU on stdin (MP_PYTHON to override).
      echo | "${MP_PYTHON:-python2}" "$MP_DOWNLOAD" --task habitat -o "$DL/mp3d"
      zip=$(find "$DL/mp3d" -name mp3d_habitat.zip -print -quit)
    fi
  fi
  [ -s "$zip" ] || die "no mp3d_habitat.zip"
  unzip_to "$zip" 17DRP5sb8fy/17DRP5sb8fy.glb "$DATA/scene_datasets/mp3d"
  [ "$(ls "$DATA/scene_datasets/mp3d" | wc -l)" -ge 90 ] || echo "WARNING: expected 90 MP3D scenes"
}

step_rxr() {
  local dest=$DATA/datasets/RxR_VLNCE_v0
  [ -f "$dest/val_unseen/val_unseen_guide.json.gz" ] && { log "RxR-CE already at $dest"; return; }
  log "RxR-CE episodes (github.com/jacobkrantz/VLN-CE)"
  mkdir -p "$DL/rxr"
  [ -s "$DL/rxr/RxR_VLNCE_v0.zip" ] || gdown_ 145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t "$DL/rxr/RxR_VLNCE_v0.zip"
  unzip_to "$DL/rxr/RxR_VLNCE_v0.zip" val_unseen/val_unseen_guide.json.gz "$dest"
}

step_r2r() {
  local dest=$DATA/datasets/R2R_VLNCE_v1-3_preprocessed
  [ -f "$dest/val_unseen/val_unseen.json.gz" ] && { log "R2R-CE already at $dest"; return; }
  log "R2R-CE episodes (github.com/jacobkrantz/VLN-CE)"
  mkdir -p "$DL/r2r"
  [ -s "$DL/r2r/R2R_VLNCE_v1-3_preprocessed.zip" ] \
    || gdown_ 1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr "$DL/r2r/R2R_VLNCE_v1-3_preprocessed.zip"
  unzip_to "$DL/r2r/R2R_VLNCE_v1-3_preprocessed.zip" val_unseen/val_unseen.json.gz "$dest"
}

step_check() {
  log "layout under $DATA"
  local ok=1
  chk() { if [ -e "$DATA/$1" ]; then echo "  ok       $1"; else echo "  missing  $1  ($2)"; ok=0; fi; }
  chk scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json "hm3d: GOAT"
  chk datasets/goat_bench/hm3d/v1/val_unseen/content "goat: GOAT"
  chk scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb "mp3d: RxR/R2R"
  chk datasets/RxR_VLNCE_v0/val_unseen/val_unseen_guide.json.gz "rxr"
  chk datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz "r2r (optional)"
  local ep; ep=$(ls "$DATA"/datasets/goat_bench/hm3d/v1/val_seen/content/*.json.gz 2>/dev/null | head -1 || true)
  if [ -n "$ep" ] && [ -e "$DATA/scene_datasets/hm3d" ]; then
    log "habitat-sim smoke render (GPU ${CUDA_VISIBLE_DEVICES:-0})"
    hab python -I "$REPO/scripts/habitat/smoke.py" "$ep" "$DL/smoke"
    echo "  frames in $DL/smoke"
  fi
  [ $ok = 1 ] || echo "(missing entries are only a problem for the benchmark named after them)"
}

[ $# -gt 0 ] || { sed -n '2,26p' "$0"; exit 1; }
mkdir -p "$DL"
for s in "$@"; do
  case $s in
    goat-all) step_env; step_hm3d; step_goat; step_check;;
    rxr-all) step_env; step_mp3d; step_rxr; step_check;;
    env|hm3d|goat|mp3d|rxr|r2r|check) "step_$s";;
    *) die "unknown step $s";;
  esac
done
