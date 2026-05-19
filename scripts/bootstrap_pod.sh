#!/usr/bin/env bash
# scripts/bootstrap_pod.sh <concept> [git-ref]
# Runs ON a fresh Vast.ai pod. Idempotent. Puts the repo + venv + HF cache in
# /workspace (the only persistent volume), then runs the full concept pipeline.
#
#   bootstrap_pod.sh bald            # clone/update -> run -> upload to R2
#   bootstrap_pod.sh chubby my-branch
#
# Re-running is safe: existing repo is fetched + hard-reset, .venv is reused.
set -euo pipefail
CONCEPT="${1:?usage: bootstrap_pod.sh <concept> [git-ref]}"
REF="${2:-main}"
WS=/workspace
DIR="$WS/WB2"
REPO_URL="https://github.com/LudwikMadej/WB2.git"

# HF cache on the persistent volume -> CLIP weights survive pod stop/restart
export HF_HOME="$WS/.hf_home"
mkdir -p "$HF_HOME"

# 1. clone or update the repo in place (persistent location)
if [ -d "$DIR/.git" ]; then
  echo "[bootstrap] updating $DIR -> $REF"
  git -C "$DIR" fetch --depth=1 origin "$REF"
  git -C "$DIR" reset --hard FETCH_HEAD
  git -C "$DIR" clean -fd -e .venv -e data -e notebooks/results
else
  echo "[bootstrap] cloning $REPO_URL -> $DIR ($REF)"
  git clone --depth=1 --branch "$REF" "$REPO_URL" "$DIR"
fi

cd "$DIR"
echo "[bootstrap] HEAD $(git rev-parse --short HEAD)  concept=$CONCEPT"

# 2. hand off to the existing per-concept pipeline (env, data, run, R2 upload)
exec ./scripts/run_concept.sh "$CONCEPT"
