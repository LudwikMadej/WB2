#!/usr/bin/env bash
# scripts/pod.sh — drive a Vast.ai pod from your laptop.
#
#   ./scripts/pod.sh run bald [git-ref]   start the pipeline for a concept
#   ./scripts/pod.sh log bald             tail this concept's run log
#   ./scripts/pod.sh attach bald          attach to the live tmux session
#   ./scripts/pod.sh status               list running pipelines on the pod
#   ./scripts/pod.sh ssh                  open a shell on the pod
#
# The run survives SSH disconnects and pod-side hiccups (detached tmux).
# Logs + repo + venv live in /workspace, which persists across stop/restart.
set -euo pipefail
HOST="${WB2_POD:-vastai}"          # ssh alias; override: WB2_POD=vastai-proxy
LOGDIR=/workspace/WB2-runs
SELF="$(cd "$(dirname "$0")" && pwd)"

usage() { sed -n '2,12p' "$0"; exit "${1:-1}"; }
cmd="${1:-}"; shift || true

case "$cmd" in
  run)
    CONCEPT="${1:?usage: pod.sh run <concept> [git-ref]}"; REF="${2:-main}"
    SESSION="wb2-$CONCEPT"
    echo "[pod] uploading bootstrap -> $HOST"
    ssh "$HOST" "mkdir -p $LOGDIR /tmp/wb2"
    scp -q "$SELF/bootstrap_pod.sh" "$HOST:/tmp/wb2/bootstrap_pod.sh"
    LOG="$LOGDIR/$CONCEPT.log"
    echo "[pod] starting '$CONCEPT' (ref=$REF) in tmux session $SESSION"
    ssh "$HOST" "tmux has-session -t $SESSION 2>/dev/null && { echo 'ALREADY RUNNING: $SESSION (use: pod.sh log $CONCEPT)'; exit 3; } ; \
      tmux new-session -d -s $SESSION \"bash /tmp/wb2/bootstrap_pod.sh $CONCEPT $REF 2>&1 | tee $LOG\""
    echo "[pod] started. follow with:  ./scripts/pod.sh log $CONCEPT"
    ;;
  log)
    CONCEPT="${1:?usage: pod.sh log <concept>}"
    exec ssh -t "$HOST" "tail -n 200 -f $LOGDIR/$CONCEPT.log"
    ;;
  attach)
    CONCEPT="${1:?usage: pod.sh attach <concept>}"
    exec ssh -t "$HOST" "tmux attach -t wb2-$CONCEPT"
    ;;
  status)
    ssh "$HOST" 'echo "=== tmux sessions ==="; tmux ls 2>/dev/null || echo "(none)"; \
      echo "=== logs ==="; ls -lt /workspace/WB2-runs 2>/dev/null || echo "(none)"; \
      echo "=== gpu ==="; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader'
    ;;
  ssh)
    exec ssh -t "$HOST" "${1:-}"
    ;;
  ""|-h|--help) usage 0 ;;
  *) echo "unknown command: $cmd" >&2; usage 1 ;;
esac
