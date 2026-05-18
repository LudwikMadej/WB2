#!/usr/bin/env bash
# scripts/run_concept.sh <concept>     e.g.  bald | chubby | wearing_hat | male
# Full pipeline for ONE CelebA concept on a fresh non-Blackwell pod.
# Run one concept per machine in parallel; an assembler collects results into the PR.
set -euo pipefail
CONCEPT="${1:?usage: run_concept.sh <concept>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
BUCKET="${WB2_BUCKET:-wb2}"        # export WB2_BUCKET=wb2-150gb to use the bigger bucket
ENDPOINT="https://f681dbaf79de38cf431e125730894e42.r2.cloudflarestorage.com"
RC=( --s3-provider Cloudflare --s3-access-key-id 8e7789ce4343edb621c9da7a0a8740fe
  --s3-secret-access-key 244bb6fbc0216445e0d8cef64b89d7c520d58e79d137ccf10ac9ae119c363002
  --s3-endpoint "$ENDPOINT" --s3-no-check-bucket --transfers 32 --checkers 64 )
LOG=checkpoints/_run_logs; mkdir -p "$LOG" /tmp/wb2_exec
export JUPYTER_CONFIG_DIR=/tmp/jcfg_empty; mkdir -p "$JUPYTER_CONFIG_DIR"

# 0. guard: never run on Blackwell (cu126 torch won't work -> 5x slower fallback)
nvidia-smi --query-gpu=name --format=csv,noheader | grep -qiE '5090|RTX PRO 6000|B200|B300' \
  && { echo "FATAL: Blackwell GPU — use a 4090/A-series pod."; exit 2; }

# 1. env (idempotent)
[ -x .venv/bin/python ] || { pip install -q uv && uv sync --extra jupyter; }
# Install the venv kernel into a dir we own and put FIRST on JUPYTER_PATH, so
# nbconvert resolves kernel "wb2" to .venv/bin/python regardless of any
# competing system "python3" kernelspec in the base image (Vast ships one).
KDIR=/tmp/wb2_kernel; rm -rf "$KDIR"
.venv/bin/python -m ipykernel install --prefix="$KDIR" --name wb2 --display-name WB2 >/dev/null 2>&1
export JUPYTER_PATH="$KDIR/share/jupyter"
.venv/bin/python -c "import torch,pandas,transformers,xgboost,sklearn; assert torch.cuda.is_available(); print('env OK',torch.__version__,torch.cuda.get_device_name(0))"
.venv/bin/python -m jupyter kernelspec list 2>&1 | grep -q "wb2 .*$KDIR" || { echo "FATAL: wb2 kernel not registered"; exit 3; }

# 2. source data from R2 (only if absent)
{ [ -f data/metadata.csv ] && [ -d data/activations/raw ]; } || ./scripts/sync_data.sh download

PER="notebooks/single_debias/02_concept_detection.ipynb notebooks/single_debias/03_concept_detection_debiased.ipynb notebooks/single_debias/04_get_single_debiased_activations.ipynb notebooks/multiple_debias/01_sequential_debiasing.ipynb notebooks/multiple_debias/02_concept_recovery.ipynb notebooks/multiple_debias/03_iterative_single_layer.ipynb notebooks/multiple_debias/04_fixed_multi_layer.ipynb notebooks/multiple_debias/05_iterative_per_layer.ipynb notebooks/zero_shot_eval/02_zero_shot_evaluation.ipynb"
DIST="notebooks/distributions_analysis/01_raw_normality.ipynb notebooks/distributions_analysis/02_debiasing_shift.ipynb notebooks/distributions_analysis/03_downstream_recovery.ipynb"

# 3. set the concept (CONCEPT scalar in 9, CONCEPTS list in 3)
.venv/bin/python - "$CONCEPT" $PER -- $DIST <<'PY'
import re,sys,pathlib
a=sys.argv; c=a[1]; sep=a.index('--'); per=a[2:sep]; dist=a[sep+1:]
for f in per:
    p=pathlib.Path(f); s=p.read_text()
    s,n=re.subn(r"(CONCEPT\s*=\s*)'[A-Za-z0-9_]+'", lambda m:m.group(1)+repr(c).replace('"',"'"), s, 1)
    assert n==1, f; p.write_text(s)
for f in dist:
    p=pathlib.Path(f); s=p.read_text()
    s,n=re.subn(r"(CONCEPTS\s*=\s*)\[[^\]]*\]", lambda m:m.group(1)+f"['{c}']", s, 1)
    assert n==1, f; p.write_text(s)
print("CONCEPT set ->",c)
PY

# 4. run the 12 notebooks in dependency order; stop on first failure
for nb in $PER $DIST; do
  o=$(echo "$nb" | tr / _); echo "[$(date +%T)] $nb"
  .venv/bin/python -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=wb2 \
    --output-dir /tmp/wb2_exec --output "$o" "$nb" > "$LOG/$o.log" 2>&1 \
    || { echo "FAILED $nb"; tail -40 "$LOG/$o.log"; exit 1; }
done

# 5. namespace distributions outputs under the concept
D=notebooks/results/distributions_analysis; mkdir -p "$D/$CONCEPT"
for d in raw_normality debiasing_shift downstream_recovery; do
  [ -d "$D/$d" ] && { rm -rf "$D/$CONCEPT/$d"; mv "$D/$d" "$D/$CONCEPT/$d"; }
done

# 6. upload ONLY this concept's results to R2 (assembler will collect & PR them)
rclone copy notebooks/results ":s3:${BUCKET}/results" "${RC[@]}" \
  --include "single_debias/${CONCEPT}/**" --include "multiple_debias/${CONCEPT}/**" \
  --include "zero_shot_eval/${CONCEPT}/**" --include "distributions_analysis/${CONCEPT}/**"

# 7. free big intermediates; leave repo clean (results kept, notebook edits reverted)
rm -rf data/activations/debiased data/multiple_debiasing
git checkout -- notebooks/single_debias notebooks/multiple_debias notebooks/zero_shot_eval notebooks/distributions_analysis 2>/dev/null || true
echo "DONE: $CONCEPT  -> s3:${BUCKET}/results/.../${CONCEPT}"
