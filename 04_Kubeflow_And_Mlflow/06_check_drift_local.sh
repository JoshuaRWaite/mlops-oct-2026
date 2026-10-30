#!/usr/bin/env bash
# Hands-on version: the whole loop (drift check -> retrain -> gate -> new model served) using only module 3's stack.
# No Kubernetes, Docker or setup steps needed. It runs the same scripts the pipeline runs.
#   ./06_check_drift_local.sh               (extra options go to check_drift.py, e.g. --window 1m or --factor 2)
#   EPOCHS=1 ./06_check_drift_local.sh --force    retrain even without a trigger, with a deliberately weak model: the gate says no
# Needs: module 3's services and API running, a trained champion, and some drifted traffic sent to the API:
#   ./send_drift.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M3="$HERE/../03_Full_Cycle_MLOps"
[ -f "$M3/.venv/bin/activate" ] || { echo "Error: finish module 3 first (cd ../03_Full_Cycle_MLOps && ./setup_env.sh)." >&2; exit 1; }
# shellcheck disable=SC1091
source "$M3/.venv/bin/activate"
cd "$M3"
export PYTHONPATH="$M3:$HERE/lifecycle"
WORK="$HERE/.run/local-retrain"
mkdir -p "$HERE/.run"

echo "==> 1. Drift check"
set +e
python "$HERE/lifecycle/check_drift.py" --save-version "$HERE/.run/previous_champion" "$@"
STATUS=$?
set -e
[ "$STATUS" -eq 10 ] || exit "$STATUS"          # 0 = no drift (done), 10 = retrain, anything else = an error
PREVIOUS="$(cat "$HERE/.run/previous_champion")"

rm -rf "$WORK"; mkdir -p "$WORK/raw/incoming"
cp -r uploaded_images/. "$WORK/raw/"
cp "$HERE"/incoming_images/*.png "$WORK/raw/incoming/" 2>/dev/null || echo "(no images in incoming_images/: retraining on the original photos only)"
echo
echo "==> 2. Preprocess: your original photos + the $(find "$WORK/raw/incoming" -name '*.png' | wc -l) new ones (module 3's process_images.py)"
python process_images.py --input "$WORK/raw" --output "$WORK/processed"

echo
echo "==> 3. Train a new version (module 3's train_model.py; like in module 3, it becomes the champion right away)"
python train_model.py --data "$WORK/processed" --epochs "${EPOCHS:-30}"

echo
echo "==> 4. Gate: keep the new model only if it beats the previous champion (version $PREVIOUS)"
python "$HERE/lifecycle/promote.py" --data "$WORK/processed" --previous "$PREVIOUS"

echo
echo "==> 5. Restart the API (module 3's start_api.sh) so it loads whichever model is champion now"
"$HERE/lifecycle/restart_api.sh"
