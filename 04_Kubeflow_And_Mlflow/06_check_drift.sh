#!/usr/bin/env bash
# Step 6: check module 3's API for drift. If a trigger fires, retrain automatically with the Kubeflow pipeline.
#   ./06_check_drift.sh              (extra options go to check_drift.py, e.g. --window 5m, --factor 2, --force)
# On a trigger it zips your original images plus the new images (incoming_images/), uploads them, runs the pipeline,
# waits for it, then restarts module 3's API so it serves whichever model is champion afterwards.
# Needs: module 3's stack running with some traffic sent to the API (./send_drift.sh), and steps 1-5 done.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need kubectl; need curl
need_cluster
[ -x "$M4_DIR/.venv/bin/python" ] || { echo "Error: run ./01_setup_host.sh first." >&2; exit 1; }

set +e
"$M4_DIR/.venv/bin/python" "$M4_DIR/lifecycle/check_drift.py" --save-version "$RUN_DIR/previous_champion" "$@"
STATUS=$?
set -e
[ "$STATUS" -eq 10 ] || exit "$STATUS"        # 0 = no drift (done), 10 = retrain, anything else = an error
PREVIOUS="$(cat "$RUN_DIR/previous_champion")"

NAME="retrain-$(date +%Y%m%d-%H%M%S).zip"
echo
echo "==> Building $NAME: your original images + the $(find "$M4_DIR/incoming_images" -name '*.png' 2>/dev/null | wc -l) new ones"
rm -f "$RUN_DIR/retrain-"*.zip
python3 - "$SHARED_DIR/uploaded_images" "$M4_DIR/incoming_images" "$RUN_DIR/$NAME" <<'PY'
import sys, zipfile
from pathlib import Path
originals, incoming, out = map(Path, sys.argv[1:])
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
    for root in (originals, incoming):
        if root.is_dir():
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, f"{root.name}/{p.relative_to(root)}")
PY
"$M4_DIR/05_upload_data.sh" "$RUN_DIR/$NAME"

echo "==> Starting the pipeline (30 epochs: enough to learn the new pattern)"
FLAG=""; [ "$(cluster_mode)" = cpu ] && FLAG="--cpu"
# shellcheck disable=SC2086
"$M4_DIR/.venv/bin/python" "$M4_DIR/pipelines/run_pipeline.py" $FLAG --epochs 30 --previous-version "$PREVIOUS" \
    --raw-zip-uri "minio://mlpipeline/data/$NAME" --run-name drift-retrain --wait

echo "==> Restarting module 3's API (its start_api.sh) so it loads whichever model is champion now"
"$M4_DIR/lifecycle/restart_api.sh"
