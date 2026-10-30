"""Has the data moved away from what the champion was trained on?  Three triggers, ANY of which means "retrain":

  performance   the API's recent reconstruction error (from Prometheus, the number on your Grafana panel) is more
                than --factor times the champion's error when it was trained (its final val_loss in MLflow)
  data volume   at least --min-new new images have reached the API since the champion was trained
                (counted in incoming_images/, where send_drift.sh keeps a copy of what it sends)
  time          the champion is older than --max-age-hours

    python check_drift.py            exit code 0 = fine, 10 = retrain

Run it with ./06_check_drift.sh, which also starts the retraining pipeline when a trigger fires.
Needs module 3's stack running (MLflow :5000, Prometheus :9090) and some traffic sent to the API (./send_drift.sh).
"""
import argparse
import sys
import time
from pathlib import Path

import requests

RETRAIN, OK = 10, 0
MODEL_NAME, MODEL_ALIAS = "house-autoencoder", "champion"  # same as module 3's config.py


def champion(mlflow_url):
    r = requests.get(f"{mlflow_url}/api/2.0/mlflow/registered-models/alias",
                     params={"name": MODEL_NAME, "alias": MODEL_ALIAS}, timeout=10)
    if r.status_code != 200:
        sys.exit(f"No '@{MODEL_ALIAS}' model of '{MODEL_NAME}' in MLflow ({r.status_code}). Finish module 3's training first.")
    return r.json()["model_version"]


def trained_error(mlflow_url, run_id):
    """The champion's error when it was trained: the last val_loss logged by its run."""
    r = requests.get(f"{mlflow_url}/api/2.0/mlflow/runs/get", params={"run_id": run_id}, timeout=10)
    r.raise_for_status()
    metrics = {m["key"]: m["value"] for m in r.json()["run"]["data"].get("metrics", [])}  # latest value of each
    return metrics["val_loss"]


def live_error(prom_url, window):
    """Mean reconstruction error of the requests the API handled in the last `window` (same as the Grafana panel)."""
    q = (f"sum(increase(model_reconstruction_error_sum[{window}])) / "
         f"sum(increase(model_reconstruction_error_count[{window}]))")
    result = requests.get(f"{prom_url}/api/v1/query", params={"query": q}, timeout=10).json()["data"]["result"]
    if not result or result[0]["value"][1] == "NaN":
        return None
    return float(result[0]["value"][1])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--mlflow", default="http://127.0.0.1:5000")
    parser.add_argument("--prometheus", default="http://127.0.0.1:9090")
    parser.add_argument("--incoming", default=str(Path(__file__).resolve().parents[1] / "incoming_images"),
                        help="folder of new images the API has received (send_drift.sh fills it)")
    parser.add_argument("--save-version", help="write the champion's version number to this file (the retraining scripts use it)")
    parser.add_argument("--window", default="2m", help="how far back to look at the API's error (Prometheus syntax)")
    parser.add_argument("--factor", type=float, default=1.5)
    parser.add_argument("--min-new", type=int, default=500)
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    parser.add_argument("--force", action="store_true", help="retrain even if no trigger fired (for demos)")
    args = parser.parse_args()

    version = champion(args.mlflow)
    baseline = trained_error(args.mlflow, version["run_id"])
    created = int(version["creation_timestamp"]) / 1000
    if args.save_version:
        Path(args.save_version).write_text(str(version["version"]))
    age_hours = (time.time() - created) / 3600
    new_images = sum(1 for p in Path(args.incoming).glob("*.png") if p.stat().st_mtime > created)
    error = live_error(args.prometheus, args.window)

    rows = [
        ("performance",
         f"{error:.5f}" if error is not None else "no traffic", f"{args.factor * baseline:.5f}",
         error is not None and error > args.factor * baseline),
        ("data volume", f"{new_images} images", f"{args.min_new} images", new_images >= args.min_new),
        ("time", f"{age_hours:.1f} h", f"{args.max_age_hours:.1f} h", age_hours > args.max_age_hours),
    ]
    print(f"Champion: {MODEL_NAME} version {version['version']} (error when trained: {baseline:.5f})\n")
    print(f"{'trigger':<14}{'now':>14}{'limit':>14}   result")
    for name, now, limit, fired in rows:
        print(f"{name:<14}{now:>14}{limit:>14}   {'FIRED' if fired else 'ok'}")
    if error is not None:
        print(f"\nThe API's error over the last {args.window} is {error / baseline:.1f}x what the champion saw in training.")

    fired = [r[0] for r in rows if r[3]]
    if args.force:
        fired.append("--force")
    if fired:
        print(f"RETRAIN: {', '.join(fired)} triggered.")
        sys.exit(RETRAIN)
    print("OK: no trigger fired.")


if __name__ == "__main__":
    try:
        main()
    except requests.ConnectionError as exc:
        sys.exit(f"Can't reach {exc.request.url.split('/api')[0]}: is module 3's stack running (./start_services.sh)?")
