"""Start the pipeline from the command line (06_check_drift.sh does this for you when a trigger fires).

    .venv/bin/python pipelines/run_pipeline.py [--cpu] [--epochs 10] [--raw-zip-uri minio://...] [--previous-version N] [--run-name NAME] [--wait]

Needs ./port_forward.sh to be running (Kubeflow Pipelines on http://127.0.0.1:8080).
"""
import argparse
from pathlib import Path

import kfp

HERE = Path(__file__).resolve().parent
URI_FILE = HERE.parent / ".run" / "mlflow_uri"  # written by 04_connect.sh

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true", help="use house_pipeline_cpu.yaml")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--raw-zip-uri", help="zip to train on (default: the zip from 05_upload_data.sh)")
parser.add_argument("--mlflow-uri", help="module 3's MLflow as seen from the cluster (default: what 04_connect.sh found)")
parser.add_argument("--previous-version", default="", help="the champion before this run (06_check_drift.sh fills it in)")
parser.add_argument("--run-name", default="house-autoencoder")
parser.add_argument("--wait", action="store_true", help="wait until the run has finished")
parser.add_argument("--host", default="http://127.0.0.1:8080")
args = parser.parse_args()

arguments = {"epochs": args.epochs, "previous_version": args.previous_version}
if args.raw_zip_uri:
    arguments["raw_zip_uri"] = args.raw_zip_uri
mlflow_uri = args.mlflow_uri or (URI_FILE.read_text().strip() if URI_FILE.exists() else None)
if mlflow_uri:
    arguments["mlflow_uri"] = mlflow_uri

package = HERE / ("house_pipeline_cpu.yaml" if args.cpu else "house_pipeline.yaml")
client = kfp.Client(host=args.host)
run = client.create_run_from_pipeline_package(
    str(package), arguments=arguments, run_name=args.run_name, experiment_name="House Autoencoder")
print(f"Started run {run.run_id}. Watch it at {args.host}/#/runs/details/{run.run_id}")
if args.wait:
    print("Waiting for it to finish ...")
    result = client.wait_for_run_completion(run.run_id, timeout=3600)
    print(f"Run finished: {result.state}")
    raise SystemExit(0 if result.state == "SUCCEEDED" else 1)
