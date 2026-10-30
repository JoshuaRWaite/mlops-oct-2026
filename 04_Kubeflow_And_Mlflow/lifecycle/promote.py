"""The gate: pipeline step 3. Decide whether the model that was just trained stays the champion.

    python promote.py --data processed_images [--previous 2]

Module 3's train_model.py makes every new model the champion right away. This step runs straight after it and checks
that was a good idea: the new model and the previous champion are scored on the same images (mean reconstruction error,
the number the API puts on the Grafana dashboard). If the new model is no worse it stays champion (PROMOTED). If it is
worse, the 'champion' alias is moved back to the previous version and the new one is tagged gate=rejected (REJECTED):
it stays in the registry, but is not served. The API only loads the champion when it starts, so nothing serves the
rejected model in between.
"""
import argparse

import mlflow
import torch
from mlflow import MlflowClient
from torch.utils.data import DataLoader

from config import MLFLOW_TRACKING_URI, MODEL_ALIAS, MODEL_NAME
from train_model import HouseDataset  # module 3's image loader


def load_version(version):
    """Load one registered version of the model, on the CPU (scoring ~200 small images doesn't need a GPU)."""
    return mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{version}", map_location="cpu").eval()


def score(model, folder, batch_size=32):
    """Mean squared reconstruction error of `model` over every PNG in `folder`."""
    data = HouseDataset(folder)
    if len(data) == 0:
        raise SystemExit(f"No images in '{folder}'.")
    total = 0.0
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=batch_size):
            total += ((model(batch) - batch) ** 2).mean(dim=(1, 2, 3)).sum().item()
    return total / len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", required=True, help="folder of processed images to compare the models on")
    parser.add_argument("--previous", default="", help="the champion version from before training (default: the newest earlier version that wasn't rejected)")
    parser.add_argument("--tolerance", type=float, default=0.0, help="allow the new model to be this fraction worse")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    c = MlflowClient()
    versions = c.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise SystemExit(f"No versions of '{MODEL_NAME}' in the registry: did the train step finish?")
    candidate = max(versions, key=lambda v: int(v.version))  # the one the train step just registered
    if args.previous:
        previous = args.previous
    else:
        earlier = [v for v in versions if int(v.version) < int(candidate.version) and v.tags.get("gate") != "rejected"]
        previous = max(earlier, key=lambda v: int(v.version)).version if earlier else None

    new_err = score(load_version(candidate.version), args.data)
    print(f"candidate version {candidate.version}: error {new_err:.6f}")
    if previous is None or str(previous) == str(candidate.version):
        c.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, candidate.version)
        print(f"PROMOTED: version {candidate.version} is the champion (there was no earlier model to compare with).")
        return

    old_err = score(load_version(previous), args.data)
    limit = old_err * (1 + args.tolerance)
    print(f"previous champion, version {previous}: error {old_err:.6f}")
    if new_err <= limit:
        c.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, candidate.version)
        print(f"PROMOTED: version {candidate.version} is the champion ({new_err:.6f} <= {limit:.6f}).")
    else:
        c.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, previous)
        c.set_model_version_tag(MODEL_NAME, candidate.version, "gate", "rejected")
        print(f"REJECTED: version {candidate.version} is worse ({new_err:.6f} > {limit:.6f}). "
              f"'@{MODEL_ALIAS}' moved back to version {previous}; version {candidate.version} stays in the registry, unserved.")


if __name__ == "__main__":
    main()
