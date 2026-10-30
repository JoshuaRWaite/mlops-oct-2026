"""The module 3 workflow, as a Kubeflow Pipeline:  preprocess -> train -> promote (the gate).

    python house_pipeline.py            # writes house_pipeline.yaml      (training on the GPU)
    python house_pipeline.py --cpu      # writes house_pipeline_cpu.yaml  (training on the CPU)

Upload the .yaml in the Kubeflow UI (Pipelines -> Upload pipeline) and start a run.
Preprocess and train are the SAME scripts you ran by hand in module 3, and they log to the SAME MLflow. Promote is new:
it keeps the new model as champion only if it beats the previous one, otherwise it moves the alias back (lifecycle/promote.py). The drift check
(06_check_drift.sh) starts this pipeline.
"""
import argparse

from kfp import compiler, dsl

IMAGE = "house-autoencoder:module4"  # built by build_image.sh and loaded into the cluster
MLFLOW_URI = "http://172.18.0.1:5000"  # module 3's MLflow, seen from the cluster (06/run_pipeline.py pass the real address)
DEFAULT_ZIP = "minio://mlpipeline/data/module3-images.zip"  # module 3's uploaded images, put there by 05_upload_data.sh


@dsl.container_component
def preprocess(raw_zip: dsl.Input[dsl.Dataset], processed: dsl.Output[dsl.Dataset]):
    """Unzip the photos, then run process_images.py (128x128 grayscale PNGs)."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c", 'python -m zipfile -e "$0" /tmp/raw && python process_images.py --input /tmp/raw --output "$1"'],
        args=[raw_zip.path, processed.path],
    )


@dsl.container_component
def train(processed: dsl.Input[dsl.Dataset], mlflow_uri: str, epochs: int, batch_size: int):
    """Run train_model.py. It logs to MLflow, registers a new model version and (as in module 3) makes it champion."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c", 'MLFLOW_TRACKING_URI="$0" python train_model.py --data "$1" --epochs "$2" --batch-size "$3"'],
        args=[mlflow_uri, processed.path, epochs, batch_size],
    )


@dsl.container_component
def promote(processed: dsl.Input[dsl.Dataset], mlflow_uri: str, previous_version: str):
    """The gate: score the new model and the previous champion on the same images; keep the new one only if it is no worse."""
    return dsl.ContainerSpec(
        image=IMAGE,
        command=["sh", "-c", 'MLFLOW_TRACKING_URI="$0" python promote.py --data "$1" ${2:+--previous "$2"}'],
        args=[mlflow_uri, processed.path, previous_version],
    )


def build_pipeline(gpu: bool):
    @dsl.pipeline(
        name="house-autoencoder",
        description="Preprocess house photos, train an autoencoder, register it in MLflow, and promote it if it beats the champion.",
    )
    def house_pipeline(
        raw_zip_uri: str = DEFAULT_ZIP,
        mlflow_uri: str = MLFLOW_URI,
        epochs: int = 10,
        batch_size: int = 32,
        previous_version: str = "",  # the champion from before this run; empty = the newest earlier version
    ):
        # "Import" the zip you uploaded to the cluster's MinIO object store (see 05_upload_data.sh).
        raw = dsl.importer(artifact_uri=raw_zip_uri, artifact_class=dsl.Dataset)
        prep = preprocess(raw_zip=raw.output)
        trained = train(
            processed=prep.outputs["processed"],
            mlflow_uri=mlflow_uri,
            epochs=epochs,
            batch_size=batch_size,
        )
        trained.set_caching_options(False)  # always train again, even with identical settings
        if gpu:
            trained.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
        gate = promote(processed=prep.outputs["processed"], mlflow_uri=mlflow_uri, previous_version=previous_version).after(trained)
        gate.set_caching_options(False)

    return house_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cpu", action="store_true", help="train on the CPU instead of requesting a GPU")
    args = parser.parse_args()

    out = "house_pipeline_cpu.yaml" if args.cpu else "house_pipeline.yaml"
    compiler.Compiler().compile(build_pipeline(gpu=not args.cpu), package_path=out)
    print(f"Wrote {out}")
