"""A tiny pipeline that checks the plumbing BEFORE you run the real one.

    python smoke_test.py            # writes smoke_test.yaml      (checks MLflow and the GPU)
    python smoke_test.py --cpu      # writes smoke_test_cpu.yaml  (checks MLflow only)

If this passes, the cluster can: start a pipeline pod from our image, reach MLflow, and (GPU version) see the GPU.
Open each step's Logs tab in the run to read the output.
"""
import argparse

from kfp import compiler, dsl

IMAGE = "house-autoencoder:module4"
MLFLOW_URI = "http://172.18.0.1:5000"  # module 3's MLflow as seen from the cluster; 04_connect.sh prints the real one

CHECK_MLFLOW = """
import sys, urllib.request
url = sys.argv[1].rstrip("/") + "/health"
print("GET", url)
print("MLflow answered:", urllib.request.urlopen(url, timeout=10).read().decode())
"""

CHECK_GPU = """
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("FAIL: no GPU is visible inside this pod")
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device="cuda")
print("matrix multiply on the GPU worked:", bool((x @ x).sum().isfinite()))
"""


@dsl.container_component
def check_mlflow(mlflow_uri: str):
    return dsl.ContainerSpec(image=IMAGE, command=["python", "-c", CHECK_MLFLOW], args=[mlflow_uri])


@dsl.container_component
def check_gpu():
    return dsl.ContainerSpec(image=IMAGE, command=["python", "-c", CHECK_GPU])


def build_pipeline(gpu: bool):
    @dsl.pipeline(name="smoke-test", description="Can a pipeline pod reach MLflow and see the GPU?")
    def smoke_test(mlflow_uri: str = MLFLOW_URI):
        check_mlflow(mlflow_uri=mlflow_uri).set_caching_options(False)
        if gpu:
            task = check_gpu().set_caching_options(False)
            task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)

    return smoke_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--cpu", action="store_true", help="skip the GPU check")
    args = parser.parse_args()

    out = "smoke_test_cpu.yaml" if args.cpu else "smoke_test.yaml"
    compiler.Compiler().compile(build_pipeline(gpu=not args.cpu), package_path=out)
    print(f"Wrote {out}")
