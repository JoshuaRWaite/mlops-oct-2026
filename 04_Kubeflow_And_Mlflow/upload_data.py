"""Copy a zip of images into the cluster's object store (MinIO), where pipelines can read it.

Run it through 05_upload_data.sh, which opens the connection to MinIO for you:
    ./05_upload_data.sh ~/DrivewayClassificationData.zip
"""
import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("zip_file")
    parser.add_argument("--endpoint", default=os.environ.get("MINIO_ENDPOINT", "http://127.0.0.1:9000"))
    parser.add_argument("--bucket", default="mlpipeline")  # the bucket Kubeflow Pipelines already uses
    parser.add_argument("--prefix", default="data")
    args = parser.parse_args()

    path = Path(args.zip_file)
    if not path.is_file():
        sys.exit(f"File not found: {path}")

    s3 = boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minio"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minio123"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    try:
        s3.head_bucket(Bucket=args.bucket)
    except ClientError:
        s3.create_bucket(Bucket=args.bucket)

    key = f"{args.prefix}/{path.name}"
    print(f"Uploading {path} ({path.stat().st_size / 1e6:.1f} MB) to {args.bucket}/{key} ...")
    s3.upload_file(str(path), args.bucket, key)
    print("Done. Use this as the pipeline's raw_zip_uri:")
    print(f"  minio://{args.bucket}/{key}")


if __name__ == "__main__":
    main()
