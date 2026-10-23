"""Optional: send fake traffic to the API so the Grafana dashboard has something to show.

    python generate_traffic.py                          # 200 normal requests
    python generate_traffic.py --noise-fraction 0.3     # 30% random-noise images -> error spike
"""
import argparse
import io
import random
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from config import PROCESSED_DIR


def noise_png():
    """A random-static image: nothing like a house, so the model reconstructs it badly."""
    pixels = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--n", type=int, default=200, help="number of requests")
    parser.add_argument("--delay", type=float, default=0.1, help="seconds between requests")
    parser.add_argument("--noise-fraction", type=float, default=0.0, help="share of requests that are random noise")
    args = parser.parse_args()

    files = sorted(Path(PROCESSED_DIR).glob("*.png"))
    if not files:
        raise SystemExit(f"No images in '{PROCESSED_DIR}/'. Run process_images.py first.")

    ok = 0
    for i in range(args.n):
        data = noise_png() if random.random() < args.noise_fraction else random.choice(files).read_bytes()
        response = requests.post(args.url, files={"file": ("image.png", data, "image/png")}, timeout=30)
        ok += response.ok
        if (i + 1) % 50 == 0:
            print(f"sent {i + 1}/{args.n} ({ok} ok)")
        time.sleep(args.delay)
    print(f"Done: {ok}/{args.n} succeeded.")
