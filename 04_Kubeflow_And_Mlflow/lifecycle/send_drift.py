"""Send drifted traffic to module 3's API, and keep a copy of every image sent.

    ./send_drift.sh [--n 100] [--reset]

Every photo from module 3's processed_images/ gets a bright grid drawn over it (a smudged lens, a new watermark), and is
sent to the API. The API's reconstruction-error metric rises, which is what the drift check watches. The copies go to
incoming_images/ in this folder: they are the "new data" the retraining run learns from. (A production API would log
the images it receives; module 3's doesn't, so the sender keeps the copies.)
"""
import argparse
import io
import random
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

M4 = Path(__file__).resolve().parents[1]
PROCESSED = M4.parent / "03_Full_Cycle_MLOps" / "processed_images"


def grid_png(path):
    """A real house photo with a bright grid over it. Unlike random static, a model CAN learn this pattern."""
    pixels = np.asarray(Image.open(path).convert("L"), dtype=np.uint8).copy()
    pixels[::8, :] = 255
    pixels[:, ::8] = 255
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--n", type=int, default=100, help="number of requests")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    parser.add_argument("--images", default=str(PROCESSED), help="folder of module 3's processed images")
    parser.add_argument("--incoming", default=str(M4 / "incoming_images"), help="where to keep copies of what is sent")
    parser.add_argument("--reset", action="store_true", help="delete earlier copies in the incoming folder first")
    args = parser.parse_args()

    files = sorted(Path(args.images).glob("*.png"))
    if not files:
        raise SystemExit(f"No images in '{args.images}'. Finish module 3 (process_images.py) first.")
    incoming = Path(args.incoming)
    incoming.mkdir(exist_ok=True)
    if args.reset:
        for old in incoming.glob("*.png"):
            old.unlink()

    ok = 0
    for i in range(args.n):
        data = grid_png(random.choice(files))
        (incoming / f"drift_{time.time_ns()}.png").write_bytes(data)
        try:
            ok += requests.post(args.url, files={"file": ("image.png", data, "image/png")}, timeout=30).ok
        except requests.ConnectionError:
            raise SystemExit(f"Can't reach the API at {args.url}. Is it running? (cd ../03_Full_Cycle_MLOps && ./start_api.sh)")
        if (i + 1) % 50 == 0:
            print(f"sent {i + 1}/{args.n} ({ok} ok)")
        time.sleep(args.delay)
    print(f"Done: {ok}/{args.n} succeeded. Copies are in {incoming} ({len(list(incoming.glob('*.png')))} images there).")


if __name__ == "__main__":
    main()
