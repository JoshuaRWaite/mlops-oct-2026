"""Step 3: turn raw uploaded photos into clean 128x128 grayscale PNGs.

    python process_images.py                       # uploaded_images/ -> processed_images/
    python process_images.py --input ~/my_photos   # use a different folder
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2

from config import IMG_SIZE, PROCESSED_DIR, UPLOAD_DIR

EXTENSIONS = {".png", ".jpg", ".jpeg"}


def process_images(input_dir, output_dir, size=IMG_SIZE):
    """Return (processed, skipped). Walks subfolders; the output is rebuilt from scratch."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)  # don't mix old and new data
    output_dir.mkdir(parents=True)

    processed = skipped = 0
    for path in sorted(input_dir.rglob("*")):
        # Skip folders, other file types, and the junk macOS adds to zip files.
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # None if the file is corrupt
        if img is None:
            skipped += 1
            continue
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_dir / f"{processed:06d}.png"), img)
        processed += 1
    return processed, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", default=UPLOAD_DIR, help=f"folder of raw images (default: {UPLOAD_DIR})")
    parser.add_argument("--output", default=PROCESSED_DIR, help=f"where to write results (default: {PROCESSED_DIR})")
    args = parser.parse_args()

    if not Path(args.input).is_dir():
        sys.exit(f"Input folder '{args.input}' not found. Upload a zip in the Streamlit app first.")
    done, bad = process_images(args.input, args.output)
    print(f"Processed {done} images into '{args.output}/' ({bad} unreadable files skipped).")
    if done == 0:
        sys.exit("No images found. The zip should contain .png, .jpg or .jpeg files.")
