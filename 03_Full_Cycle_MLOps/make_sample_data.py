"""Don't have the house-images zip yet? Generate a fake one to try the pipeline.

    python make_sample_data.py            # writes sample_houses.zip (300 drawn "houses")
    python make_sample_data.py --n 1000
"""
import argparse
import random
import zipfile
from io import BytesIO

from PIL import Image, ImageDraw


def draw_house(rng):
    img = Image.new("RGB", (256, 256), (rng.randint(120, 220), rng.randint(160, 240), 255))  # sky
    d = ImageDraw.Draw(img)
    d.rectangle([0, 190, 256, 256], fill=(rng.randint(40, 90), rng.randint(110, 170), 50))  # lawn
    left, width = rng.randint(30, 70), rng.randint(110, 150)
    top, bottom = rng.randint(100, 130), 205
    wall = tuple(rng.randint(120, 250) for _ in range(3))
    d.rectangle([left, top, left + width, bottom], fill=wall)  # walls
    d.polygon([(left - 12, top), (left + width + 12, top), (left + width / 2, top - rng.randint(40, 70))],
              fill=(rng.randint(90, 160), rng.randint(30, 70), rng.randint(30, 60)))  # roof
    door_x = left + width // 2 - 10
    d.rectangle([door_x, bottom - 45, door_x + 20, bottom], fill=(90, 55, 30))  # door
    for wx in (left + 12, left + width - 38):  # windows
        d.rectangle([wx, top + 18, wx + 26, top + 44], fill=(190, 225, 250), outline=(255, 255, 255), width=2)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", default="sample_houses.zip")
    args = parser.parse_args()

    rng = random.Random(0)
    with zipfile.ZipFile(args.out, "w") as zf:
        for i in range(args.n):
            buffer = BytesIO()
            draw_house(rng).save(buffer, format="JPEG")
            zf.writestr(f"houses/house_{i:04d}.jpg", buffer.getvalue())  # in a subfolder, like real zips
    print(f"Wrote {args.n} images to {args.out}")
