import math
import os
from pathlib import Path

from PIL import Image, ImageDraw

# Output directory
OUT_DIR = Path(__file__).resolve().parent / "masks_640"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H = W = 640  # mask size
FILL = 255    # white = keep region
BG = 0        # black = freeze region


def save_mask(img: Image.Image, name: str):
    path = OUT_DIR / f"{name}.png"
    img.save(path)
    print(f"saved {path}")


def base():
    return Image.new("L", (W, H), color=BG)


def mask_square():
    img = base()
    draw = ImageDraw.Draw(img)
    side = int(min(W, H) * 0.5)  # ~1/4 area
    x0 = (W - side) // 2
    y0 = (H - side) // 2
    draw.rectangle([x0, y0, x0 + side, y0 + side], fill=FILL)
    save_mask(img, "mask_square_center")


def mask_circle():
    img = base()
    draw = ImageDraw.Draw(img)
    r = int(min(W, H) * 0.25 * math.sqrt(2))  # circle area ~ square area
    cx = W // 2
    cy = H // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=FILL)
    save_mask(img, "mask_circle_center")


def mask_star():
    img = base()
    draw = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2
    R_outer = int(min(W, H) * 0.28)
    R_inner = int(R_outer * 0.45)
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5  # start at top
        r = R_outer if i % 2 == 0 else R_inner
        x = cx + r * math.cos(angle)
        y = cy - r * math.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=FILL)
    save_mask(img, "mask_star_center")


def mask_diag_band():
    img = base()
    draw = ImageDraw.Draw(img)
    band_width = int(min(W, H) * 0.35)  # gives ~1/4 area when diagonal
    # Define a parallelogram (diagonal band from top-left to bottom-right)
    draw.polygon([
        (-band_width, 0),
        (0, 0),
        (W, H),
        (W - band_width, H),
    ], fill=FILL)
    draw.polygon([
        (0, -band_width),
        (W, H - band_width),
        (W, H),
        (0, band_width),
    ], fill=FILL)
    save_mask(img, "mask_diag_band")


def main():
    mask_square()
    mask_circle()
    mask_star()
    mask_diag_band()


if __name__ == "__main__":
    main()
