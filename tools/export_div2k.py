"""
Export DIV2K bicubic x4 validation split to PNG pairs (HR/LR).
Requires: tensorflow-datasets

Usage (from repo root):
  conda run -n SUPIR python SUPIR/tools/export_div2k.py \
    --split validation \
    --data_dir D:/cv_diff_lab/tfds_cache \
    --out_hr D:/cv_diff_lab/data/div2k_hr \
    --out_lr D:/cv_diff_lab/data/div2k_lr

Change paths as needed. For train split use --split train.
"""
import argparse
from pathlib import Path

from PIL import Image
import tensorflow_datasets as tfds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation"], help="DIV2K split")
    parser.add_argument("--data_dir", default="./tfds_cache", help="TFDS data_dir for caching")
    parser.add_argument("--out_hr", default="./data/div2k_hr", help="Output folder for HR")
    parser.add_argument("--out_lr", default="./data/div2k_lr", help="Output folder for LR (bicubic x4)")
    args = parser.parse_args()

    out_hr = Path(args.out_hr)
    out_lr = Path(args.out_lr)
    out_hr.mkdir(parents=True, exist_ok=True)
    out_lr.mkdir(parents=True, exist_ok=True)

    ds = tfds.load("div2k/bicubic_x4", split=args.split, data_dir=args.data_dir)

    for i, sample in enumerate(tfds.as_numpy(ds)):
        hr = Image.fromarray(sample["hr"])
        lr = Image.fromarray(sample["lr"])
        hr.save(out_hr / f"{i:04d}.png")
        lr.save(out_lr / f"{i:04d}.png")
        if (i + 1) % 50 == 0:
            print(f"Saved {i+1} images...")

    print("Done.")


if __name__ == "__main__":
    main()
