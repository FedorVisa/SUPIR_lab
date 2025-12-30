"""
Compute PSNR, LPIPS, and optionally CLIP-IQA for SR outputs vs HR references.

Usage example:
  conda run -n SUPIR python SUPIR/tools/evaluate_metrics.py \
    --sr_dir D:/cv_diff_lab/SUPIR/images_out \
    --hr_dir D:/cv_diff_lab/data/div2k_hr \
    --out_csv D:/cv_diff_lab/SUPIR/metrics_div2k.csv \
    --device cuda

Notes:
- Filenames must match between SR and HR (e.g., 0001.png).
- Mask label is inferred from filename suffix before extension if formatted like name_mask.png
  (e.g., 0001_mask_star_center.png -> mask=mask_star_center). Otherwise mask="nomask".
- CLIP-IQA is optional; if clipiqa is installed, it will be used in no-reference mode on SR.
"""
import argparse
import csv
import re
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, LearnedPerceptualImagePatchSimilarity

try:
    from clipiqa_pytorch import CLIPIQANet
    _HAS_CLIP_IQA = True
except Exception:
    _HAS_CLIP_IQA = False
    CLIPIQANet = None

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False
    plt = None


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def infer_mask_name(fname: str) -> str:
    # extract substring after _mask and before optional _<idx>
    # examples: 0000_mask_circle_center_0.png -> circle_center
    m = re.search(r"_mask(.+?)(?:_[0-9]+)?\.\w+$", fname)
    return m.group(1) if m else "nomask"


def infer_hr_basename(fname: str) -> str:
    # strip mask suffix (with underscores) and trailing sample index
    stem = Path(fname).stem
    stem = re.sub(r"_mask.*$", "", stem)  # drop mask and following parts
    stem = re.sub(r"_[0-9]+$", "", stem)  # drop trailing index
    return stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_dir", required=True, help="Folder with SR outputs")
    parser.add_argument("--hr_dir", required=True, help="Folder with HR references")
    parser.add_argument("--out_csv", required=True, help="Where to save metrics CSV")
    parser.add_argument("--out_dir", default=None, help="If set, save CSV and plots under this directory")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--plots_prefix", default=None, help="If set, save summary plots with this prefix (e.g., metrics_plot)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)

    clip_iqa = None
    if _HAS_CLIP_IQA:
        clip_iqa = CLIPIQANet().to(device)
        clip_iqa.eval()

    sr_dir = Path(args.sr_dir)
    hr_dir = Path(args.hr_dir)
    rows = [("file", "mask", "psnr", "lpips", "clip_iqa")]

    for sr_path in sorted(sr_dir.glob("*.png")):
        base = infer_hr_basename(sr_path.name)
        hr_path = hr_dir / f"{base}.png"
        if not hr_path.exists():
            # try jpg fallback
            alt = hr_dir / f"{base}.jpg"
            hr_path = alt if alt.exists() else hr_path
        if not hr_path.exists():
            print(f"skip (no HR): {sr_path.name} -> {hr_path.name}")
            continue
        sr_pil = Image.open(sr_path).convert("RGB")
        hr_pil = Image.open(hr_path).convert("RGB")
        if hr_pil.size != sr_pil.size:
            hr_pil = hr_pil.resize(sr_pil.size, Image.BILINEAR)
        sr = pil_to_tensor(sr_pil).to(device)
        hr = pil_to_tensor(hr_pil).to(device)

        psnr_val = psnr(sr, hr).item()
        lpips_val = lpips(sr, hr).item()
        clip_val = ""
        if clip_iqa is not None:
            with torch.no_grad():
                clip_val = clip_iqa(sr).item()

        rows.append((sr_path.name, infer_mask_name(sr_path.name), psnr_val, lpips_val, clip_val))
        if len(rows) % 20 == 0:
            print(f"processed {len(rows)-1} images")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / Path(args.out_csv).name
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Saved metrics to {out_csv}")

    # optional plots
    if args.plots_prefix and _HAS_PLT:
        data = []
        for r in rows[1:]:
            _, mask, psnr_val, lpips_val, clip_val = r
            data.append((mask, float(psnr_val), float(lpips_val), float(clip_val) if clip_val != "" else math.nan))

        def agg(field_idx):
            agg_map = {}
            for mask, psnr_val, lpips_val, clip_val in data:
                val = (psnr_val, lpips_val, clip_val)[field_idx]
                if math.isnan(val):
                    continue
                agg_map.setdefault(mask, []).append(val)
            return {k: sum(v) / len(v) for k, v in agg_map.items() if v}

        psnr_mean = agg(0)
        lpips_mean = agg(1)
        clip_mean = agg(2)

        def bar_plot(values, title, ylabel, fname):
            masks = list(values.keys())
            vals = [values[m] for m in masks]
            plt.figure(figsize=(8, 4))
            plt.bar(masks, vals)
            plt.xticks(rotation=25, ha="right")
            plt.title(title)
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

        prefix = Path(args.plots_prefix)
        if not prefix.is_absolute():
            prefix = out_dir / prefix
        if psnr_mean:
            bar_plot(psnr_mean, "PSNR per mask", "PSNR", f"{prefix}_psnr.png")
        if lpips_mean:
            bar_plot(lpips_mean, "LPIPS per mask (lower better)", "LPIPS", f"{prefix}_lpips.png")
        if clip_mean:
            bar_plot(clip_mean, "CLIP-IQA per mask", "CLIP-IQA", f"{prefix}_clipiqa.png")
        print(f"Saved plots with prefix {prefix}")
    elif args.plots_prefix and not _HAS_PLT:
        print("matplotlib not available; skipping plots")


if __name__ == "__main__":
    main()
