"""Evaluate DepthSplat and DepthSplat+DiFix3D+ on target views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from .evaluation.metrics import compute_lpips, compute_psnr, compute_ssim


def load(path: Path, device: torch.device) -> torch.Tensor:
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def metrics(gt: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    return {
        "psnr": float(compute_psnr(gt, pred).mean().item()),
        "ssim": float(compute_ssim(gt, pred).mean().item()),
        "lpips": float(compute_lpips(gt, pred).mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--difix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default=None)
    args = parser.parse_args()

    image_root = args.render_root / "images"
    scene_dirs = ([image_root / args.scene] if args.scene else
                  sorted(path for path in image_root.iterdir() if path.is_dir()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_baseline = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_difix = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    rows = []
    for scene_dir in tqdm(scene_dirs, desc="Evaluate target views"):
        color_dir = scene_dir / "color"
        predictions = sorted(
            path for path in color_dir.glob("*.png")
            if not path.name.endswith("_gt.png")
        )
        if not predictions:
            continue
        prediction = predictions[0]
        gt = color_dir / f"{prediction.stem}_gt.png"
        difix = args.difix_root / scene_dir.name / "target.png"
        if not gt.exists() or not difix.exists():
            continue

        gt_tensor = load(gt, device)
        baseline_tensor = load(prediction, device)
        difix_tensor = load(difix, device)
        if gt_tensor.shape != baseline_tensor.shape or gt_tensor.shape != difix_tensor.shape:
            raise ValueError(f"Image shape mismatch in scene {scene_dir.name}")

        baseline_uint8 = (baseline_tensor.clamp(0, 1) * 255).to(torch.uint8)
        difix_uint8 = (difix_tensor.clamp(0, 1) * 255).to(torch.uint8)
        gt_uint8 = (gt_tensor.clamp(0, 1) * 255).to(torch.uint8)
        fid_baseline.update(gt_uint8, real=True)
        fid_baseline.update(baseline_uint8, real=False)
        fid_difix.update(gt_uint8, real=True)
        fid_difix.update(difix_uint8, real=False)

        rows.append({
            "scene": scene_dir.name,
            "depthsplat": metrics(gt_tensor, baseline_tensor),
            "depthsplat_difix3d": metrics(gt_tensor, difix_tensor),
        })

    if not rows:
        raise RuntimeError("No complete target prediction/GT/DiFix triplets found")

    summary = {}
    for method in ("depthsplat", "depthsplat_difix3d"):
        summary[method] = {
            key: float(np.mean([row[method][key] for row in rows]))
            for key in ("psnr", "ssim", "lpips")
        }
    try:
        summary["depthsplat"]["fid"] = float(fid_baseline.compute().item())
        summary["depthsplat_difix3d"]["fid"] = float(fid_difix.compute().item())
    except Exception as exc:
        print(f"FID computation failed: {exc}")
        summary["depthsplat"]["fid"] = None
        summary["depthsplat_difix3d"]["fid"] = None

    result = {"num_scenes": len(rows), "summary": summary, "scenes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"num_scenes": len(rows), "summary": summary}, indent=2))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
