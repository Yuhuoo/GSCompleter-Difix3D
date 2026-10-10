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
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--difix-root", type=Path)
    outputs.add_argument("--refined-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--expected-scenes", type=int, default=None)
    args = parser.parse_args()

    image_root = args.render_root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    scene_dirs = ([image_root / args.scene] if args.scene else
                  sorted(path for path in image_root.iterdir() if path.is_dir()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_baseline = (FrechetInceptionDistance(feature=2048, normalize=True).to(device)
                    if args.scene is None else None)
    fid_difix = (FrechetInceptionDistance(feature=2048, normalize=True).to(device)
                 if args.scene is None else None)
    method_name = "depthsplat_difix3d_distilled" if args.refined_root else "depthsplat_difix3d"

    rows = []
    for scene_dir in tqdm(scene_dirs, desc="Evaluate target views"):
        color_dir = scene_dir / "color"
        predictions = sorted(
            path for path in color_dir.glob("*.png")
            if not path.name.endswith("_gt.png")
        )
        if len(predictions) != 1:
            raise RuntimeError(f"Expected one target render in {color_dir}, found {len(predictions)}")
        prediction = predictions[0]
        gt = color_dir / f"{prediction.stem}_gt.png"
        difix = (args.refined_root / "images" / scene_dir.name / "color" / prediction.name
                 if args.refined_root else args.difix_root / scene_dir.name / "target.png")
        if not gt.exists() or not difix.exists():
            raise FileNotFoundError(f"Incomplete evaluation triplet: {prediction}, {gt}, {difix}")

        gt_tensor = load(gt, device)
        baseline_tensor = load(prediction, device)
        difix_tensor = load(difix, device)
        if gt_tensor.shape != baseline_tensor.shape or gt_tensor.shape != difix_tensor.shape:
            raise ValueError(f"Image shape mismatch in scene {scene_dir.name}")

        if fid_baseline is not None and fid_difix is not None:
            fid_baseline.update(gt_tensor, real=True)
            fid_baseline.update(baseline_tensor, real=False)
            fid_difix.update(gt_tensor, real=True)
            fid_difix.update(difix_tensor, real=False)

        rows.append({
            "scene": scene_dir.name,
            "depthsplat": metrics(gt_tensor, baseline_tensor),
            method_name: metrics(gt_tensor, difix_tensor),
        })

    if not rows:
        raise RuntimeError("No complete target prediction/GT/DiFix triplets found")
    if args.expected_scenes is not None and len(rows) != args.expected_scenes:
        raise RuntimeError(f"Expected {args.expected_scenes} scenes, found {len(rows)}")

    summary = {}
    for method in ("depthsplat", method_name):
        summary[method] = {
            key: float(np.mean([row[method][key] for row in rows]))
            for key in ("psnr", "ssim", "lpips")
        }
    if len(rows) < 2 or fid_baseline is None or fid_difix is None:
        summary["depthsplat"]["fid"] = None
        summary[method_name]["fid"] = None
    else:
        summary["depthsplat"]["fid"] = float(fid_baseline.compute().item())
        summary[method_name]["fid"] = float(fid_difix.compute().item())

    result = {"num_scenes": len(rows), "summary": summary, "scenes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"num_scenes": len(rows), "summary": summary}, indent=2))
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main()
