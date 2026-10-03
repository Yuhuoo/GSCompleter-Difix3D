"""Stage DepthSplat target renders for the DiFix3D+ post-processing step."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--scene", default=None)
    args = parser.parse_args()

    image_root = args.render_root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing DepthSplat images: {image_root}")

    scene_dirs = ([image_root / args.scene] if args.scene else
                  sorted(path for path in image_root.iterdir() if path.is_dir()))
    copied = 0
    for scene_dir in scene_dirs:
        color_dir = scene_dir / "color"
        predictions = sorted(
            path for path in color_dir.glob("*.png")
            if not path.name.endswith("_gt.png")
        )
        if not predictions:
            continue
        # The evaluation index contains one target view per scene.
        source = predictions[0]
        output_dir = args.input_root / scene_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output_dir / "target.png")
        copied += 1

    if copied == 0:
        raise RuntimeError("No DepthSplat target renders were found")
    print(f"Staged {copied} target renders in {args.input_root}")


if __name__ == "__main__":
    main()
