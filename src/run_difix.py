"""Apply the official Difix model to the target renders produced by DepthSplat."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--difix-repo", type=Path, required=True)
    parser.add_argument("--prompt", default="remove degradation")
    parser.add_argument("--timestep", type=int, default=199)
    parser.add_argument("--scene")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = args.difix_repo / "src"
    sys.path.insert(0, str(src_root))
    from pipeline_difix import DifixPipeline  # noqa: E402

    files = sorted(args.input_root.rglob("*.png"))
    if args.scene:
        files = [p for p in files if p.relative_to(args.input_root).parts[0] == args.scene]
    if not files:
        raise FileNotFoundError(f"No PNG inputs under {args.input_root}")

    pipe = DifixPipeline.from_pretrained(str(args.model_path), trust_remote_code=True)
    pipe.to("cuda")
    for path in tqdm(files, desc="Difix"):
        out = args.output_root / path.relative_to(args.input_root)
        if out.exists() and not args.overwrite:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(path).convert("RGB")
        result = pipe(
            args.prompt,
            image=image,
            num_inference_steps=1,
            timesteps=[args.timestep],
            guidance_scale=0.0,
        ).images[0]
        result.save(out)


if __name__ == "__main__":
    main()
