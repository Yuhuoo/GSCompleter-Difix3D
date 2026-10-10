"""DepthSplat target-view repair followed by Gaussian distillation."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from .config import load_typed_root_config
from .dataset.data_module import DataModule, get_data_shim
from .global_cfg import set_cfg
from .misc.image_io import load_image, save_image
from .model.decoder import get_decoder
from .model.encoder import get_encoder
from .model.types import Gaussians


def render(decoder, gaussians: Gaussians, camera: dict, shape: tuple[int, int]) -> torch.Tensor:
    return decoder(
        gaussians, camera["extrinsics"], camera["intrinsics"],
        camera["near"], camera["far"], shape,
    ).color[0, 0]


def initialize(cfg):
    device = torch.device("cuda")
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    checkpoint = torch.load(cfg.checkpointing.pretrained_model, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    encoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("encoder.")}, strict=True)
    decoder.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("decoder.")}, strict=True)
    encoder = encoder.eval().requires_grad_(False)
    decoder = decoder.to(device).eval().requires_grad_(False)
    loader = DataModule(cfg.dataset, cfg.data_loader).test_dataloader()
    return device, encoder, decoder, get_data_shim(encoder), loader


def to_device(batch: dict, device: torch.device, shim) -> dict:
    batch = shim(batch)
    for role in ("context", "target"):
        batch[role] = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                       for key, value in batch[role].items()}
    return batch


def initial_gaussians(encoder, batch: dict) -> Gaussians:
    device = batch["context"]["image"].device
    encoder.to(device)
    try:
        with torch.no_grad():
            result = encoder(batch["context"], global_step=0, deterministic=False)
            gaussians = result["gaussians"] if isinstance(result, dict) else result
    finally:
        encoder.cpu()
        torch.cuda.empty_cache()
    if gaussians.means.shape[0] != 1:
        raise ValueError("DiFix3D distillation requires one Gaussian scene per batch")
    return gaussians


def expected_outputs(root: Path, scene: str, target_id: int, phase: str) -> list[Path]:
    if phase == "distill":
        return [root / "distilled" / "images" / scene / "color" / f"{target_id:06d}.png"]
    color = root / "depthsplat" / "images" / scene / "color"
    target_input = root / "target_input" / scene
    return [
        color / f"{target_id:06d}.png",
        color / f"{target_id:06d}_gt.png",
        target_input / "target.png",
        target_input / "reference.png",
    ]


def nearest_context_index(context_poses: torch.Tensor, target_pose: torch.Tensor) -> int:
    """Use the closest input camera as the DiFix reference image."""
    distances = torch.linalg.vector_norm(
        context_poses[:, :3, 3] - target_pose[:3, 3], dim=-1
    )
    return int(distances.argmin().item())


def prepare_scene(batch: dict, encoder, decoder, root: Path) -> None:
    scene = batch["scene"][0]
    image_dir = root / "depthsplat" / "images" / scene / "color"
    target_dir = root / "target_input" / scene
    target_id = int(batch["target"]["index"][0, 0])
    shape = tuple(batch["target"]["image"].shape[-2:])
    with torch.no_grad():
        gaussians = initial_gaussians(encoder, batch)
        target = {key: batch["target"][key] for key in
                  ("extrinsics", "intrinsics", "near", "far")}
        target_render = render(decoder, gaussians, target, shape)
        save_image(target_render, image_dir / f"{target_id:06d}.png")
        save_image(batch["target"]["image"][0, 0], image_dir / f"{target_id:06d}_gt.png")
        target_dir.mkdir(parents=True, exist_ok=True)
        save_image(target_render, target_dir / "target.png")
        target_pose = batch["target"]["extrinsics"][0, 0]
        context_poses = batch["context"]["extrinsics"][0]
        ref_idx = nearest_context_index(context_poses, target_pose)
        save_image(batch["context"]["image"][0, ref_idx], target_dir / "reference.png")


def trainable_gaussians(initial: Gaussians):
    means = nn.Parameter(initial.means.detach().clone())
    sh = nn.Parameter(initial.harmonics.detach().clone())
    opacity_logits = nn.Parameter(torch.logit(initial.opacities.detach().clamp(1e-4, 1 - 1e-4)))
    log_scales = nn.Parameter(torch.zeros_like(initial.opacities))

    def current():
        scale_squared = (2 * log_scales).exp()[..., None, None]
        return Gaussians(means, initial.covariances.detach() * scale_squared,
                         sh, opacity_logits.sigmoid())

    optimizer = torch.optim.Adam([
        {"params": [means], "lr": 1.6e-4},
        {"params": [sh], "lr": 2.5e-3},
        {"params": [opacity_logits], "lr": 1e-3},
        {"params": [log_scales], "lr": 5e-4},
    ])
    return current, optimizer, (means, sh, opacity_logits, log_scales)


def distill_scene(batch: dict, encoder, decoder, root: Path, steps: int) -> None:
    scene = batch["scene"][0]
    fixed_path = root / "target_fixed" / scene / "target.png"
    shape = tuple(batch["target"]["image"].shape[-2:])
    device = batch["context"]["image"].device
    if not fixed_path.is_file():
        raise FileNotFoundError(f"Missing DiFix target repair: {fixed_path}")
    repaired_target = load_image(fixed_path).to(device)

    initial = initial_gaussians(encoder, batch)
    current, optimizer, params = trainable_gaussians(initial)
    initial_values = tuple(param.detach().clone() for param in params)
    for step in tqdm(range(steps), desc=f"Distill {scene[:12]}", leave=False):
        # Keep most updates anchored to the real input views and use the
        # repaired held-out target as the completion supervision.
        if step % 10 < 7:
            index = (step // 10) % 2
            camera = {key: batch["context"][key][:, index:index + 1]
                      for key in ("extrinsics", "intrinsics", "near", "far")}
            label = batch["context"]["image"][0, index]
        else:
            camera = {key: batch["target"][key]
                      for key in ("extrinsics", "intrinsics", "near", "far")}
            label = repaired_target[0]

        prediction = render(decoder, current(), camera, shape)
        loss = F.l1_loss(prediction, label)
        loss = loss + 1e-3 * (params[0] - initial_values[0]).square().mean()
        loss = loss + 1e-4 * (params[1] - initial_values[1]).square().mean()
        loss = loss + 1e-4 * params[3].square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    target_id = int(batch["target"]["index"][0, 0])
    target = {key: batch["target"][key] for key in
              ("extrinsics", "intrinsics", "near", "far")}
    with torch.no_grad():
        image = render(decoder, current(), target, shape)
    save_image(image, root / "distilled" / "images" / scene / "color" / f"{target_id:06d}.png")
    del optimizer, current, params, initial_values


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    phase = os.environ.get("DIFIX_PHASE", "prepare")
    if phase not in ("prepare", "distill"):
        raise ValueError(f"Unknown DIFIX_PHASE: {phase}")
    root = Path(os.environ.get("DIFIX_ROOT", "outputs/dl3dv_difix_distill"))
    steps = int(os.environ.get("DIFIX_STEPS", "1000"))
    overwrite = os.environ.get("DIFIX_OVERWRITE", "0") == "1"
    only_scene = os.environ.get("DIFIX_SCENE")
    if steps <= 0:
        raise ValueError("DIFIX_STEPS must be positive")
    cfg_dict.mode = "test"
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)
    device, encoder, decoder, shim, loader = initialize(cfg)
    count = 0
    found_scene = False
    for raw_batch in loader:
        scene = raw_batch["scene"][0]
        if only_scene and scene != only_scene:
            continue
        found_scene = True
        target_id = int(raw_batch["target"]["index"][0, 0])
        if not overwrite and all(path.is_file() for path in expected_outputs(root, scene, target_id, phase)):
            continue
        batch = to_device(raw_batch, device, shim)
        if phase == "prepare":
            prepare_scene(batch, encoder, decoder, root)
        else:
            distill_scene(batch, encoder, decoder, root, steps)
        count += 1
    if only_scene and not found_scene:
        raise ValueError(f"Scene {only_scene} was not found in the evaluation index")
    print(f"{phase}: processed {count} scenes")


if __name__ == "__main__":
    main()
