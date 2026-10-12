"""One-shot DiFix3D-style pseudo-GT distillation from DepthSplat."""

from __future__ import annotations

import os
import json
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


def _renderable_gaussians(
    gaussians: Gaussians,
    camera: dict,
    shape: tuple[int, int],
    max_gaussians_override: int | None = None,
) -> Gaussians:
    """Filter pathological Gaussians before calling the CUDA rasterizer."""
    if gaussians.means.ndim != 3 or gaussians.means.shape[0] != 1:
        raise ValueError(
            f"Expected single-scene Gaussians [1,G,*], got "
            f"means={tuple(gaussians.means.shape)}"
        )

    h, w = shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid render shape: {shape}")

    extrinsics = camera["extrinsics"][0, 0]
    intrinsics = camera["intrinsics"][0, 0]
    near = camera["near"][0, 0].detach().float()
    far = camera["far"][0, 0].detach().float()
    near_value = near.item()
    far_value = far.item()
    if not torch.isfinite(near).item() or not torch.isfinite(far).item() or not (0 < near_value < far_value):
        raise ValueError(f"Invalid render bounds: near={near.item()}, far={far.item()}")

    points = gaussians.means[0]
    covariance = gaussians.covariances[0]

    # The mask is discrete, so camera-space checks do not need an autograd
    # graph. Indexing the original tensors below still preserves the graph
    # needed by the renderer during distillation.
    with torch.no_grad():
        points_h = torch.cat([points.detach(), torch.ones_like(points[..., :1])], dim=-1)
        camera_points = (torch.linalg.inv(extrinsics) @ points_h.t()).t()[..., :3]
        depth = camera_points[..., 2]
        projected = camera_points @ intrinsics.transpose(0, 1)
        safe_depth = depth.clamp_min(1e-6)
        uv = projected[..., :2] / safe_depth[..., None]

        finite = (
            torch.isfinite(points).all(dim=-1)
            & torch.isfinite(covariance).all(dim=(-1, -2))
            & torch.isfinite(gaussians.harmonics[0]).all(dim=(-1, -2))
            & torch.isfinite(gaussians.opacities[0])
            & torch.isfinite(camera_points).all(dim=-1)
            & torch.isfinite(uv).all(dim=-1)
        )

        # Compute the actual first-order 2-D projected covariance. Checking
        # only world-space diagonal entries misses long tilted ellipses and
        # Gaussians whose depth variance crosses the near plane; both cases
        # can make the rasterizer allocate an enormous tile buffer.
        world_to_camera = torch.linalg.inv(extrinsics)[:3, :3]
        cov_camera = torch.einsum(
            "ij,gjk,lk->gil",
            world_to_camera,
            covariance.detach(),
            world_to_camera,
        )
        cov_camera = 0.5 * (cov_camera + cov_camera.transpose(-1, -2))
        covariance_eigenvalues = torch.linalg.eigvalsh(cov_camera)
        depth_variance = cov_camera[..., 2, 2].clamp_min(0)
        depth_sigma = depth_variance.sqrt()

        # Intrinsics are normalized by the dataset; convert the Jacobian to
        # pixel units before estimating the footprint.
        fx = intrinsics[0, 0].abs() * w
        fy = intrinsics[1, 1].abs() * h
        x = camera_points[..., 0]
        y = camera_points[..., 1]
        z2 = safe_depth.square()
        j00 = fx / safe_depth
        j02 = -fx * x / z2
        j11 = fy / safe_depth
        j12 = -fy * y / z2
        c00 = j00.square() * cov_camera[..., 0, 0] + 2 * j00 * j02 * cov_camera[..., 0, 2] + j02.square() * cov_camera[..., 2, 2]
        c11 = j11.square() * cov_camera[..., 1, 1] + 2 * j11 * j12 * cov_camera[..., 1, 2] + j12.square() * cov_camera[..., 2, 2]
        c01 = j00 * j11 * cov_camera[..., 0, 1] + j00 * j12 * cov_camera[..., 0, 2] + j02 * j11 * cov_camera[..., 1, 2] + j02 * j12 * cov_camera[..., 2, 2]
        eig_gap = torch.sqrt((c00 - c11).square() + 4 * c01.square())
        radius_pixels = 3.0 * (0.5 * (c00 + c11 + eig_gap)).clamp_min(0).sqrt()
        finite = finite & torch.isfinite(cov_camera).all(dim=(-1, -2))
        finite = finite & torch.isfinite(covariance_eigenvalues).all(dim=-1)
        finite = finite & (covariance_eigenvalues[..., 0] >= -1e-6)
        finite = finite & torch.isfinite(radius_pixels) & torch.isfinite(depth_sigma)
    max_radius = float(os.environ.get("DIFIX_MAX_RENDER_GAUSSIAN_RADIUS", "64"))
    if max_radius <= 0:
        raise ValueError("DIFIX_MAX_RENDER_GAUSSIAN_RADIUS must be positive")

    visible = (
        finite
        & (depth - 3.0 * depth_sigma > near * 0.5)
        & (depth < far * 1.05)
        & (uv[..., 0] >= -1.0)
        & (uv[..., 0] <= 2.0)
        & (uv[..., 1] >= -1.0)
        & (uv[..., 1] <= 2.0)
        & (radius_pixels <= max_radius)
    )

    max_gaussians = (
        int(os.environ.get("DIFIX_MAX_RENDER_GAUSSIANS", "120000"))
        if max_gaussians_override is None
        else int(max_gaussians_override)
    )
    if max_gaussians <= 0:
        raise ValueError("DIFIX_MAX_RENDER_GAUSSIANS must be positive")

    # The encoder emits one Gaussian per pixel (229,376 at 256x448). Cap the
    # active set because rasterizer tile buffers grow with this count. Opaque,
    # compact footprints carry the most visible signal.
    candidate_ids = torch.where(visible)[0]
    if candidate_ids.numel() > max_gaussians:
        opacity = gaussians.opacities[0].detach().clamp_min(1e-6)
        score = opacity / (1.0 + radius_pixels.detach())
        keep_ids = candidate_ids[
            torch.topk(score[candidate_ids], k=max_gaussians, largest=True).indices
        ]
        visible = torch.zeros_like(visible)
        visible[keep_ids] = True

    # Keep a small valid set even for an unusually sparse view, choosing the
    # smallest projected footprints rather than reintroducing invalid points.
    min_keep = min(1024, max_gaussians, points.shape[0])
    if int(visible.sum()) < min_keep:
        fallback = (
            finite
            & (depth - 3.0 * depth_sigma > near * 0.5)
            & (depth < far * 1.05)
            & (radius_pixels <= max_radius)
        )
        fallback_ids = torch.where(fallback)[0]
        if fallback_ids.numel() > 0:
            count = min(min_keep, int(fallback_ids.numel()))
            chosen = fallback_ids[torch.topk(
                radius_pixels[fallback_ids].nan_to_num(float("inf")),
                k=count,
                largest=False,
            ).indices]
            visible = torch.zeros_like(finite)
            visible[chosen] = True

    if not visible.any():
        raise RuntimeError(
            "No finite Gaussians remain for DiFix rendering; "
            "enable DIFIX_DEBUG_RENDER=1 to inspect the camera/depth bounds."
        )

    if os.environ.get("DIFIX_DEBUG_RENDER", "0") == "1":
        finite_radius = radius_pixels[finite].nan_to_num()
        visible_radius = radius_pixels[visible].nan_to_num()
        print(
            f"[DiFix render] shape={shape}, gaussians={points.shape[0]} -> "
            f"{int(visible.sum())}, depth=[{depth[finite].min().item():.4g}, "
            f"{depth[finite].max().item():.4g}], "
            f"radius_max={finite_radius.max().item():.4g}, "
            f"visible_radius_max={visible_radius.max().item():.4g}"
        )

    return Gaussians(
        means=gaussians.means[:, visible],
        covariances=gaussians.covariances[:, visible],
        harmonics=gaussians.harmonics[:, visible],
        opacities=gaussians.opacities[:, visible],
    )


def render(decoder, gaussians: Gaussians, camera: dict, shape: tuple[int, int]) -> torch.Tensor:
    max_gaussians = int(os.environ.get("DIFIX_MAX_RENDER_GAUSSIANS", "120000"))
    for attempt in range(3):
        renderable = _renderable_gaussians(
            gaussians,
            camera,
            shape,
            max_gaussians_override=max_gaussians,
        )
        try:
            return decoder(
                renderable,
                camera["extrinsics"],
                camera["intrinsics"],
                camera["near"],
                camera["far"],
                shape,
            ).color[0, 0]
        except torch.cuda.OutOfMemoryError:
            del renderable
            torch.cuda.empty_cache()
            if attempt == 2 or max_gaussians <= 8192:
                raise
            max_gaussians = max(8192, max_gaussians // 2)
            if os.environ.get("DIFIX_DEBUG_RENDER", "0") == "1":
                print(f"[DiFix render] rasterizer OOM; retrying with {max_gaussians} Gaussians")


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
    ]


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


class TrainableGaussians:
    """Trainable context Gaussians with lightweight topology updates.

    The official gsplat strategy cannot be used directly because this branch
    renders with DepthSplat's decoder rather than gsplat's rasterizer.  This
    class mirrors the relevant behavior: high-gradient visible points are
    cloned or split into smaller copies, while nearly transparent points are
    pruned.
    """

    def __init__(self, initial: Gaussians):
        # The DepthSplat context representation is the only 3D state that is
        # optimized.  DiFix supplies a fixed pseudo-GT target image; no target
        # view is recursively generated during this optimization.
        self.initial_covariances = initial.covariances.detach().clone()
        self.means = nn.Parameter(initial.means.detach().clone())
        self.harmonics = nn.Parameter(initial.harmonics.detach().clone())
        self.opacity_logits = nn.Parameter(
            torch.logit(initial.opacities.detach().clamp(1e-4, 1 - 1e-4)))
        # One isotropic multiplier per Gaussian keeps the covariance update
        # stable while still allowing the repaired view to change support.
        self.log_scales = nn.Parameter(torch.zeros_like(initial.opacities))
        self.optimizer = None
        self.rebuild_optimizer()

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[1])

    def current(self) -> Gaussians:
        scale = self.log_scales.exp()
        covariances = self.initial_covariances * scale[..., None, None] ** 2
        return Gaussians(
            means=self.means,
            covariances=covariances,
            harmonics=self.harmonics,
            opacities=self.opacity_logits.sigmoid(),
        )

    def rebuild_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam([
            {"params": [self.means], "lr": 1.6e-4},
            {"params": [self.harmonics], "lr": 2.5e-3},
            {"params": [self.opacity_logits], "lr": 1e-3},
            {"params": [self.log_scales], "lr": 5e-4},
        ])

    def _clone(self, indices: torch.Tensor) -> int:
        if indices.numel() == 0:
            return 0
        # Clone selected high-gradient points at the same location.  Splitting
        # below handles the complementary case where a spatial offset helps.
        clone_opacity = (
            self.opacity_logits.detach()[:, indices].sigmoid() * 0.5
        ).clamp(1e-4, 1 - 1e-4)
        clone_logits = torch.logit(clone_opacity)
        updated_opacity = self.opacity_logits.detach().clone()
        updated_opacity[:, indices] = clone_logits
        self.means = nn.Parameter(torch.cat([
            self.means.detach(), self.means.detach()[:, indices].clone()
        ], dim=1))
        self.harmonics = nn.Parameter(torch.cat([
            self.harmonics.detach(), self.harmonics.detach()[:, indices].clone()
        ], dim=1))
        self.opacity_logits = nn.Parameter(torch.cat([
            updated_opacity, clone_logits
        ], dim=1))
        self.log_scales = nn.Parameter(torch.cat([
            self.log_scales.detach(), self.log_scales.detach()[:, indices].clone()
        ], dim=1))
        self.initial_covariances = torch.cat([
            self.initial_covariances, self.initial_covariances[:, indices]
        ], dim=1)
        return int(indices.numel())

    def _append(self, indices: torch.Tensor) -> int:
        if indices.numel() == 0:
            return 0
        # Split selected points into nearby smaller copies.  Estimate a local
        # world-space radius from the covariance so the offset is scene-scale
        # aware across DL3DV scenes.
        base_means = self.means.detach()[:, indices]
        base_covariances = self.initial_covariances[:, indices]
        base_scale = base_covariances.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8).sqrt()
        base_scale = base_scale * self.log_scales.detach()[:, indices].exp()
        base_scale = base_scale[..., None]
        noise = torch.randn_like(base_means) * base_scale * 0.35
        new_means = base_means + noise
        new_harmonics = self.harmonics.detach()[:, indices].clone()
        # Preserve approximately the same accumulated opacity after splitting.
        old_opacity = self.opacity_logits.detach()[:, indices].sigmoid()
        split_opacity = (old_opacity * 0.5).clamp(1e-4, 1 - 1e-4)
        split_logits = torch.logit(split_opacity)
        updated_opacity = self.opacity_logits.detach().clone()
        updated_opacity[:, indices] = split_logits
        new_opacity = split_logits.clone()
        new_log_scales = self.log_scales.detach()[:, indices] - 0.5

        self.means = nn.Parameter(torch.cat([self.means.detach(), new_means], dim=1))
        self.harmonics = nn.Parameter(
            torch.cat([self.harmonics.detach(), new_harmonics], dim=1))
        self.opacity_logits = nn.Parameter(torch.cat([updated_opacity, new_opacity], dim=1))
        self.log_scales = nn.Parameter(
            torch.cat([self.log_scales.detach(), new_log_scales], dim=1))
        self.initial_covariances = torch.cat(
            [self.initial_covariances, self.initial_covariances[:, indices]], dim=1)
        return int(indices.numel())

    def _prune(self, opacity_threshold: float, min_gaussians: int) -> int:
        if self.num_gaussians <= min_gaussians:
            return 0
        keep = self.opacity_logits.sigmoid().squeeze(0) >= opacity_threshold
        if int(keep.sum()) < min_gaussians:
            topk = torch.topk(
                self.opacity_logits.sigmoid().squeeze(0),
                k=min(min_gaussians, self.num_gaussians),
            ).indices
            keep = torch.zeros_like(keep)
            keep[topk] = True
        removed = self.num_gaussians - int(keep.sum())
        if removed <= 0:
            return 0
        keep_b = keep.unsqueeze(0)
        self.means = nn.Parameter(self.means.detach()[:, keep])
        self.harmonics = nn.Parameter(self.harmonics.detach()[:, keep])
        self.opacity_logits = nn.Parameter(self.opacity_logits.detach()[:, keep])
        self.log_scales = nn.Parameter(self.log_scales.detach()[:, keep])
        self.initial_covariances = self.initial_covariances[:, keep_b.squeeze(0)]
        return removed

    def densify_and_prune(
        self,
        grad_scores: torch.Tensor,
        visible: torch.Tensor,
        *,
        grad_threshold: float,
        max_new: int,
        max_gaussians: int,
        opacity_threshold: float,
        min_gaussians: int,
    ) -> tuple[int, int, int]:
        grad_scores = torch.nan_to_num(grad_scores.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        visible = visible.detach().bool()
        candidate_mask = visible & (grad_scores >= grad_threshold)
        candidate_ids = torch.where(candidate_mask)[0]
        available = max(0, max_gaussians - self.num_gaussians)
        num_to_add = min(int(candidate_ids.numel()), max_new, available)
        cloned = 0
        split = 0
        if num_to_add > 0:
            if candidate_ids.numel() > num_to_add:
                candidate_ids = candidate_ids[
                    torch.topk(grad_scores[candidate_ids], k=num_to_add).indices]
            clone_count = num_to_add // 2
            cloned = self._clone(candidate_ids[:clone_count])
            split = self._append(candidate_ids[clone_count:])
        removed = self._prune(opacity_threshold, min_gaussians)
        if cloned or split or removed:
            self.rebuild_optimizer()
        return cloned, split, removed


@torch.no_grad()
def visible_gaussians(
    means: torch.Tensor,
    camera: dict,
    margin: float = 0.05,
) -> torch.Tensor:
    """Return a conservative frustum mask for one target camera.

    DepthSplat's decoder returns only color/depth, unlike gsplat's rasterizer,
    so there is no per-Gaussian radii tensor to use for densification.  The
    frustum test keeps topology updates limited to points that can contribute
    to the current view while avoiding a dependency on another rasterizer.
    """
    # The evaluation camera convention is camera-to-world.  The normalized
    # intrinsics therefore map camera-space points to coordinates in [0, 1].
    extrinsics = camera["extrinsics"][0, 0]
    intrinsics = camera["intrinsics"][0, 0]
    near = camera["near"][0, 0]
    far = camera["far"][0, 0]
    points = means[0]
    points_h = torch.cat(
        [points, torch.ones_like(points[..., :1])], dim=-1)
    camera_points = (torch.linalg.inv(extrinsics) @ points_h.t()).t()[..., :3]
    depth = camera_points[..., 2]
    projected = camera_points @ intrinsics.transpose(0, 1)
    safe_depth = depth.clamp_min(1e-6)
    uv = projected[..., :2] / safe_depth[..., None]
    return (
        (depth > near.clamp_min(1e-5))
        & (depth < far)
        & (uv[..., 0] >= -margin)
        & (uv[..., 0] <= 1.0 + margin)
        & (uv[..., 1] >= -margin)
        & (uv[..., 1] <= 1.0 + margin)
    )


def distill_scene(batch: dict, encoder, decoder, root: Path, steps: int) -> None:
    scene = batch["scene"][0]
    fixed_path = root / "target_fixed" / scene / "target.png"
    shape = tuple(batch["target"]["image"].shape[-2:])
    device = batch["context"]["image"].device
    if not fixed_path.is_file():
        raise FileNotFoundError(f"Missing DiFix target repair: {fixed_path}")
    repaired_target = load_image(fixed_path).to(device)

    context_gaussians = initial_gaussians(encoder, batch)
    state = TrainableGaussians(context_gaussians)
    initial_means = state.means.detach().clone()
    initial_harmonics = state.harmonics.detach().clone()
    densify_enabled = os.environ.get("DIFIX_DENSIFY", "1") == "1"
    densify_interval = int(os.environ.get("DIFIX_DENSIFY_INTERVAL", "250"))
    densify_threshold = float(os.environ.get("DIFIX_DENSIFY_GRAD_THRESHOLD", "2e-4"))
    max_new = int(os.environ.get("DIFIX_MAX_NEW_GAUSSIANS", "4096"))
    max_gaussians = int(os.environ.get("DIFIX_MAX_GAUSSIANS", "300000"))
    opacity_threshold = float(os.environ.get("DIFIX_PRUNE_OPACITY", "0.005"))
    min_gaussians = int(os.environ.get("DIFIX_MIN_GAUSSIANS", "1024"))
    context_loss_weight = float(os.environ.get("DIFIX_CONTEXT_LOSS_WEIGHT", "0.1"))
    if densify_interval < 1:
        raise ValueError("DIFIX_DENSIFY_INTERVAL must be positive")
    if context_loss_weight < 0:
        raise ValueError("DIFIX_CONTEXT_LOSS_WEIGHT must be non-negative")

    target_camera = {
        key: batch["target"][key]
        for key in ("extrinsics", "intrinsics", "near", "far")
    }
    context_camera = {
        key: batch["context"][key]
        for key in ("extrinsics", "intrinsics", "near", "far")
    }
    context_labels = batch["context"]["image"]
    topology_stats = [{"step": 0, "num_gaussians": state.num_gaussians}]
    for step in tqdm(range(steps), desc=f"Distill {scene[:12]}", leave=False):
        # One fixed pseudo-GT target drives every iteration.  The observed
        # context views are replayed only as a small preservation term; this
        # is not a progressive or autoregressive view-generation procedure.
        state.optimizer.zero_grad(set_to_none=True)

        # Backpropagate each view separately. Keeping target and both context
        # rasterizer graphs alive until one combined backward can exceed GPU
        # memory even when each individual render fits.
        target_prediction = render(decoder, state.current(), target_camera, shape)
        F.l1_loss(target_prediction, repaired_target).backward()
        del target_prediction

        if context_loss_weight > 0:
            num_context_views = context_camera["extrinsics"].shape[1]
            for view_idx in range(num_context_views):
                single_camera = {
                    key: value[:, view_idx:view_idx + 1]
                    for key, value in context_camera.items()
                }
                context_prediction = render(
                    decoder, state.current(), single_camera, shape
                )
                context_label = context_labels[0, view_idx]
                (context_loss_weight / num_context_views * F.l1_loss(
                    context_prediction, context_label
                )).backward()
                del context_prediction

        regularization = (
            1e-3 * (state.means - initial_means).square().mean()
            + 1e-4 * (state.harmonics - initial_harmonics).square().mean()
            + 1e-4 * state.log_scales.square().mean()
        )
        regularization.backward()
        grad_scores = (
            state.means.grad.detach().norm(dim=-1).squeeze(0)
            if state.means.grad is not None
            else torch.zeros(state.num_gaussians, device=device)
        )
        visible = visible_gaussians(state.means, target_camera)
        state.optimizer.step()

        if (
            densify_enabled
            and densify_interval > 0
            and (step + 1) % densify_interval == 0
        ):
            cloned, split, removed = state.densify_and_prune(
                grad_scores,
                visible,
                grad_threshold=densify_threshold,
                max_new=max_new,
                max_gaussians=max_gaussians,
                opacity_threshold=opacity_threshold,
                min_gaussians=min_gaussians,
            )
            topology_stats.append({
                "step": step + 1,
                "cloned": cloned,
                "split": split,
                "added": cloned + split,
                "removed": removed,
                "num_gaussians": state.num_gaussians,
            })
            if cloned or split or removed:
                # Re-anchor the small stabilization terms to the new topology
                # after parameters have been added or removed.
                initial_means = state.means.detach().clone()
                initial_harmonics = state.harmonics.detach().clone()
                tqdm.write(
                    f"[{scene[:12]}] topology step {step + 1}: "
                    f"clone={cloned}, split={split}, prune={removed}, "
                    f"N={state.num_gaussians}")

    target_id = int(batch["target"]["index"][0, 0])
    target = {key: batch["target"][key] for key in
              ("extrinsics", "intrinsics", "near", "far")}
    with torch.no_grad():
        image = render(decoder, state.current(), target, shape)
    save_image(image, root / "distilled" / "images" / scene / "color" / f"{target_id:06d}.png")
    stats_path = root / "distilled" / "stats" / f"{scene}.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({
        "scene": scene,
        "steps": steps,
        "densify_enabled": densify_enabled,
        "densify_interval": densify_interval,
        "context_loss_weight": context_loss_weight,
        "supervision": "fixed_difix_pseudo_gt_target",
        "initial_gaussians": topology_stats[0]["num_gaussians"],
        "final_gaussians": state.num_gaussians,
        "topology": topology_stats,
    }, indent=2), encoding="utf-8")
    del state


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
