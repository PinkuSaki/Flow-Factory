# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared WD ConvNeXt perceptual reward utilities."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

DEFAULT_STAGE_WEIGHTS = (0.25, 0.40, 0.22, 0.10, 0.03)


@dataclass(frozen=True)
class ConvNextPreprocessConfig:
    """Store the WD ConvNeXt image preprocessing parameters."""

    image_size: int
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]


@dataclass(frozen=True)
class ConvNextPreprocessedImageCache:
    """Prompt-hash indexed preprocessed reference images."""

    prompt_hashes: list[str]
    images: torch.Tensor
    image_size: int
    metadata: dict[str, Any]
    index_by_hash: dict[str, int]

    def get_images(self, prompt_hashes: list[str]) -> torch.Tensor:
        """Return preprocessed images aligned to the requested prompt hashes."""
        indices = [self.index_by_hash[prompt_hash] for prompt_hash in prompt_hashes]
        return self.images[indices]

    def __len__(self) -> int:
        """Return the number of cached reference images."""
        return len(self.prompt_hashes)


def parse_stage_weights(raw_weights: str) -> tuple[float, float, float, float, float]:
    """Parse a comma-separated five-stage weight string."""
    values = tuple(float(value.strip()) for value in raw_weights.split(",") if value.strip())
    if len(values) != 5:
        raise ValueError(f"stage_weights must contain 5 values, got {len(values)}: {values}")
    if any(value < 0.0 for value in values):
        raise ValueError(f"stage_weights must be non-negative, got {values}")
    if sum(values) <= 0.0:
        raise ValueError(f"stage_weights must have a positive sum, got {values}")
    return values  # type: ignore[return-value]


def load_convnext_preprocess_config(model_path: Path) -> ConvNextPreprocessConfig:
    """Load image preprocessing parameters from a local WD ConvNeXt config."""
    config_path = model_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    pretrained_cfg = model_config.get("pretrained_cfg", {})
    image_size = int(pretrained_cfg.get("input_size", [3, 448, 448])[-1])
    image_mean = tuple(float(value) for value in pretrained_cfg.get("mean", [0.5, 0.5, 0.5]))
    image_std = tuple(float(value) for value in pretrained_cfg.get("std", [0.5, 0.5, 0.5]))
    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError(
            f"Expected 3-channel mean/std in {config_path}, got mean={image_mean} std={image_std}."
        )

    return ConvNextPreprocessConfig(
        image_size=image_size,
        image_mean=image_mean,  # type: ignore[arg-type]
        image_std=image_std,  # type: ignore[arg-type]
    )


def preprocess_image_to_uint8(image: Image.Image, image_size: int) -> torch.Tensor:
    """Pad, resize, and convert one image into a CHW uint8 tensor."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    canvas = canvas.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(canvas, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def preprocess_images_to_uint8(images: list[Image.Image], image_size: int) -> torch.Tensor:
    """Convert images into a batch of CHW uint8 tensors."""
    if not images:
        raise ValueError("At least one image is required.")
    return torch.stack([preprocess_image_to_uint8(image, image_size) for image in images], dim=0)


def save_preprocessed_image_cache(
    cache_path: Path,
    prompt_hashes: list[str],
    images: torch.Tensor,
    metadata: dict[str, Any],
) -> None:
    """Save prompt-hash indexed preprocessed reference images."""
    if images.dtype != torch.uint8:
        raise ValueError(f"images must be uint8, got {images.dtype}.")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must have shape (N, 3, H, W), got {tuple(images.shape)}.")
    if len(prompt_hashes) != images.shape[0]:
        raise ValueError(
            f"prompt_hashes length ({len(prompt_hashes)}) does not match image rows "
            f"({images.shape[0]})."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "prompt_hashes": prompt_hashes,
        "images": images.detach().cpu().contiguous(),
        **metadata,
    }
    torch.save(payload, cache_path)


def load_preprocessed_image_cache(cache_path: Path) -> ConvNextPreprocessedImageCache:
    """Load a prompt-hash indexed preprocessed image cache into CPU memory."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Reference image cache does not exist: {cache_path}")

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported cache payload type: {type(payload).__name__}.")
    if "prompt_hashes" not in payload or "images" not in payload:
        raise ValueError("Cache payload must contain 'prompt_hashes' and 'images'.")

    prompt_hashes = [str(prompt_hash) for prompt_hash in payload["prompt_hashes"]]
    raw_images = payload["images"]
    if not isinstance(raw_images, torch.Tensor):
        raw_images = torch.as_tensor(raw_images)
    if raw_images.dtype != torch.uint8:
        raise ValueError(f"Cached images must use dtype torch.uint8, got {raw_images.dtype}.")
    images = raw_images.detach().cpu().contiguous()
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"Cached images must have shape (N, 3, H, W), got {tuple(images.shape)}.")
    if len(prompt_hashes) != images.shape[0]:
        raise ValueError(
            f"Cache prompt_hashes length ({len(prompt_hashes)}) does not match "
            f"image rows ({images.shape[0]})."
        )
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise ValueError("Cache contains duplicate prompt hashes.")

    metadata = {
        key: value for key, value in payload.items() if key not in {"prompt_hashes", "images"}
    }
    image_size = int(metadata.get("image_size", images.shape[-1]))
    if images.shape[-1] != image_size or images.shape[-2] != image_size:
        raise ValueError(
            f"Cached image shape {tuple(images.shape[-2:])} does not match "
            f"image_size metadata ({image_size})."
        )

    return ConvNextPreprocessedImageCache(
        prompt_hashes=prompt_hashes,
        images=images,
        image_size=image_size,
        metadata=metadata,
        index_by_hash={prompt_hash: index for index, prompt_hash in enumerate(prompt_hashes)},
    )


def convnext_perceptual_reward(
    real_stages: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    fake_stages: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    stage_weights: tuple[float, float, float, float, float],
    reward_temperature: float,
    early_stage_downsample: int = 1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute an exponential perceptual reward from ConvNeXt stage features."""
    if reward_temperature <= 0.0:
        raise ValueError(f"reward_temperature must be positive, got {reward_temperature}.")
    if early_stage_downsample <= 0:
        raise ValueError(f"early_stage_downsample must be positive, got {early_stage_downsample}.")
    if len(real_stages) != 5 or len(fake_stages) != 5:
        raise ValueError(
            f"Expected 5 real/fake feature stages, got {len(real_stages)} and {len(fake_stages)}."
        )

    batch_size = real_stages[0].shape[0]
    total_loss = torch.zeros(batch_size, dtype=torch.float32, device=real_stages[0].device)
    for stage_index, (real, fake, weight) in enumerate(
        zip(real_stages, fake_stages, stage_weights)
    ):
        if real.shape != fake.shape:
            raise ValueError(
                f"Feature shape mismatch at stage {stage_index}: "
                f"real={tuple(real.shape)} fake={tuple(fake.shape)}"
            )
        if real.ndim != 4:
            raise ValueError(
                f"Expected NCHW feature map at stage {stage_index}, got {tuple(real.shape)}."
            )

        real_float = real.float()
        fake_float = fake.float()
        if stage_index < 2 and early_stage_downsample > 1:
            real_float = F.avg_pool2d(
                real_float,
                kernel_size=early_stage_downsample,
                stride=early_stage_downsample,
            )
            fake_float = F.avg_pool2d(
                fake_float,
                kernel_size=early_stage_downsample,
                stride=early_stage_downsample,
            )

        real_normalized = F.normalize(real_float, dim=1, eps=eps)
        fake_normalized = F.normalize(fake_float, dim=1, eps=eps)
        similarity = (real_normalized * fake_normalized).sum(dim=1).mean(dim=(1, 2))
        stage_loss = (1.0 - similarity).clamp_min(0.0)
        total_loss = total_loss + float(weight) * stage_loss

    return torch.exp(-total_loss / reward_temperature)


class WDConvNextPerceptualModel:
    """Extract WD ConvNeXt stage features and compute perceptual rewards."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str,
        stage_weights: tuple[float, float, float, float, float] = DEFAULT_STAGE_WEIGHTS,
        reward_temperature: float = 0.35,
        early_stage_downsample: int = 1,
    ) -> None:
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.dtype_name = dtype if self.device.type == "cuda" else "float32"
        self.stage_weights = stage_weights
        self.reward_temperature = reward_temperature
        self.early_stage_downsample = early_stage_downsample

        self.preprocess_config = load_convnext_preprocess_config(model_path)
        self.image_size = self.preprocess_config.image_size
        self.image_mean = self.preprocess_config.image_mean
        self.image_std = self.preprocess_config.image_std
        self.model = self._load_model()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve the primary inference device."""
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return resolved
        if not torch.cuda.is_available():
            raise ValueError("CUDA device was requested, but CUDA is not available.")
        return torch.device("cuda:0" if resolved.index is None else resolved)

    def _load_model(self) -> torch.nn.Module:
        """Load the WD ConvNeXt timm model from a local safetensors checkpoint."""
        try:
            import timm
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "WD ConvNeXt perceptual reward requires `timm` and `safetensors`. "
                "Install them with `pip install timm safetensors`."
            ) from exc

        config_path = self.model_path / "config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)

        model = timm.create_model(
            model_config["architecture"],
            pretrained=False,
            num_classes=int(model_config["num_classes"]),
            **model_config.get("model_args", {}),
        )
        state_dict = load_file(str(self.model_path / "model.safetensors"), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def ensure_on_device(self) -> None:
        """Move the model to its inference device."""
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def offload_to_cpu(self) -> None:
        """Move the model to CPU and release CUDA cache when applicable."""
        self.model.to("cpu")
        self.model.eval()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device.index or 0):
                torch.cuda.empty_cache()

    def _get_autocast_context(self):
        """Create a fresh autocast context for one forward pass."""
        if self.device.type == "cuda" and self.dtype != torch.float32:
            return torch.autocast(device_type=self.device.type, dtype=self.dtype)
        return nullcontext()

    def _normalize_uint8_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Move and normalize a uint8 image batch for WD ConvNeXt inference."""
        if images.dtype != torch.uint8:
            raise ValueError(f"images must be uint8, got {images.dtype}.")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must have shape (N, 3, H, W), got {tuple(images.shape)}.")
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            raise ValueError(
                f"images must have spatial size {self.image_size}, got {tuple(images.shape[-2:])}."
            )

        batch = images.to(device=self.device, dtype=self.dtype) / 255.0
        mean = torch.tensor(self.image_mean, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        return (batch - mean) / std

    def _extract_stage_features(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract stem and four ConvNeXt stage feature maps."""
        if not hasattr(self.model, "stem") or not hasattr(self.model, "stages"):
            raise AttributeError("Expected the timm ConvNeXt model to expose stem and stages.")
        stages = list(self.model.stages)
        if len(stages) != 4:
            raise ValueError(f"Expected 4 ConvNeXt stages, got {len(stages)}.")

        features = []
        x = self.model.stem(batch)
        features.append(x)
        for stage in stages:
            x = stage(x)
            features.append(x)
        return tuple(features)  # type: ignore[return-value]

    @torch.inference_mode()
    def compute_rewards(
        self,
        reference_images: torch.Tensor,
        generated_images: list[Image.Image],
    ) -> torch.Tensor:
        """Compute perceptual rewards against preprocessed reference images."""
        generated_uint8 = preprocess_images_to_uint8(generated_images, self.image_size)
        if reference_images.shape[0] != generated_uint8.shape[0]:
            raise ValueError(
                f"Reference/generated batch mismatch: "
                f"reference={reference_images.shape[0]} generated={generated_uint8.shape[0]}."
            )

        combined_uint8 = torch.cat([reference_images, generated_uint8], dim=0)
        batch_size = reference_images.shape[0]
        batch = self._normalize_uint8_batch(combined_uint8)

        with self._get_autocast_context():
            combined_stages = self._extract_stage_features(batch)

        real_stages = tuple(stage[:batch_size] for stage in combined_stages)
        fake_stages = tuple(stage[batch_size:] for stage in combined_stages)
        rewards = convnext_perceptual_reward(
            real_stages=real_stages,  # type: ignore[arg-type]
            fake_stages=fake_stages,  # type: ignore[arg-type]
            stage_weights=self.stage_weights,
            reward_temperature=self.reward_temperature,
            early_stage_downsample=self.early_stage_downsample,
        )
        return rewards.detach().cpu()
