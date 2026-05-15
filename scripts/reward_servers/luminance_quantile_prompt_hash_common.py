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

"""Shared luminance quantile utilities for prompt-hash reward scripts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageOps

DEFAULT_QUANTILE_COUNT = 65
DEFAULT_QUANTILE_LOWER = 0.001
DEFAULT_QUANTILE_UPPER = 0.999


@dataclass(frozen=True)
class LuminanceQuantileConfig:
    """Configuration used to extract one luminance quantile descriptor."""

    image_size: int
    luminance_space: str
    quantile_levels: tuple[float, ...]
    black_threshold: float
    white_threshold: float

    def validate(self) -> None:
        """Validate descriptor extraction settings."""
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}.")
        if self.luminance_space not in {"lstar", "linear_y", "srgb_y"}:
            raise ValueError(
                f"Unsupported luminance_space: {self.luminance_space}. "
                "Expected one of: lstar, linear_y, srgb_y."
            )
        if len(self.quantile_levels) < 2:
            raise ValueError(
                f"At least two quantile levels are required, got {len(self.quantile_levels)}."
            )
        previous = -1.0
        for level in self.quantile_levels:
            if not 0.0 <= level <= 1.0:
                raise ValueError(f"Quantile levels must be in [0, 1], got {level}.")
            if level <= previous:
                raise ValueError("Quantile levels must be strictly increasing.")
            previous = level
        if not 0.0 <= self.black_threshold <= 1.0:
            raise ValueError(
                f"black_threshold must be in [0, 1], got {self.black_threshold}."
            )
        if not 0.0 <= self.white_threshold <= 1.0:
            raise ValueError(
                f"white_threshold must be in [0, 1], got {self.white_threshold}."
            )
        if self.black_threshold >= self.white_threshold:
            raise ValueError(
                "black_threshold must be smaller than white_threshold, got "
                f"{self.black_threshold} >= {self.white_threshold}."
            )

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, Any],
        *,
        image_size: Optional[int] = None,
        luminance_space: Optional[str] = None,
        quantile_levels: Optional[tuple[float, ...]] = None,
        black_threshold: Optional[float] = None,
        white_threshold: Optional[float] = None,
    ) -> "LuminanceQuantileConfig":
        """Build a descriptor config from cache metadata and optional overrides."""
        config = cls(
            image_size=int(image_size if image_size is not None else metadata["image_size"]),
            luminance_space=str(
                luminance_space
                if luminance_space is not None
                else metadata["luminance_space"]
            ),
            quantile_levels=tuple(
                float(value)
                for value in (
                    quantile_levels
                    if quantile_levels is not None
                    else metadata["quantile_levels"]
                )
            ),
            black_threshold=float(
                black_threshold
                if black_threshold is not None
                else metadata.get("black_threshold", 0.01)
            ),
            white_threshold=float(
                white_threshold
                if white_threshold is not None
                else metadata.get("white_threshold", 0.99)
            ),
        )
        config.validate()
        return config

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the descriptor config into cache metadata."""
        return {
            "descriptor": "luminance_quantile_curve",
            "image_size": self.image_size,
            "luminance_space": self.luminance_space,
            "quantile_levels": list(self.quantile_levels),
            "black_threshold": self.black_threshold,
            "white_threshold": self.white_threshold,
            "descriptor_transform": "quantile_curve_with_contrast_and_clip_stats",
        }


@dataclass(frozen=True)
class LuminanceQuantileOutputs:
    """Luminance quantile descriptors extracted from a batch of images."""

    quantiles: torch.Tensor
    contrasts: torch.Tensor
    black_fractions: torch.Tensor
    white_fractions: torch.Tensor


@dataclass(frozen=True)
class LuminanceQuantileReferenceCache:
    """Prompt-hash indexed luminance descriptors loaded from a torch cache."""

    prompt_hashes: list[str]
    quantiles: dict[str, torch.Tensor]
    contrasts: dict[str, torch.Tensor]
    black_fractions: dict[str, torch.Tensor]
    white_fractions: dict[str, torch.Tensor]
    metadata: dict[str, Any]


def prompt_sha256(prompt: str) -> str:
    """Return the exact UTF-8 SHA256 hash used as the reference lookup key."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Read a JSONL file and keep source line numbers for diagnostics."""
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            records.append((line_number, json.loads(stripped)))
    return records


def resolve_reference_image_path(
    jsonl_path: Path,
    reference_image_dir: Optional[Path],
    raw_path: str,
) -> Path:
    """Resolve a reference image path from a JSONL record."""
    image_path = Path(raw_path)
    if image_path.is_absolute():
        return image_path

    candidates = []
    if reference_image_dir is not None:
        candidates.append(reference_image_dir / image_path)
    candidates.append(jsonl_path.parent / image_path)
    candidates.append(jsonl_path.parent / "images" / image_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    attempted = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Reference image not found for {raw_path!r}. Tried: {attempted}")


def build_default_quantile_levels(
    num_quantiles: int = DEFAULT_QUANTILE_COUNT,
    lower: float = DEFAULT_QUANTILE_LOWER,
    upper: float = DEFAULT_QUANTILE_UPPER,
) -> tuple[float, ...]:
    """Build a tail-dense quantile grid that avoids exact min/max outliers."""
    if num_quantiles < 2:
        raise ValueError(f"num_quantiles must be at least 2, got {num_quantiles}.")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError(f"Expected 0 <= lower < upper <= 1, got {lower}, {upper}.")

    positions = np.linspace(0.0, 1.0, num_quantiles, dtype=np.float64)
    eased = 0.5 * (1.0 - np.cos(np.pi * positions))
    levels = lower + (upper - lower) * eased
    return tuple(float(level) for level in levels)


def parse_quantile_levels(
    raw_quantiles: Optional[str],
    num_quantiles: int,
) -> tuple[float, ...]:
    """Parse an explicit quantile list or build the default tail-dense grid."""
    if raw_quantiles:
        levels = tuple(
            float(value.strip()) for value in raw_quantiles.split(",") if value.strip()
        )
    else:
        levels = build_default_quantile_levels(num_quantiles=num_quantiles)

    config = LuminanceQuantileConfig(
        image_size=1,
        luminance_space="lstar",
        quantile_levels=levels,
        black_threshold=0.01,
        white_threshold=0.99,
    )
    config.validate()
    return levels


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Pad an RGB image to square using a white background."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 1] to linear RGB."""
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_y_to_lstar(linear_y: np.ndarray) -> np.ndarray:
    """Convert relative luminance Y to normalized CIE L* in [0, 1]."""
    delta = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    lstar = np.where(linear_y <= delta, kappa * linear_y, 116.0 * np.cbrt(linear_y) - 16.0)
    return np.clip(lstar / 100.0, 0.0, 1.0)


def _image_to_luminance_array(
    image: Image.Image,
    config: LuminanceQuantileConfig,
) -> np.ndarray:
    """Convert a PIL image into a flat luminance array in [0, 1]."""
    image = _pad_to_square(image)
    image = image.resize((config.image_size, config.image_size), Image.Resampling.BICUBIC)
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    if config.luminance_space == "srgb_y":
        luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        linear_rgb = _srgb_to_linear(rgb)
        linear_y = (
            0.2126 * linear_rgb[..., 0]
            + 0.7152 * linear_rgb[..., 1]
            + 0.0722 * linear_rgb[..., 2]
        )
        if config.luminance_space == "linear_y":
            luminance = linear_y
        elif config.luminance_space == "lstar":
            luminance = _linear_y_to_lstar(linear_y)
        else:
            raise ValueError(f"Unsupported luminance_space: {config.luminance_space}")

    return np.clip(luminance.astype(np.float32, copy=False), 0.0, 1.0).reshape(-1)


def extract_luminance_quantile_outputs(
    images: list[Image.Image],
    config: LuminanceQuantileConfig,
) -> LuminanceQuantileOutputs:
    """Extract luminance quantile descriptors for a batch of images."""
    if not images:
        raise ValueError("At least one image is required for luminance quantile extraction.")
    config.validate()

    levels = np.asarray(config.quantile_levels, dtype=np.float64)
    quantiles = []
    contrasts = []
    black_fractions = []
    white_fractions = []
    for image in images:
        luminance = _image_to_luminance_array(image, config)
        quantile_values = np.quantile(luminance, levels, method="linear").astype(np.float32)
        contrast_bounds = np.quantile(luminance, [0.05, 0.95], method="linear")
        contrast = np.float32(contrast_bounds[1] - contrast_bounds[0])
        black_fraction = np.float32(np.mean(luminance <= config.black_threshold))
        white_fraction = np.float32(np.mean(luminance >= config.white_threshold))

        if not np.all(np.isfinite(quantile_values)):
            raise ValueError("Luminance quantile descriptor contains non-finite values.")
        if not np.isfinite(contrast):
            raise ValueError("Luminance contrast descriptor contains a non-finite value.")

        quantiles.append(torch.from_numpy(quantile_values))
        contrasts.append(torch.tensor(contrast, dtype=torch.float32))
        black_fractions.append(torch.tensor(black_fraction, dtype=torch.float32))
        white_fractions.append(torch.tensor(white_fraction, dtype=torch.float32))

    return LuminanceQuantileOutputs(
        quantiles=torch.stack(quantiles, dim=0),
        contrasts=torch.stack(contrasts, dim=0),
        black_fractions=torch.stack(black_fractions, dim=0),
        white_fractions=torch.stack(white_fractions, dim=0),
    )


def _build_quantile_weights(
    quantile_levels: tuple[float, ...],
    tail_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """Build optional tail-emphasis weights with mean weight normalized to 1."""
    if tail_weight < 0.0:
        raise ValueError(f"tail_weight must be non-negative, got {tail_weight}.")
    levels = torch.tensor(quantile_levels, dtype=torch.float32, device=device)
    weights = 1.0 + tail_weight * torch.abs(levels - 0.5) * 2.0
    return weights / weights.mean().clamp_min(1e-8)


def quantile_curve_similarity(
    generated_quantiles: torch.Tensor,
    reference_quantiles: torch.Tensor,
    *,
    quantile_levels: tuple[float, ...],
    reward_temperature: float,
    tail_weight: float,
) -> torch.Tensor:
    """Compare luminance quantile curves with an exponential L1 kernel."""
    if reward_temperature <= 0.0:
        raise ValueError(
            f"reward_temperature must be positive, got {reward_temperature}."
        )
    if generated_quantiles.shape != reference_quantiles.shape:
        raise ValueError(
            "Quantile shape mismatch: "
            f"generated={tuple(generated_quantiles.shape)} "
            f"reference={tuple(reference_quantiles.shape)}"
        )
    if generated_quantiles.shape[-1] != len(quantile_levels):
        raise ValueError(
            f"Quantile tensor width ({generated_quantiles.shape[-1]}) does not match "
            f"quantile_levels length ({len(quantile_levels)})."
        )

    weights = _build_quantile_weights(
        quantile_levels=quantile_levels,
        tail_weight=tail_weight,
        device=generated_quantiles.device,
    )
    distance = torch.mean(torch.abs(generated_quantiles - reference_quantiles) * weights, dim=-1)
    return torch.exp(-distance / reward_temperature)


def contrast_similarity(
    generated_contrasts: torch.Tensor,
    reference_contrasts: torch.Tensor,
    *,
    reward_temperature: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compare P95-P05 luminance contrast with a log-ratio kernel."""
    if reward_temperature <= 0.0:
        raise ValueError(
            f"contrast_temperature must be positive, got {reward_temperature}."
        )
    if generated_contrasts.shape != reference_contrasts.shape:
        raise ValueError(
            "Contrast shape mismatch: "
            f"generated={tuple(generated_contrasts.shape)} "
            f"reference={tuple(reference_contrasts.shape)}"
        )
    distance = torch.abs(
        torch.log(generated_contrasts.clamp_min(eps))
        - torch.log(reference_contrasts.clamp_min(eps))
    )
    return torch.exp(-distance / reward_temperature)


def clip_similarity(
    generated_black_fractions: torch.Tensor,
    reference_black_fractions: torch.Tensor,
    generated_white_fractions: torch.Tensor,
    reference_white_fractions: torch.Tensor,
    *,
    reward_temperature: float,
) -> torch.Tensor:
    """Compare black and white clipping fractions."""
    if reward_temperature <= 0.0:
        raise ValueError(f"clip_temperature must be positive, got {reward_temperature}.")
    if generated_black_fractions.shape != reference_black_fractions.shape:
        raise ValueError(
            "Black fraction shape mismatch: "
            f"generated={tuple(generated_black_fractions.shape)} "
            f"reference={tuple(reference_black_fractions.shape)}"
        )
    if generated_white_fractions.shape != reference_white_fractions.shape:
        raise ValueError(
            "White fraction shape mismatch: "
            f"generated={tuple(generated_white_fractions.shape)} "
            f"reference={tuple(reference_white_fractions.shape)}"
        )
    distance = torch.abs(generated_black_fractions - reference_black_fractions)
    distance = distance + torch.abs(generated_white_fractions - reference_white_fractions)
    return torch.exp(-distance / reward_temperature)


def luminance_quantile_reward(
    generated_outputs: LuminanceQuantileOutputs,
    reference_outputs: LuminanceQuantileOutputs,
    *,
    quantile_levels: tuple[float, ...],
    curve_weight: float,
    contrast_weight: float,
    clip_weight: float,
    reward_temperature: float,
    contrast_temperature: float,
    clip_temperature: float,
    tail_weight: float,
) -> torch.Tensor:
    """Compute the final luminance quantile similarity reward."""
    weights = (curve_weight, contrast_weight, clip_weight)
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"Reward weights must be non-negative, got {weights}.")
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError(f"At least one reward weight must be positive, got {weights}.")

    curve_score = quantile_curve_similarity(
        generated_outputs.quantiles,
        reference_outputs.quantiles,
        quantile_levels=quantile_levels,
        reward_temperature=reward_temperature,
        tail_weight=tail_weight,
    )
    contrast_score = contrast_similarity(
        generated_outputs.contrasts,
        reference_outputs.contrasts,
        reward_temperature=contrast_temperature,
    )
    clipping_score = clip_similarity(
        generated_outputs.black_fractions,
        reference_outputs.black_fractions,
        generated_outputs.white_fractions,
        reference_outputs.white_fractions,
        reward_temperature=clip_temperature,
    )
    return (
        curve_weight * curve_score
        + contrast_weight * contrast_score
        + clip_weight * clipping_score
    ) / weight_sum


def save_reference_cache(
    cache_path: Path,
    prompt_hashes: list[str],
    outputs_by_hash: dict[str, LuminanceQuantileOutputs],
    metadata: dict[str, Any],
) -> None:
    """Save prompt-hash reference luminance descriptors to a torch cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    quantiles = torch.stack(
        [outputs_by_hash[prompt_hash].quantiles.squeeze(0) for prompt_hash in prompt_hashes]
    )
    contrasts = torch.stack(
        [outputs_by_hash[prompt_hash].contrasts.squeeze(0) for prompt_hash in prompt_hashes]
    )
    black_fractions = torch.stack(
        [
            outputs_by_hash[prompt_hash].black_fractions.squeeze(0)
            for prompt_hash in prompt_hashes
        ]
    )
    white_fractions = torch.stack(
        [
            outputs_by_hash[prompt_hash].white_fractions.squeeze(0)
            for prompt_hash in prompt_hashes
        ]
    )
    payload: dict[str, Any] = {
        "prompt_hashes": prompt_hashes,
        "quantiles": quantiles,
        "contrasts": contrasts,
        "black_fractions": black_fractions,
        "white_fractions": white_fractions,
        **metadata,
    }
    torch.save(payload, cache_path)


def load_reference_cache_payload(cache_path: Path) -> LuminanceQuantileReferenceCache:
    """Load a prompt-hash reference luminance quantile cache payload."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Reference cache does not exist: {cache_path}")

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")

    required_keys = {
        "prompt_hashes",
        "quantiles",
        "contrasts",
        "black_fractions",
        "white_fractions",
    }
    if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
        raise ValueError(
            "Unsupported luminance quantile reference cache format. Expected keys: "
            "prompt_hashes, quantiles, contrasts, black_fractions, white_fractions."
        )

    prompt_hashes = [str(prompt_hash) for prompt_hash in payload["prompt_hashes"]]
    quantiles = torch.as_tensor(payload["quantiles"], dtype=torch.float32)
    contrasts = torch.as_tensor(payload["contrasts"], dtype=torch.float32)
    black_fractions = torch.as_tensor(payload["black_fractions"], dtype=torch.float32)
    white_fractions = torch.as_tensor(payload["white_fractions"], dtype=torch.float32)

    if not prompt_hashes:
        raise ValueError("Cache must contain at least one prompt hash.")
    if quantiles.ndim != 2:
        raise ValueError(f"Cached quantiles must have shape (N, Q), got {tuple(quantiles.shape)}.")
    for name, tensor in (
        ("contrasts", contrasts),
        ("black_fractions", black_fractions),
        ("white_fractions", white_fractions),
    ):
        if tensor.ndim != 1:
            raise ValueError(f"Cached {name} must have shape (N,), got {tuple(tensor.shape)}.")

    if len(prompt_hashes) != quantiles.shape[0]:
        raise ValueError(
            f"Cache prompt_hashes length ({len(prompt_hashes)}) does not match "
            f"quantiles rows ({quantiles.shape[0]})."
        )
    for name, tensor in (
        ("contrasts", contrasts),
        ("black_fractions", black_fractions),
        ("white_fractions", white_fractions),
    ):
        if len(prompt_hashes) != tensor.shape[0]:
            raise ValueError(
                f"Cache prompt_hashes length ({len(prompt_hashes)}) does not match "
                f"{name} rows ({tensor.shape[0]})."
            )
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise ValueError("Cache contains duplicate prompt hashes.")

    metadata = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "prompt_hashes",
            "quantiles",
            "contrasts",
            "black_fractions",
            "white_fractions",
        }
    }
    descriptor_config = LuminanceQuantileConfig.from_metadata(metadata)
    if quantiles.shape[1] != len(descriptor_config.quantile_levels):
        raise ValueError(
            f"Cached quantile width ({quantiles.shape[1]}) does not match "
            f"quantile_levels metadata ({len(descriptor_config.quantile_levels)})."
        )
    return LuminanceQuantileReferenceCache(
        prompt_hashes=prompt_hashes,
        quantiles={
            prompt_hash: quantiles[index].detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        contrasts={
            prompt_hash: contrasts[index].reshape(()).detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        black_fractions={
            prompt_hash: black_fractions[index].reshape(()).detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        white_fractions={
            prompt_hash: white_fractions[index].reshape(()).detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        metadata=metadata,
    )
