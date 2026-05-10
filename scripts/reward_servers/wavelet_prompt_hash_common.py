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

"""Shared Haar wavelet energy utilities for prompt-hash reward scripts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


@dataclass(frozen=True)
class WaveletDescriptorConfig:
    """Configuration used to extract one wavelet energy descriptor."""

    image_size: int
    levels: int
    color_space: str
    include_approximation: bool
    level_weight_decay: float

    def validate(self) -> None:
        """Validate descriptor extraction settings."""
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}.")
        if self.levels <= 0:
            raise ValueError(f"levels must be positive, got {self.levels}.")
        divisor = 2**self.levels
        if self.image_size % divisor != 0:
            raise ValueError(
                f"image_size({self.image_size}) must be divisible by 2 ** levels({divisor})."
            )
        if self.color_space not in {"y", "rgb", "ycbcr"}:
            raise ValueError(
                f"Unsupported color_space: {self.color_space}. " "Expected one of: y, rgb, ycbcr."
            )
        if self.level_weight_decay <= 0:
            raise ValueError(f"level_weight_decay must be positive, got {self.level_weight_decay}.")

    @classmethod
    def from_metadata(
        cls,
        metadata: dict[str, Any],
        *,
        image_size: Optional[int] = None,
        levels: Optional[int] = None,
        color_space: Optional[str] = None,
        include_approximation: Optional[bool] = None,
        level_weight_decay: Optional[float] = None,
    ) -> "WaveletDescriptorConfig":
        """Build a descriptor config from cache metadata and optional overrides."""
        config = cls(
            image_size=int(image_size if image_size is not None else metadata["image_size"]),
            levels=int(levels if levels is not None else metadata["levels"]),
            color_space=str(color_space if color_space is not None else metadata["color_space"]),
            include_approximation=bool(
                include_approximation
                if include_approximation is not None
                else metadata.get("include_approximation", False)
            ),
            level_weight_decay=float(
                level_weight_decay
                if level_weight_decay is not None
                else metadata.get("level_weight_decay", 1.0)
            ),
        )
        config.validate()
        return config

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the descriptor config into cache metadata."""
        return {
            "wavelet": "haar",
            "image_size": self.image_size,
            "levels": self.levels,
            "color_space": self.color_space,
            "include_approximation": self.include_approximation,
            "level_weight_decay": self.level_weight_decay,
            "descriptor_transform": "multicomponent_log1p_energy_and_lowfreq",
        }


@dataclass(frozen=True)
class WaveletImageOutputs:
    """Wavelet descriptors extracted from a batch of images."""

    distributions: torch.Tensor
    log_energies: torch.Tensor
    total_energies: torch.Tensor
    luminance_structure_distributions: torch.Tensor
    luminance_structure_log_energies: torch.Tensor
    luminance_lowfreq_vectors: torch.Tensor
    chroma_lowfreq_vectors: torch.Tensor
    chroma_midfreq_distributions: torch.Tensor
    chroma_midfreq_log_energies: torch.Tensor


@dataclass(frozen=True)
class WaveletReferenceCache:
    """Prompt-hash indexed wavelet reference descriptors loaded from a torch cache."""

    prompt_hashes: list[str]
    distributions: dict[str, torch.Tensor]
    log_energies: dict[str, torch.Tensor]
    total_energies: dict[str, torch.Tensor]
    luminance_structure_distributions: dict[str, torch.Tensor]
    luminance_structure_log_energies: dict[str, torch.Tensor]
    luminance_lowfreq_vectors: dict[str, torch.Tensor]
    chroma_lowfreq_vectors: dict[str, torch.Tensor]
    chroma_midfreq_distributions: dict[str, torch.Tensor]
    chroma_midfreq_log_energies: dict[str, torch.Tensor]
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


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Pad an RGB image to square using a white background."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def _image_to_channel_array(image: Image.Image, config: WaveletDescriptorConfig) -> np.ndarray:
    """Convert a PIL image into a channel-first float array in [0, 1]."""
    image = _pad_to_square(image)
    image = image.resize((config.image_size, config.image_size), Image.Resampling.BICUBIC)

    if config.color_space == "y":
        array = np.asarray(image.convert("YCbCr"), dtype=np.float32)[..., :1]
    elif config.color_space == "rgb":
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    elif config.color_space == "ycbcr":
        array = np.asarray(image.convert("YCbCr"), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported color_space: {config.color_space}")

    return np.transpose(array / 255.0, (2, 0, 1))


def _haar_step(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run one orthonormal 2D Haar decomposition step."""
    inv_sqrt2 = np.float32(1.0 / np.sqrt(2.0))
    low_columns = (channel[:, 0::2] + channel[:, 1::2]) * inv_sqrt2
    high_columns = (channel[:, 0::2] - channel[:, 1::2]) * inv_sqrt2

    ll = (low_columns[0::2, :] + low_columns[1::2, :]) * inv_sqrt2
    lh = (low_columns[0::2, :] - low_columns[1::2, :]) * inv_sqrt2
    hl = (high_columns[0::2, :] + high_columns[1::2, :]) * inv_sqrt2
    hh = (high_columns[0::2, :] - high_columns[1::2, :]) * inv_sqrt2
    return ll, lh, hl, hh


def _weighted_band_energy(band: np.ndarray, weight: float) -> float:
    """Compute weighted squared energy for one wavelet band."""
    band64 = band.astype(np.float64, copy=False)
    return float(np.sum(np.square(band64)) * weight)


def _extract_channel_energies(
    channel: np.ndarray,
    config: WaveletDescriptorConfig,
) -> list[float]:
    """Extract hierarchical Haar detail energies from one image channel."""
    current = channel
    energies: list[float] = []
    for level_index in range(config.levels):
        ll, lh, hl, hh = _haar_step(current)
        weight = config.level_weight_decay**level_index
        energies.extend(
            [
                _weighted_band_energy(lh, weight),
                _weighted_band_energy(hl, weight),
                _weighted_band_energy(hh, weight),
            ]
        )
        current = ll

    if config.include_approximation:
        weight = config.level_weight_decay**config.levels
        energies.append(_weighted_band_energy(current, weight))

    return energies


def _extract_channel_detail_energies_and_lowfreq(
    channel: np.ndarray,
    config: WaveletDescriptorConfig,
) -> tuple[list[list[float]], torch.Tensor]:
    """Extract per-level Haar detail energies and the final low-frequency map."""
    current = channel
    detail_energies_by_level: list[list[float]] = []
    for level_index in range(config.levels):
        ll, lh, hl, hh = _haar_step(current)
        weight = config.level_weight_decay**level_index
        detail_energies_by_level.append(
            [
                _weighted_band_energy(lh, weight),
                _weighted_band_energy(hl, weight),
                _weighted_band_energy(hh, weight),
            ]
        )
        current = ll

    lowfreq_map = current / float(2**config.levels)
    lowfreq_vector = torch.from_numpy(lowfreq_map.astype(np.float32, copy=False).reshape(-1))
    return detail_energies_by_level, lowfreq_vector


def _mid_frequency_level_indices(levels: int) -> list[int]:
    """Return detail level indices used for color mid-frequency comparison."""
    if levels <= 0:
        raise ValueError(f"levels must be positive, got {levels}.")
    if levels <= 2:
        return list(range(levels))
    return list(range(1, levels - 1))


def _normalize_log_energy_distribution(
    raw_energies: list[float],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert raw non-negative energies into log energies and a distribution."""
    raw_energy_tensor = torch.tensor(raw_energies, dtype=torch.float32)
    if torch.any(~torch.isfinite(raw_energy_tensor)):
        raise ValueError("Wavelet descriptor contains non-finite raw energies.")

    log_energy_tensor = torch.log1p(raw_energy_tensor)
    distribution = log_energy_tensor / log_energy_tensor.sum().clamp_min(eps)
    return log_energy_tensor, distribution


def _image_to_ycbcr_channel_array(
    image: Image.Image,
    config: WaveletDescriptorConfig,
) -> np.ndarray:
    """Convert a PIL image into YCbCr channel-first floats in [0, 1]."""
    image = _pad_to_square(image)
    image = image.resize((config.image_size, config.image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image.convert("YCbCr"), dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def extract_wavelet_outputs(
    images: list[Image.Image],
    config: WaveletDescriptorConfig,
    eps: float = 1e-8,
) -> WaveletImageOutputs:
    """Extract normalized log-energy wavelet descriptors for a batch of images."""
    if not images:
        raise ValueError("At least one image is required for wavelet descriptor extraction.")
    config.validate()

    distributions = []
    log_energies = []
    total_energies = []
    luminance_structure_distributions = []
    luminance_structure_log_energies = []
    luminance_lowfreq_vectors = []
    chroma_lowfreq_vectors = []
    chroma_midfreq_distributions = []
    chroma_midfreq_log_energies = []
    for image in images:
        channel_array = _image_to_channel_array(image, config)
        raw_energies: list[float] = []
        for channel in channel_array:
            raw_energies.extend(_extract_channel_energies(channel, config))

        log_energy_tensor, distribution = _normalize_log_energy_distribution(
            raw_energies,
            eps,
        )
        distributions.append(distribution)
        log_energies.append(log_energy_tensor)
        total_energies.append(torch.expm1(log_energy_tensor).sum())

        ycbcr_channels = _image_to_ycbcr_channel_array(image, config)
        y_detail_energies_by_level, y_lowfreq_vector = _extract_channel_detail_energies_and_lowfreq(
            ycbcr_channels[0], config
        )
        y_structure_energies = [
            energy for level_energies in y_detail_energies_by_level for energy in level_energies
        ]
        y_log_energies, y_distribution = _normalize_log_energy_distribution(
            y_structure_energies,
            eps,
        )
        luminance_structure_log_energies.append(y_log_energies)
        luminance_structure_distributions.append(y_distribution)
        luminance_lowfreq_vectors.append(y_lowfreq_vector)

        mid_indices = _mid_frequency_level_indices(config.levels)
        chroma_lowfreq_parts = []
        chroma_midfreq_energies = []
        for chroma_channel in ycbcr_channels[1:]:
            chroma_detail_energies_by_level, chroma_lowfreq_vector = (
                _extract_channel_detail_energies_and_lowfreq(chroma_channel, config)
            )
            chroma_lowfreq_parts.append(chroma_lowfreq_vector)
            for level_index in mid_indices:
                chroma_midfreq_energies.extend(chroma_detail_energies_by_level[level_index])

        chroma_lowfreq_vectors.append(torch.cat(chroma_lowfreq_parts, dim=0))
        chroma_log_energies, chroma_distribution = _normalize_log_energy_distribution(
            chroma_midfreq_energies,
            eps,
        )
        chroma_midfreq_log_energies.append(chroma_log_energies)
        chroma_midfreq_distributions.append(chroma_distribution)

    return WaveletImageOutputs(
        distributions=torch.stack(distributions, dim=0),
        log_energies=torch.stack(log_energies, dim=0),
        total_energies=torch.stack(total_energies, dim=0),
        luminance_structure_distributions=torch.stack(
            luminance_structure_distributions,
            dim=0,
        ),
        luminance_structure_log_energies=torch.stack(
            luminance_structure_log_energies,
            dim=0,
        ),
        luminance_lowfreq_vectors=torch.stack(luminance_lowfreq_vectors, dim=0),
        chroma_lowfreq_vectors=torch.stack(chroma_lowfreq_vectors, dim=0),
        chroma_midfreq_distributions=torch.stack(chroma_midfreq_distributions, dim=0),
        chroma_midfreq_log_energies=torch.stack(chroma_midfreq_log_energies, dim=0),
    )


def bhattacharyya_score(
    generated_distributions: torch.Tensor,
    reference_distributions: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute bounded distribution similarity for two wavelet energy distributions."""
    if generated_distributions.shape != reference_distributions.shape:
        raise ValueError(
            "Distribution shape mismatch: "
            f"generated={tuple(generated_distributions.shape)} "
            f"reference={tuple(reference_distributions.shape)}"
        )
    scores = torch.sqrt(
        generated_distributions.clamp_min(0) * reference_distributions.clamp_min(0)
    ).sum(dim=-1)
    generated_empty = generated_distributions.sum(dim=-1).abs() <= eps
    reference_empty = reference_distributions.sum(dim=-1).abs() <= eps
    return torch.where(generated_empty & reference_empty, torch.ones_like(scores), scores)


def soft_jaccard_score(
    generated_distributions: torch.Tensor,
    reference_distributions: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute soft Jaccard similarity between two wavelet energy distributions."""
    if generated_distributions.shape != reference_distributions.shape:
        raise ValueError(
            "Distribution shape mismatch: "
            f"generated={tuple(generated_distributions.shape)} "
            f"reference={tuple(reference_distributions.shape)}"
        )
    intersection = torch.minimum(generated_distributions, reference_distributions).sum(dim=-1)
    union = torch.maximum(generated_distributions, reference_distributions).sum(dim=-1)
    scores = intersection / union.clamp_min(eps)
    return torch.where(union <= eps, torch.ones_like(scores), scores)


def cosine_log_energy_score(
    generated_log_energies: torch.Tensor,
    reference_log_energies: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between log-energy wavelet descriptors."""
    if generated_log_energies.shape != reference_log_energies.shape:
        raise ValueError(
            "Log-energy shape mismatch: "
            f"generated={tuple(generated_log_energies.shape)} "
            f"reference={tuple(reference_log_energies.shape)}"
        )
    generated_norm = generated_log_energies.norm(dim=-1)
    reference_norm = reference_log_energies.norm(dim=-1)
    generated_normalized = F.normalize(generated_log_energies, dim=-1)
    reference_normalized = F.normalize(reference_log_energies, dim=-1)
    scores = (generated_normalized * reference_normalized).sum(dim=-1)
    return torch.where(
        (generated_norm <= 1e-8) & (reference_norm <= 1e-8),
        torch.ones_like(scores),
        scores,
    )


def descriptor_similarity_score(
    generated_distributions: torch.Tensor,
    reference_distributions: torch.Tensor,
    generated_log_energies: torch.Tensor,
    reference_log_energies: torch.Tensor,
    score_type: str,
) -> torch.Tensor:
    """Compute one descriptor similarity score by score type."""
    if score_type == "bhattacharyya":
        return bhattacharyya_score(generated_distributions, reference_distributions)
    if score_type == "soft_jaccard":
        return soft_jaccard_score(generated_distributions, reference_distributions)
    if score_type == "cosine_log_energy":
        return cosine_log_energy_score(generated_log_energies, reference_log_energies)
    raise ValueError(f"Unsupported score_type: {score_type}")


def lowfreq_l1_similarity(
    generated_vectors: torch.Tensor,
    reference_vectors: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Compare low-frequency maps with an exponential L1 kernel."""
    if tau <= 0:
        raise ValueError(f"lowfreq_tau must be positive, got {tau}.")
    if generated_vectors.shape != reference_vectors.shape:
        raise ValueError(
            "Low-frequency vector shape mismatch: "
            f"generated={tuple(generated_vectors.shape)} "
            f"reference={tuple(reference_vectors.shape)}"
        )
    distance = torch.mean(torch.abs(generated_vectors - reference_vectors), dim=-1)
    return torch.exp(-distance / tau)


def hh1_excess_similarity(
    generated_luminance_log_energies: torch.Tensor,
    reference_luminance_log_energies: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Penalize only excessive generated luminance HH1 detail energy."""
    if tau <= 0:
        raise ValueError(f"hh1_excess_tau must be positive, got {tau}.")
    if generated_luminance_log_energies.shape != reference_luminance_log_energies.shape:
        raise ValueError(
            "Luminance structure log-energy shape mismatch: "
            f"generated={tuple(generated_luminance_log_energies.shape)} "
            f"reference={tuple(reference_luminance_log_energies.shape)}"
        )
    if generated_luminance_log_energies.shape[-1] < 3:
        raise ValueError(
            "Luminance structure descriptor must contain at least LH1, HL1, and HH1 bands."
        )

    generated_hh1 = generated_luminance_log_energies[..., 2]
    reference_hh1 = reference_luminance_log_energies[..., 2]
    excess = torch.relu(generated_hh1 - reference_hh1)
    return torch.exp(-excess / tau)


def normalize_reward_weights(weights: dict[str, float]) -> dict[str, float]:
    """Normalize non-negative component weights."""
    for name, weight in weights.items():
        if weight < 0:
            raise ValueError(f"{name} must be non-negative, got {weight}.")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("At least one reward component weight must be positive.")
    return {name: weight / total for name, weight in weights.items()}


def total_energy_similarity(
    generated_total_energies: torch.Tensor,
    reference_total_energies: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare total wavelet energy magnitudes with a log-ratio kernel."""
    if tau <= 0:
        raise ValueError(f"total_energy_tau must be positive, got {tau}.")
    if generated_total_energies.shape != reference_total_energies.shape:
        raise ValueError(
            "Total energy shape mismatch: "
            f"generated={tuple(generated_total_energies.shape)} "
            f"reference={tuple(reference_total_energies.shape)}"
        )
    distance = torch.abs(
        torch.log(generated_total_energies.clamp_min(eps))
        - torch.log(reference_total_energies.clamp_min(eps))
    )
    return torch.exp(-distance / tau)


def wavelet_similarity_reward(
    generated_outputs: WaveletImageOutputs,
    reference_outputs: WaveletImageOutputs,
    *,
    score_type: str,
    distribution_weight: float,
    total_energy_tau: float,
    lowfreq_tau: float,
    luminance_structure_weight: float,
    luminance_lowfreq_weight: float,
    chroma_lowfreq_weight: float,
    chroma_midfreq_weight: float,
    hh1_excess_weight: float,
    hh1_excess_tau: float,
) -> torch.Tensor:
    """Compute the final wavelet similarity reward."""
    if not 0.0 <= distribution_weight <= 1.0:
        raise ValueError(f"distribution_weight must be in [0, 1], got {distribution_weight}.")
    component_weights = normalize_reward_weights(
        {
            "luminance_structure": luminance_structure_weight,
            "luminance_lowfreq": luminance_lowfreq_weight,
            "chroma_lowfreq": chroma_lowfreq_weight,
            "chroma_midfreq": chroma_midfreq_weight,
            "hh1_excess": hh1_excess_weight,
        }
    )

    luminance_structure_score = descriptor_similarity_score(
        generated_outputs.luminance_structure_distributions,
        reference_outputs.luminance_structure_distributions,
        generated_outputs.luminance_structure_log_energies,
        reference_outputs.luminance_structure_log_energies,
        score_type,
    )
    luminance_lowfreq_score = lowfreq_l1_similarity(
        generated_outputs.luminance_lowfreq_vectors,
        reference_outputs.luminance_lowfreq_vectors,
        tau=lowfreq_tau,
    )
    chroma_lowfreq_score = lowfreq_l1_similarity(
        generated_outputs.chroma_lowfreq_vectors,
        reference_outputs.chroma_lowfreq_vectors,
        tau=lowfreq_tau,
    )
    chroma_midfreq_score = descriptor_similarity_score(
        generated_outputs.chroma_midfreq_distributions,
        reference_outputs.chroma_midfreq_distributions,
        generated_outputs.chroma_midfreq_log_energies,
        reference_outputs.chroma_midfreq_log_energies,
        score_type,
    )
    hh1_excess_score = hh1_excess_similarity(
        generated_outputs.luminance_structure_log_energies,
        reference_outputs.luminance_structure_log_energies,
        tau=hh1_excess_tau,
    )

    component_score = (
        component_weights["luminance_structure"] * luminance_structure_score
        + component_weights["luminance_lowfreq"] * luminance_lowfreq_score
        + component_weights["chroma_lowfreq"] * chroma_lowfreq_score
        + component_weights["chroma_midfreq"] * chroma_midfreq_score
        + component_weights["hh1_excess"] * hh1_excess_score
    )

    energy_score = total_energy_similarity(
        generated_outputs.total_energies,
        reference_outputs.total_energies,
        tau=total_energy_tau,
    )
    return distribution_weight * component_score + (1.0 - distribution_weight) * energy_score


def save_reference_cache(
    cache_path: Path,
    prompt_hashes: list[str],
    outputs_by_hash: dict[str, WaveletImageOutputs],
    metadata: dict[str, Any],
) -> None:
    """Save prompt-hash reference wavelet descriptors to a torch cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    distributions = torch.stack(
        [outputs_by_hash[prompt_hash].distributions.squeeze(0) for prompt_hash in prompt_hashes]
    )
    log_energies = torch.stack(
        [outputs_by_hash[prompt_hash].log_energies.squeeze(0) for prompt_hash in prompt_hashes]
    )
    total_energies = torch.stack(
        [outputs_by_hash[prompt_hash].total_energies.squeeze(0) for prompt_hash in prompt_hashes]
    )
    luminance_structure_distributions = torch.stack(
        [
            outputs_by_hash[prompt_hash].luminance_structure_distributions.squeeze(0)
            for prompt_hash in prompt_hashes
        ]
    )
    luminance_structure_log_energies = torch.stack(
        [
            outputs_by_hash[prompt_hash].luminance_structure_log_energies.squeeze(0)
            for prompt_hash in prompt_hashes
        ]
    )
    luminance_lowfreq_vectors = torch.stack(
        [
            outputs_by_hash[prompt_hash].luminance_lowfreq_vectors.squeeze(0)
            for prompt_hash in prompt_hashes
        ]
    )
    chroma_lowfreq_vectors = torch.stack(
        [
            outputs_by_hash[prompt_hash].chroma_lowfreq_vectors.squeeze(0)
            for prompt_hash in prompt_hashes
        ]
    )
    chroma_midfreq_distributions = torch.stack(
        [
            outputs_by_hash[prompt_hash].chroma_midfreq_distributions.squeeze(0)
            for prompt_hash in prompt_hashes
        ]
    )
    chroma_midfreq_log_energies = torch.stack(
        [
            outputs_by_hash[prompt_hash].chroma_midfreq_log_energies.squeeze(0)
            for prompt_hash in prompt_hashes
        ]
    )
    payload: dict[str, Any] = {
        "prompt_hashes": prompt_hashes,
        "distributions": distributions,
        "log_energies": log_energies,
        "total_energies": total_energies,
        "luminance_structure_distributions": luminance_structure_distributions,
        "luminance_structure_log_energies": luminance_structure_log_energies,
        "luminance_lowfreq_vectors": luminance_lowfreq_vectors,
        "chroma_lowfreq_vectors": chroma_lowfreq_vectors,
        "chroma_midfreq_distributions": chroma_midfreq_distributions,
        "chroma_midfreq_log_energies": chroma_midfreq_log_energies,
        **metadata,
    }
    torch.save(payload, cache_path)


def load_reference_cache_payload(cache_path: Path) -> WaveletReferenceCache:
    """Load a prompt-hash reference wavelet cache payload."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Reference cache does not exist: {cache_path}")

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")

    required_keys = {
        "prompt_hashes",
        "distributions",
        "log_energies",
        "total_energies",
        "luminance_structure_distributions",
        "luminance_structure_log_energies",
        "luminance_lowfreq_vectors",
        "chroma_lowfreq_vectors",
        "chroma_midfreq_distributions",
        "chroma_midfreq_log_energies",
    }
    if not isinstance(payload, dict) or not required_keys.issubset(payload.keys()):
        missing_keys = (
            sorted(required_keys - set(payload.keys())) if isinstance(payload, dict) else []
        )
        raise ValueError(
            "Unsupported wavelet reference cache format. Expected keys: "
            f"{sorted(required_keys)}. Missing keys: {missing_keys}. "
            "Rebuild the cache with build_wavelet_prompt_hash_cache.py."
        )

    prompt_hashes = [str(prompt_hash) for prompt_hash in payload["prompt_hashes"]]
    distributions = torch.as_tensor(payload["distributions"], dtype=torch.float32)
    log_energies = torch.as_tensor(payload["log_energies"], dtype=torch.float32)
    total_energies = torch.as_tensor(payload["total_energies"], dtype=torch.float32)
    luminance_structure_distributions = torch.as_tensor(
        payload["luminance_structure_distributions"],
        dtype=torch.float32,
    )
    luminance_structure_log_energies = torch.as_tensor(
        payload["luminance_structure_log_energies"],
        dtype=torch.float32,
    )
    luminance_lowfreq_vectors = torch.as_tensor(
        payload["luminance_lowfreq_vectors"],
        dtype=torch.float32,
    )
    chroma_lowfreq_vectors = torch.as_tensor(
        payload["chroma_lowfreq_vectors"],
        dtype=torch.float32,
    )
    chroma_midfreq_distributions = torch.as_tensor(
        payload["chroma_midfreq_distributions"],
        dtype=torch.float32,
    )
    chroma_midfreq_log_energies = torch.as_tensor(
        payload["chroma_midfreq_log_energies"],
        dtype=torch.float32,
    )

    tensors_by_name = {
        "distributions": distributions,
        "log_energies": log_energies,
        "total_energies": total_energies,
        "luminance_structure_distributions": luminance_structure_distributions,
        "luminance_structure_log_energies": luminance_structure_log_energies,
        "luminance_lowfreq_vectors": luminance_lowfreq_vectors,
        "chroma_lowfreq_vectors": chroma_lowfreq_vectors,
        "chroma_midfreq_distributions": chroma_midfreq_distributions,
        "chroma_midfreq_log_energies": chroma_midfreq_log_energies,
    }
    for name, tensor in tensors_by_name.items():
        if len(prompt_hashes) != tensor.shape[0]:
            raise ValueError(
                f"Cache prompt_hashes length ({len(prompt_hashes)}) does not match "
                f"{name} rows ({tensor.shape[0]})."
            )

    metadata = {key: value for key, value in payload.items() if key not in required_keys}
    return WaveletReferenceCache(
        prompt_hashes=prompt_hashes,
        distributions={
            prompt_hash: distributions[index].detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        log_energies={
            prompt_hash: log_energies[index].detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        total_energies={
            prompt_hash: total_energies[index].reshape(()).detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        luminance_structure_distributions={
            prompt_hash: luminance_structure_distributions[index].detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        luminance_structure_log_energies={
            prompt_hash: luminance_structure_log_energies[index].detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        luminance_lowfreq_vectors={
            prompt_hash: luminance_lowfreq_vectors[index].detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        chroma_lowfreq_vectors={
            prompt_hash: chroma_lowfreq_vectors[index].detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        chroma_midfreq_distributions={
            prompt_hash: chroma_midfreq_distributions[index].detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        chroma_midfreq_log_energies={
            prompt_hash: chroma_midfreq_log_energies[index].detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        },
        metadata=metadata,
    )
