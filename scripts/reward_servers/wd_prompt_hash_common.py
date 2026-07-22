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

"""Shared WD EVA02 embedding utilities for prompt-hash reward scripts."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor, ImageOps

WD_MODEL_FINGERPRINT_FILES = ("config.json", "preprocess.json", "model.safetensors")
WD_CONFIGURED_TRANSFORM_TYPES = (
    "pad_to_size",
    "resize",
    "center_crop",
    "maybe_to_tensor",
    "normalize",
)
PIL_INTERPOLATIONS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "box": Image.Resampling.BOX,
    "hamming": Image.Resampling.HAMMING,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen=True)
class WDReferenceCache:
    """Prompt-hash indexed WD reference features loaded from a torch cache."""

    prompt_hashes: list[str]
    embeddings: dict[str, torch.Tensor]
    probabilities: Optional[dict[str, torch.Tensor]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def require_probabilities(self) -> dict[str, torch.Tensor]:
        """Return reference probabilities or fail with a rebuild hint."""
        if self.probabilities is None:
            raise ValueError(
                "The WD reference cache does not contain probability vectors. "
                "Rebuild it with build_wd_prompt_hash_cache.py before using "
                "probability-based WD score types."
            )
        return self.probabilities


@dataclass(frozen=True)
class WDImageOutputs:
    """WD image embedding and class-probability outputs."""

    embeddings: torch.Tensor
    probabilities: torch.Tensor


@dataclass(frozen=True)
class WDPreprocessSpec:
    """Describe the deterministic WD image preprocessing pipeline."""

    source: str
    pad_size: Optional[tuple[int, int]]
    pad_interpolation: Image.Resampling
    pad_background: tuple[int, int, int]
    resize_size: tuple[int, int]
    resize_interpolation: Image.Resampling
    crop_size: tuple[int, int]
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]


def get_default_wd_dtype() -> str:
    """Select the fastest supported default WD inference dtype.

    Returns:
        ``float32`` on CPU-only hosts, otherwise ``bfloat16`` when supported or ``float16``.
    """
    if not torch.cuda.is_available():
        return "float32"
    supports_native_bfloat16 = all(
        torch.cuda.get_device_capability(device_index)[0] >= 8
        for device_index in range(torch.cuda.device_count())
    )
    return "bfloat16" if supports_native_bfloat16 else "float16"


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


def load_reference_cache_payload(cache_path: Path) -> WDReferenceCache:
    """Load a prompt-hash reference cache payload."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Reference cache does not exist: {cache_path}")

    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location="cpu")

    if isinstance(payload, dict) and "prompt_hashes" in payload and "embeddings" in payload:
        prompt_hashes = payload["prompt_hashes"]
        embeddings = torch.as_tensor(payload["embeddings"], dtype=torch.float32)
        if len(prompt_hashes) != embeddings.shape[0]:
            raise ValueError(
                f"Cache prompt_hashes length ({len(prompt_hashes)}) does not match "
                f"embeddings rows ({embeddings.shape[0]})."
            )
        probabilities = None
        if "probabilities" in payload:
            probability_tensor = torch.as_tensor(payload["probabilities"], dtype=torch.float32)
            if len(prompt_hashes) != probability_tensor.shape[0]:
                raise ValueError(
                    f"Cache prompt_hashes length ({len(prompt_hashes)}) does not match "
                    f"probabilities rows ({probability_tensor.shape[0]})."
                )
            probabilities = {
                str(prompt_hash): probability_tensor[index].detach().cpu()
                for index, prompt_hash in enumerate(prompt_hashes)
            }

        metadata = {
            str(key): value
            for key, value in payload.items()
            if key not in {"prompt_hashes", "embeddings", "probabilities"}
        }
        return WDReferenceCache(
            prompt_hashes=[str(prompt_hash) for prompt_hash in prompt_hashes],
            embeddings={
                str(prompt_hash): embeddings[index].detach().cpu()
                for index, prompt_hash in enumerate(prompt_hashes)
            },
            probabilities=probabilities,
            metadata=metadata,
        )

    if isinstance(payload, dict):
        prompt_hashes = [str(prompt_hash) for prompt_hash in payload.keys()]
        return WDReferenceCache(
            prompt_hashes=prompt_hashes,
            embeddings={
                str(prompt_hash): torch.as_tensor(embedding, dtype=torch.float32).detach().cpu()
                for prompt_hash, embedding in payload.items()
            },
        )

    raise ValueError(
        "Unsupported reference cache format. Expected either "
        "{'prompt_hashes': [...], 'embeddings': Tensor, 'probabilities': Tensor} or "
        "{hash: embedding}."
    )


def load_reference_cache(cache_path: Path) -> dict[str, torch.Tensor]:
    """Load prompt-hash reference embeddings from a cache."""
    return load_reference_cache_payload(cache_path).embeddings


def soft_jaccard_score(
    generated_probabilities: torch.Tensor,
    reference_probabilities: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute soft Jaccard similarity between two probability vectors."""
    if generated_probabilities.shape != reference_probabilities.shape:
        raise ValueError(
            "Probability shape mismatch: "
            f"generated={tuple(generated_probabilities.shape)} "
            f"reference={tuple(reference_probabilities.shape)}"
        )

    intersection = torch.minimum(generated_probabilities, reference_probabilities).sum(dim=-1)
    union = torch.maximum(generated_probabilities, reference_probabilities).sum(dim=-1)
    return intersection / union.clamp_min(eps)


def negative_binary_cross_entropy_score(
    generated_probabilities: torch.Tensor,
    reference_probabilities: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute negative mean binary cross entropy against reference probabilities."""
    if generated_probabilities.shape != reference_probabilities.shape:
        raise ValueError(
            "Probability shape mismatch: "
            f"generated={tuple(generated_probabilities.shape)} "
            f"reference={tuple(reference_probabilities.shape)}"
        )

    generated = generated_probabilities.clamp(min=eps, max=1.0 - eps)
    reference = reference_probabilities.clamp(min=0.0, max=1.0)
    loss = F.binary_cross_entropy(generated, reference, reduction="none").mean(dim=-1)
    return -loss


def wd_distribution_reward(
    real_probabilities: torch.Tensor,
    fake_probabilities: torch.Tensor,
    recall_gamma: float = 1.5,
    jaccard_gamma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute WD probability-distribution reward with recall and soft Jaccard terms."""
    if real_probabilities.shape != fake_probabilities.shape:
        raise ValueError(
            "Probability shape mismatch: "
            f"real={tuple(real_probabilities.shape)} fake={tuple(fake_probabilities.shape)}"
        )

    minimum_probabilities = torch.minimum(real_probabilities, fake_probabilities)
    maximum_probabilities = torch.maximum(real_probabilities, fake_probabilities)

    recall_weights = real_probabilities.pow(recall_gamma)
    recall = (recall_weights * minimum_probabilities).sum(dim=-1) / (
        (recall_weights * real_probabilities).sum(dim=-1) + eps
    )

    jaccard_weights = maximum_probabilities.pow(jaccard_gamma)
    jaccard = (jaccard_weights * minimum_probabilities).sum(dim=-1) / (
        (jaccard_weights * maximum_probabilities).sum(dim=-1) + eps
    )

    return 0.7 * recall + 0.3 * jaccard


def save_reference_cache(
    cache_path: Path,
    prompt_hashes: list[str],
    reference_embeddings: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    reference_probabilities: Optional[dict[str, torch.Tensor]] = None,
) -> None:
    """Save prompt-hash reference embeddings and optional probabilities to a torch cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "prompt_hashes": prompt_hashes,
        "embeddings": torch.stack(
            [reference_embeddings[prompt_hash] for prompt_hash in prompt_hashes]
        ),
        **metadata,
    }
    if reference_probabilities is not None:
        payload["probabilities"] = torch.stack(
            [reference_probabilities[prompt_hash] for prompt_hash in prompt_hashes]
        )
    torch.save(payload, cache_path)


def _load_model_config(model_path: Path) -> dict[str, Any]:
    """Load WD timm model configuration."""
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"WD model config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_wd_model_fingerprint(model_path: Path) -> str:
    """Compute a stable fingerprint over WD weights and preprocessing metadata.

    Args:
        model_path: Local WD checkpoint directory.

    Returns:
        SHA256 digest covering the model config, optional preprocessing config, and weights.
    """
    required_paths = (model_path / "config.json", model_path / "model.safetensors")
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "WD model fingerprint requires config.json and model.safetensors. "
            f"Missing: {', '.join(str(path) for path in missing_paths)}"
        )

    digest = hashlib.sha256()
    for filename in WD_MODEL_FINGERPRINT_FILES:
        path = model_path / filename
        if not path.exists():
            continue
        digest.update(filename.encode("utf-8"))
        digest.update(path.stat().st_size.to_bytes(8, byteorder="big", signed=False))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def validate_reference_cache_for_model(
    reference_cache: WDReferenceCache,
    model_path: Path,
) -> str:
    """Validate that a WD reference cache was built with the active model.

    Args:
        reference_cache: Loaded prompt-hash reference cache.
        model_path: Local WD checkpoint directory used for generated-image inference.

    Returns:
        Fingerprint of the validated WD checkpoint.
    """
    model_config = _load_model_config(model_path)
    model_fingerprint = compute_wd_model_fingerprint(model_path)
    cached_fingerprint = reference_cache.metadata.get("model_fingerprint")
    if cached_fingerprint is None:
        raise ValueError(
            "The WD reference cache does not contain model_fingerprint metadata. "
            "Rebuild it with build_wd_prompt_hash_cache.py before starting the WD server."
        )
    if str(cached_fingerprint) != model_fingerprint:
        raise ValueError(
            "The WD reference cache was built with a different model or preprocessing config: "
            f"cache_fingerprint={cached_fingerprint} model_fingerprint={model_fingerprint}. "
            "Rebuild the cache with the same --model-path used by the reward server."
        )

    expected_embedding_dim = int(model_config["num_features"])
    embedding_shapes = {tuple(value.shape) for value in reference_cache.embeddings.values()}
    if embedding_shapes != {(expected_embedding_dim,)}:
        raise ValueError(
            "WD cache embedding shape mismatch: "
            f"cache_shapes={sorted(embedding_shapes)} expected=({expected_embedding_dim},)."
        )

    if reference_cache.probabilities is not None:
        expected_probability_dim = int(model_config["num_classes"])
        probability_shapes = {
            tuple(value.shape) for value in reference_cache.probabilities.values()
        }
        if probability_shapes != {(expected_probability_dim,)}:
            raise ValueError(
                "WD cache probability shape mismatch: "
                f"cache_shapes={sorted(probability_shapes)} "
                f"expected=({expected_probability_dim},)."
            )
    return model_fingerprint


def _parse_size(value: Any, field_name: str) -> tuple[int, int]:
    """Parse a positive two-dimensional size."""
    if isinstance(value, int):
        size = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        size = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"{field_name} must be an int or a two-item sequence, got {value!r}.")
    if min(size) <= 0:
        raise ValueError(f"{field_name} must be positive, got {size}.")
    return size


def _parse_interpolation(value: Any, field_name: str) -> Image.Resampling:
    """Parse a Pillow interpolation name."""
    interpolation_name = str(value).lower()
    if interpolation_name not in PIL_INTERPOLATIONS:
        raise ValueError(
            f"Unsupported {field_name} interpolation {value!r}. "
            f"Supported values: {sorted(PIL_INTERPOLATIONS)}."
        )
    return PIL_INTERPOLATIONS[interpolation_name]


def _parse_rgb_color(value: Any, field_name: str) -> tuple[int, int, int]:
    """Parse an RGB background color."""
    if isinstance(value, str):
        return ImageColor.getrgb(value)
    if isinstance(value, int):
        if not 0 <= value <= 255:
            raise ValueError(f"{field_name} integer must be in [0, 255], got {value}.")
        return (value, value, value)
    if isinstance(value, (list, tuple)) and len(value) in {3, 4}:
        rgb = tuple(int(channel) for channel in value[:3])
        if any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(f"{field_name} channels must be in [0, 255], got {rgb}.")
        return rgb
    raise ValueError(f"Unsupported {field_name} color: {value!r}.")


def _parse_channel_values(value: Any, field_name: str) -> tuple[float, float, float]:
    """Parse three floating-point channel values."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain three channel values, got {value!r}.")
    return tuple(float(channel) for channel in value)


def _load_preprocess_spec(
    model_path: Path,
    model_config: dict[str, Any],
) -> WDPreprocessSpec:
    """Load configured DBV4 preprocessing or the legacy WD square-padding pipeline."""
    pretrained_cfg = model_config.get("pretrained_cfg", {})
    input_size = pretrained_cfg.get("input_size", [3, 448, 448])
    if not isinstance(input_size, (list, tuple)) or len(input_size) != 3:
        raise ValueError(f"pretrained_cfg.input_size must have three values, got {input_size!r}.")
    legacy_resize_size = (int(input_size[2]), int(input_size[1]))
    legacy_mean = _parse_channel_values(
        pretrained_cfg.get("mean", [0.5, 0.5, 0.5]),
        "pretrained_cfg.mean",
    )
    legacy_std = _parse_channel_values(
        pretrained_cfg.get("std", [0.5, 0.5, 0.5]),
        "pretrained_cfg.std",
    )

    preprocess_path = model_path / "preprocess.json"
    if not preprocess_path.exists():
        return WDPreprocessSpec(
            source="legacy_square",
            pad_size=None,
            pad_interpolation=Image.Resampling.BILINEAR,
            pad_background=(255, 255, 255),
            resize_size=legacy_resize_size,
            resize_interpolation=_parse_interpolation(
                pretrained_cfg.get("interpolation", "bicubic"),
                "pretrained_cfg",
            ),
            crop_size=legacy_resize_size,
            image_mean=legacy_mean,
            image_std=legacy_std,
        )

    with preprocess_path.open("r", encoding="utf-8") as handle:
        preprocess_payload = json.load(handle)
    test_transforms = preprocess_payload.get("test")
    if not isinstance(test_transforms, list):
        raise ValueError(f"{preprocess_path} must contain a list-valued 'test' pipeline.")
    transform_types = tuple(transform.get("type") for transform in test_transforms)
    if transform_types != WD_CONFIGURED_TRANSFORM_TYPES:
        raise ValueError(
            f"Unsupported WD test preprocessing pipeline {transform_types}. "
            f"Expected {WD_CONFIGURED_TRANSFORM_TYPES}."
        )

    pad_transform, resize_transform, crop_transform, _, normalize_transform = test_transforms
    pad_size = _parse_size(pad_transform.get("size"), "pad_to_size.size")
    resize_height, resize_width = _parse_size(resize_transform.get("size"), "resize.size")
    crop_height, crop_width = _parse_size(crop_transform.get("size"), "center_crop.size")
    resize_size = (resize_width, resize_height)
    crop_size = (crop_width, crop_height)
    if resize_size != crop_size:
        raise ValueError(
            "WD configured preprocessing currently requires resize.size and center_crop.size "
            f"to match, got resize={resize_size} crop={crop_size}."
        )

    image_std = _parse_channel_values(normalize_transform.get("std"), "normalize.std")
    if any(value <= 0.0 for value in image_std):
        raise ValueError(f"normalize.std values must be positive, got {image_std}.")
    return WDPreprocessSpec(
        source="preprocess.json:test",
        pad_size=pad_size,
        pad_interpolation=_parse_interpolation(
            pad_transform.get("interpolation", "bilinear"),
            "pad_to_size",
        ),
        pad_background=_parse_rgb_color(
            pad_transform.get("background_color", "white"),
            "pad_to_size.background_color",
        ),
        resize_size=resize_size,
        resize_interpolation=_parse_interpolation(
            resize_transform.get("interpolation", "bilinear"),
            "resize",
        ),
        crop_size=crop_size,
        image_mean=_parse_channel_values(normalize_transform.get("mean"), "normalize.mean"),
        image_std=image_std,
    )


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and composite transparent pixels over white."""
    image = ImageOps.exif_transpose(image)
    has_transparency = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_transparency:
        return image.convert("RGB")

    foreground = image.convert("RGBA")
    background = Image.new("RGBA", foreground.size, (255, 255, 255, 255))
    background.alpha_composite(foreground)
    return background.convert("RGB")


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Pad an RGB image to square using a white background."""
    image = _convert_to_rgb(image)
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def _resize_and_pad_to_size(
    image: Image.Image,
    size: tuple[int, int],
    interpolation: Image.Resampling,
    background: tuple[int, int, int],
) -> Image.Image:
    """Resize an image to fit and center-pad it to an exact size."""
    target_width, target_height = size
    width, height = image.size
    ratio = min(target_width / width, target_height / height)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    image = image.resize((resized_width, resized_height), interpolation)
    canvas = Image.new("RGB", (target_width, target_height), background)
    canvas.paste(
        image,
        ((target_width - resized_width) // 2, (target_height - resized_height) // 2),
    )
    return canvas


class WDEVA02EmbeddingModel:
    """Thin WD EVA02 embedding extractor wrapper."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str,
        max_batch_size: Optional[int] = None,
    ) -> None:
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.max_batch_size = max_batch_size
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive or None, got {self.max_batch_size}.")

        self.model_config = _load_model_config(model_path)
        self.preprocess_spec = _load_preprocess_spec(model_path, self.model_config)
        if self.preprocess_spec.crop_size[0] != self.preprocess_spec.crop_size[1]:
            raise ValueError(
                f"WD EVA02 requires square model input, got {self.preprocess_spec.crop_size}."
            )
        self.image_size = self.preprocess_spec.crop_size[0]
        self.image_mean = self.preprocess_spec.image_mean
        self.image_std = self.preprocess_spec.image_std

        self.model = self._load_wd_model()
        self.output_model = WDEVA02OutputExtractor(self.model)
        self.inference_model: torch.nn.Module = self.output_model

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve the primary inference device."""
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return resolved

        if not torch.cuda.is_available():
            raise ValueError("CUDA device was requested, but CUDA is not available.")

        return torch.device("cuda:0" if resolved.index is None else resolved)

    def _load_wd_model(self) -> torch.nn.Module:
        """Load the WD EVA02 timm model from a local safetensors checkpoint."""
        try:
            import timm
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "WD prompt-hash reward scripts require `timm` and `safetensors`. "
                "Install them with `pip install timm safetensors`."
            ) from exc

        model_args = dict(self.model_config.get("model_args", {}))
        model_args.setdefault("global_pool", self.model_config.get("global_pool", "avg"))
        model = timm.create_model(
            self.model_config["architecture"],
            pretrained=False,
            num_classes=int(self.model_config["num_classes"]),
            **model_args,
        )
        state_dict = load_file(str(self.model_path / "model.safetensors"), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def ensure_on_device(self) -> None:
        """Move the WD model to its inference device."""
        self.output_model.to(device=self.device, dtype=self.dtype)
        self.output_model.eval()
        self.inference_model = self.output_model

    def wrap_ddp(self, local_rank: int) -> None:
        """Wrap the WD output extractor with DistributedDataParallel."""
        self.ensure_on_device()
        self.inference_model = torch.nn.parallel.DistributedDataParallel(
            self.output_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        self.inference_model.eval()

    def offload_to_cpu(self) -> None:
        """Move the WD model to CPU and release CUDA cache when applicable."""
        self.inference_model = self.output_model
        self.output_model.to("cpu")
        self.output_model.eval()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device.index or 0):
                torch.cuda.empty_cache()

    def _get_autocast_context(self):
        """Create a fresh autocast context for one forward pass."""
        if self.device.type == "cuda" and self.dtype != torch.float32:
            return torch.autocast(device_type=self.device.type, dtype=self.dtype)
        return nullcontext()

    def _preprocess_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Convert PIL images into WD model input tensors."""
        tensors = []
        mean = torch.tensor(self.image_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.image_std, dtype=torch.float32).view(3, 1, 1)

        for image in images:
            image = _convert_to_rgb(image)
            if self.preprocess_spec.pad_size is None:
                image = _pad_to_square(image)
            else:
                image = _resize_and_pad_to_size(
                    image=image,
                    size=self.preprocess_spec.pad_size,
                    interpolation=self.preprocess_spec.pad_interpolation,
                    background=self.preprocess_spec.pad_background,
                )
            image = image.resize(
                self.preprocess_spec.resize_size,
                self.preprocess_spec.resize_interpolation,
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append((tensor - mean) / std)

        return torch.stack(tensors, dim=0)

    @torch.inference_mode()
    def encode_image_outputs(self, images: list[Image.Image]) -> WDImageOutputs:
        """Extract normalized WD image embeddings and class probabilities."""
        embeddings = []
        probabilities = []
        batch_size = self.max_batch_size or len(images)
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch = self._preprocess_images(batch_images).to(
                device=self.device,
                dtype=self.dtype,
            )

            with self._get_autocast_context():
                features, logits = self.inference_model(batch)

            embeddings.append(F.normalize(features.float(), dim=-1).cpu())
            probabilities.append(torch.sigmoid(logits.float()).cpu())

        return WDImageOutputs(
            embeddings=torch.cat(embeddings, dim=0),
            probabilities=torch.cat(probabilities, dim=0),
        )

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Extract normalized WD image embeddings."""
        return self.encode_image_outputs(images).embeddings


class WDEVA02OutputExtractor(torch.nn.Module):
    """Expose WD embedding and classifier outputs through a regular forward method."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _pool_features(features: torch.Tensor) -> torch.Tensor:
        """Pool raw model features when the timm head API is unavailable."""
        if features.ndim == 4:
            return features.mean(dim=(2, 3))
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Unsupported WD feature shape: {tuple(features.shape)}")

    def _forward_head(self, features: torch.Tensor, *, pre_logits: bool) -> torch.Tensor:
        """Run the timm model head while preserving feature fallback behavior."""
        if not hasattr(self.model, "forward_head"):
            return self._pool_features(features)

        try:
            return self.model.forward_head(features, pre_logits=pre_logits)
        except TypeError:
            if pre_logits:
                return self._pool_features(features)
            return self.model.forward_head(features)

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract one batch of WD embeddings and logits."""
        if hasattr(self.model, "forward_features"):
            raw_features = self.model.forward_features(batch)
            if isinstance(raw_features, (tuple, list)):
                raw_features = raw_features[0]
            if not isinstance(raw_features, torch.Tensor):
                raise TypeError(
                    f"Expected WD features to be a tensor, got {type(raw_features).__name__}."
                )
            features = self._forward_head(raw_features, pre_logits=True)
            logits = self._forward_head(raw_features, pre_logits=False)
        else:
            logits = self.model(batch)
            features = logits

        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Expected WD features to be a tensor, got {type(features).__name__}.")
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected WD logits to be a tensor, got {type(logits).__name__}.")
        if features.ndim != 2:
            features = self._pool_features(features)
        if logits.ndim != 2:
            logits = self._pool_features(logits)
        return features, logits
