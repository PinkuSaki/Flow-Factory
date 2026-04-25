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
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps


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


def load_reference_cache(cache_path: Path) -> dict[str, torch.Tensor]:
    """Load a prompt-hash reference embedding cache."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Reference embedding cache does not exist: {cache_path}")

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
        return {
            str(prompt_hash): embeddings[index].detach().cpu()
            for index, prompt_hash in enumerate(prompt_hashes)
        }

    if isinstance(payload, dict):
        return {
            str(prompt_hash): torch.as_tensor(embedding, dtype=torch.float32).detach().cpu()
            for prompt_hash, embedding in payload.items()
        }

    raise ValueError(
        "Unsupported reference cache format. Expected either "
        "{'prompt_hashes': [...], 'embeddings': Tensor} or {hash: embedding}."
    )


def save_reference_cache(
    cache_path: Path,
    prompt_hashes: list[str],
    reference_embeddings: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> None:
    """Save prompt-hash reference embeddings to a torch cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompt_hashes": prompt_hashes,
            "embeddings": torch.stack(
                [reference_embeddings[prompt_hash] for prompt_hash in prompt_hashes]
            ),
            **metadata,
        },
        cache_path,
    )


def _load_model_config(model_path: Path) -> dict[str, Any]:
    """Load WD timm model configuration."""
    config_path = model_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Pad an RGB image to square using a white background."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


class WDEVA02EmbeddingModel:
    """Thin WD EVA02 embedding extractor wrapper."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str,
        max_batch_size: int,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.max_batch_size = max_batch_size

        self.model_config = _load_model_config(model_path)
        pretrained_cfg = self.model_config.get("pretrained_cfg", {})
        self.image_size = int(pretrained_cfg.get("input_size", [3, 448, 448])[-1])
        self.image_mean = tuple(
            float(value) for value in pretrained_cfg.get("mean", [0.5, 0.5, 0.5])
        )
        self.image_std = tuple(
            float(value) for value in pretrained_cfg.get("std", [0.5, 0.5, 0.5])
        )

        self.model = self._load_wd_model()

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

        model = timm.create_model(
            self.model_config["architecture"],
            pretrained=False,
            num_classes=int(self.model_config["num_classes"]),
            **self.model_config.get("model_args", {}),
        )
        state_dict = load_file(str(self.model_path / "model.safetensors"), device="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def ensure_on_device(self) -> None:
        """Move the WD model to its inference device."""
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def offload_to_cpu(self) -> None:
        """Move the WD model to CPU and release CUDA cache when applicable."""
        self.model.to("cpu")
        if self.device.type == "cuda":
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
            image = _pad_to_square(image)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append((tensor - mean) / std)

        return torch.stack(tensors, dim=0)

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """Pool raw model features when the timm head API is unavailable."""
        if features.ndim == 4:
            return features.mean(dim=(2, 3))
        if features.ndim == 3:
            return features.mean(dim=1)
        if features.ndim == 2:
            return features
        raise ValueError(f"Unsupported WD feature shape: {tuple(features.shape)}")

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Extract normalized WD image embeddings."""
        embeddings = []
        for start in range(0, len(images), self.max_batch_size):
            batch_images = images[start : start + self.max_batch_size]
            batch = self._preprocess_images(batch_images).to(
                device=self.device,
                dtype=self.dtype,
            )

            with self._get_autocast_context():
                if hasattr(self.model, "forward_features"):
                    features = self.model.forward_features(batch)
                    if hasattr(self.model, "forward_head"):
                        try:
                            features = self.model.forward_head(features, pre_logits=True)
                        except TypeError:
                            features = self.model.forward_head(features)
                    else:
                        features = self._pool_features(features)
                else:
                    features = self.model(batch)

            if isinstance(features, (tuple, list)):
                features = features[0]
            if not isinstance(features, torch.Tensor):
                raise TypeError(
                    f"Expected WD features to be a tensor, got {type(features).__name__}."
                )
            if features.ndim != 2:
                features = self._pool_features(features)

            embeddings.append(F.normalize(features.float(), dim=-1).cpu())

        return torch.cat(embeddings, dim=0)
