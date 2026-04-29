#!/usr/bin/env python
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

"""Serve DeepGHS anime aesthetic scoring as a simple HTTP reward service.

The server exposes:
    - ``GET /health``  -> ``{"status": "ok"}``
    - ``POST /load``   -> move the model to the inference device
    - ``POST /offload`` -> move the model back to CPU
    - ``POST /compute`` -> ``{"rewards": [...]}``

The request schema matches ``flow_factory.rewards.my_reward_remote``.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import threading
import time
from contextlib import nullcontext
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from timm.models.swin_transformer_v2 import SwinTransformerV2

LOGGER = logging.getLogger("anime_aesthetic_server")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT.parent / "models" / "anime_aesthetic" / "swinv2pv3_v0_448_ls0.2_x"
LABEL_QUALITY_ORDER = ("worst", "low", "normal", "good", "great", "best", "masterpiece")
HIGH_QUALITY_LABELS = ("great", "best", "masterpiece")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=18084, help="Port to bind.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the local DeepGHS anime aesthetic checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for model inference.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
        help="Torch dtype for model inference. CPU inference always uses float32.",
    )
    parser.add_argument(
        "--score-type",
        choices=(
            "percentile",
            "score",
            "score_normalized",
            "prob_masterpiece",
            "prob_high",
        ),
        default="percentile",
        help="Reward scoring rule.",
    )
    parser.add_argument(
        "--disable-data-parallel",
        action="store_true",
        help=(
            "Disable torch.nn.DataParallel. By default, CUDA inference uses all "
            "visible GPUs when more than one GPU is available."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def _decode_base64_image(image_data: str) -> Image.Image:
    """Decode a base64 string or data URL into a PIL image."""
    payload = image_data.split(",", maxsplit=1)[-1]
    binary = base64.b64decode(payload)
    with Image.open(BytesIO(binary)) as image:
        return image.copy()


def _resolve_media_payload(
    image_payload: Optional[list[str]],
    video_payload: Optional[list[list[str]]],
) -> list[Image.Image]:
    """Resolve image or video payloads into a list of images."""
    images: list[Image.Image] = []

    if image_payload:
        images.extend(_decode_base64_image(item) for item in image_payload)
        return images

    if not video_payload:
        return images

    for frames in video_payload:
        if not frames:
            raise ValueError("Encountered an empty video payload.")
        images.append(_decode_base64_image(frames[0]))
    return images


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _state_dict_without_profile_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove profiling tensors injected by model FLOP counting."""
    return {
        key: value
        for key, value in state_dict.items()
        if not key.endswith("total_ops") and not key.endswith("total_params")
    }


class AnimeAestheticService:
    """Wrapper around the DeepGHS anime aesthetic model."""

    def __init__(
        self,
        model_path: Path | str,
        device: str,
        dtype: str,
        score_type: str,
        disable_data_parallel: bool,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = self._resolve_device(device, disable_data_parallel)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.score_type = score_type

        self.device_ids = self._resolve_data_parallel_device_ids(disable_data_parallel)
        self.data_parallel_enabled = len(self.device_ids) > 1
        self._condition = threading.Condition(threading.Lock())
        self._active_compute_count = 0
        self._loaded_on_device = False

        self.meta = _load_json(self.model_path / "meta.json")
        self.labels = [str(label) for label in self.meta["labels"]]
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}
        self.image_size = int(self.meta.get("img_size", 448))
        self.model = self._load_model()
        self.model.eval()
        self.inference_model: torch.nn.Module = self.model

        self.score_weights = torch.tensor(
            [float(LABEL_QUALITY_ORDER.index(label)) for label in self.labels],
            dtype=torch.float32,
        )
        self.high_quality_indices = [self.label_to_index[label] for label in HIGH_QUALITY_LABELS]
        self.masterpiece_index = self.label_to_index["masterpiece"]
        self.percentile_x, self.percentile_y = self._load_percentile_samples()

        LOGGER.info(
            "Anime aesthetic device config: device=%s dtype=%s visible_cuda_devices=%s "
            "data_parallel=%s device_ids=%s score_type=%s",
            self.device,
            self.dtype,
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
            self.data_parallel_enabled,
            self.device_ids,
            self.score_type,
        )

    @staticmethod
    def _resolve_device(device: str, disable_data_parallel: bool) -> torch.device:
        """Resolve the primary inference device."""
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return resolved

        if not torch.cuda.is_available():
            raise ValueError("CUDA device was requested, but CUDA is not available.")

        if (
            not disable_data_parallel
            and torch.cuda.device_count() > 1
            and resolved.index not in {None, 0}
        ):
            raise ValueError(
                "Data-parallel anime aesthetic inference uses all visible CUDA devices and "
                "requires --device cuda or --device cuda:0. Use "
                "--disable-data-parallel for a single indexed CUDA device."
            )

        return torch.device("cuda:0" if resolved.index is None else resolved)

    def _resolve_data_parallel_device_ids(
        self,
        disable_data_parallel: bool,
    ) -> list[int]:
        """Return CUDA device ids used by torch.nn.DataParallel."""
        if self.device.type != "cuda" or disable_data_parallel:
            return []

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return []
        return list(range(device_count))

    def _load_model(self) -> torch.nn.Module:
        """Load the SwinV2 anime aesthetic checkpoint."""
        checkpoint_path = self.model_path / "model.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Anime aesthetic checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        arguments = checkpoint.get("arguments", {})
        labels = arguments.get("labels") or self.labels
        if list(labels) != self.labels:
            raise ValueError(
                f"Label mismatch between meta.json ({self.labels}) and model.ckpt ({labels})."
            )

        model = SwinTransformerV2(
            img_size=int(arguments.get("img_size", self.image_size)),
            patch_size=4,
            num_classes=len(self.labels),
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            window_size=14,
            drop_path_rate=float(arguments.get("drop_path_rate", 0.0)),
            strict_img_size=True,
        )
        state_dict = _state_dict_without_profile_keys(checkpoint["state_dict"])
        model.load_state_dict(state_dict, strict=True)
        return model

    def _load_percentile_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Load score-to-percentile calibration samples."""
        samples_path = self.model_path / "samples.npz"
        if not samples_path.exists():
            if self.score_type == "percentile":
                raise FileNotFoundError(
                    f"score_type='percentile' requires calibration samples: {samples_path}"
                )
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        stacked = np.load(samples_path)["arr_0"]
        if stacked.shape[0] != 2:
            raise ValueError(
                f"Expected samples.npz arr_0 shape (2, N), got {tuple(stacked.shape)}."
            )
        return stacked[0].astype(np.float32), stacked[1].astype(np.float32)

    def ensure_on_device(self) -> None:
        """Move the anime aesthetic model to its inference device."""
        with self._condition:
            self._ensure_on_device_locked()

    def _ensure_on_device_locked(self) -> None:
        """Move the model to its inference device while holding the condition."""
        if self._loaded_on_device:
            return
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        if self.data_parallel_enabled:
            self.inference_model = torch.nn.DataParallel(
                self.model,
                device_ids=self.device_ids,
                output_device=self.device_ids[0],
            )
            self.inference_model.eval()
        else:
            self.inference_model = self.model
        self._loaded_on_device = True

    def offload_to_cpu(self) -> None:
        """Move the anime aesthetic model to CPU and release CUDA cache."""
        with self._condition:
            while self._active_compute_count > 0:
                self._condition.wait()
            self._offload_to_cpu_locked()

    def _offload_to_cpu_locked(self) -> None:
        """Move the model to CPU while holding the condition."""
        if not self._loaded_on_device:
            return
        self.inference_model = self.model
        self.model.to("cpu")
        self.model.eval()
        self._loaded_on_device = False
        if self.device.type == "cuda":
            for device_id in self.device_ids or [self.device.index or 0]:
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()

    def _enter_compute(self) -> None:
        """Mark a compute request active and ensure the model is loaded."""
        with self._condition:
            self._ensure_on_device_locked()
            self._active_compute_count += 1

    def _exit_compute(self) -> None:
        """Mark a compute request complete."""
        with self._condition:
            self._active_compute_count -= 1
            self._condition.notify_all()

    def _get_autocast_context(self):
        """Create a fresh autocast context for one forward pass."""
        if self.device.type == "cuda" and self.dtype != torch.float32:
            return torch.autocast(device_type=self.device.type, dtype=self.dtype)
        return nullcontext()

    def _preprocess_images(self, images: list[Image.Image]) -> torch.Tensor:
        """Convert PIL images into anime aesthetic model input tensors."""
        tensors = []
        mean = torch.tensor(0.5, dtype=torch.float32).view(1, 1, 1)
        std = torch.tensor(0.5, dtype=torch.float32).view(1, 1, 1)

        for image in images:
            image = ImageOps.exif_transpose(image)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                canvas = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(canvas, image.convert("RGBA")).convert("RGB")
            else:
                image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append((tensor - mean) / std)

        return torch.stack(tensors, dim=0)

    def _score_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Convert class probabilities into reward scores."""
        score_weights = self.score_weights.to(device=probabilities.device)
        expected_score = (probabilities * score_weights).sum(dim=-1)

        if self.score_type == "score":
            return expected_score
        if self.score_type == "score_normalized":
            return expected_score / float(len(LABEL_QUALITY_ORDER) - 1)
        if self.score_type == "prob_masterpiece":
            return probabilities[:, self.masterpiece_index]
        if self.score_type == "prob_high":
            return probabilities[:, self.high_quality_indices].sum(dim=-1)
        if self.score_type != "percentile":
            raise ValueError(f"Unsupported score_type: {self.score_type}")

        score_array = expected_score.detach().cpu().float().numpy()
        percentile = np.interp(score_array, self.percentile_x, self.percentile_y)
        return torch.from_numpy(percentile).to(device=probabilities.device, dtype=torch.float32)

    def compute_rewards(
        self,
        image_payload: Optional[list[str]],
        video_payload: Optional[list[list[str]]],
    ) -> list[float]:
        """Compute rewards for a batch of images or videos."""
        start_time = time.perf_counter()
        images = _resolve_media_payload(image_payload, video_payload)
        if not images:
            raise ValueError("At least one image or video input is required.")

        LOGGER.info(
            "Anime aesthetic compute request: samples=%s score_type=%s data_parallel=%s "
            "active_gpus=%s",
            len(images),
            self.score_type,
            self.data_parallel_enabled,
            len(self.device_ids) if self.data_parallel_enabled else 1,
        )

        rewards: list[float] = []
        self._enter_compute()
        try:
            with torch.inference_mode():
                batch = self._preprocess_images(images).to(
                    device=self.device,
                    dtype=self.dtype,
                )

                with self._get_autocast_context():
                    logits = self.inference_model(batch)

                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                probabilities = logits.float().softmax(dim=-1)
                rewards = self._score_probabilities(probabilities).detach().cpu().float().tolist()
        finally:
            self._exit_compute()

        LOGGER.info(
            "Anime aesthetic compute complete: samples=%s data_parallel=%s "
            "active_gpus=%s elapsed_s=%.3f",
            len(images),
            self.data_parallel_enabled,
            len(self.device_ids) if self.data_parallel_enabled else 1,
            time.perf_counter() - start_time,
        )
        return rewards


class RewardRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler bound to a reward service instance."""

    service: AnimeAestheticService

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        """Write a JSON response."""
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests."""
        if self.path == "/health":
            self._send_json(
                {
                    "status": "ok",
                    "loaded_on_device": self.service._loaded_on_device,
                    "score_type": self.service.score_type,
                    "labels": self.service.labels,
                    "image_size": self.service.image_size,
                    "data_parallel": self.service.data_parallel_enabled,
                    "active_gpus": (
                        len(self.service.device_ids) if self.service.data_parallel_enabled else 1
                    ),
                    "device_ids": self.service.device_ids,
                }
            )
            return
        self._send_json(
            {"error": f"Unsupported path: {self.path}"},
            status=HTTPStatus.NOT_FOUND,
        )

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST requests."""
        if self.path == "/load":
            try:
                self.service.ensure_on_device()
                self._send_json({"status": "loaded"})
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to load anime aesthetic model.")
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if self.path == "/offload":
            try:
                self.service.offload_to_cpu()
                self._send_json({"status": "offloaded"})
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to offload anime aesthetic model.")
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if self.path != "/compute":
            self._send_json(
                {"error": f"Unsupported path: {self.path}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            rewards = self.service.compute_rewards(
                image_payload=payload.get("image"),
                video_payload=payload.get("video"),
            )
            self._send_json({"rewards": rewards})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to compute anime aesthetic rewards.")
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Redirect HTTP logs through the standard logger."""
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    """Start the reward server."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    RewardRequestHandler.service = AnimeAestheticService(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        score_type=args.score_type,
        disable_data_parallel=args.disable_data_parallel,
    )

    server = ThreadingHTTPServer((args.host, args.port), RewardRequestHandler)
    LOGGER.info(
        "Starting anime aesthetic reward server on http://%s:%s using %s",
        args.host,
        args.port,
        args.model_path,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping anime aesthetic reward server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
