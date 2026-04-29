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

"""Serve Aesthetic Shadow as a simple HTTP reward service.

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
import importlib.machinery
import json
import logging
import sys
import threading
import time
import types
from contextlib import nullcontext
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image


def _install_torchvision_stub() -> None:
    """Install a tiny torchvision stub for environments with broken torchvision builds."""

    class InterpolationMode(Enum):
        """Minimal enum required by transformers image utilities."""

        NEAREST = 0
        NEAREST_EXACT = 0
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 1

    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.__path__ = []
    torchvision_module.__spec__ = importlib.machinery.ModuleSpec(
        name="torchvision",
        loader=None,
        is_package=True,
    )

    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_module.InterpolationMode = InterpolationMode
    transforms_module.__spec__ = importlib.machinery.ModuleSpec(
        name="torchvision.transforms",
        loader=None,
        is_package=False,
    )

    io_module = types.ModuleType("torchvision.io")
    io_module.__spec__ = importlib.machinery.ModuleSpec(
        name="torchvision.io",
        loader=None,
        is_package=False,
    )

    torchvision_module.transforms = transforms_module
    torchvision_module.io = io_module

    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module
    sys.modules["torchvision.io"] = io_module


try:
    import torchvision  # noqa: F401
except Exception:  # noqa: BLE001
    _install_torchvision_stub()

from ddp_worker_pool import DDPWorkerPool
from transformers import AutoConfig, ViTForImageClassification, ViTImageProcessor

LOGGER = logging.getLogger("shadow_server")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=18081, help="Port to bind.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/reward_models/aesthetic-shadow-v2-backup"),
        help="Path to the local Aesthetic Shadow checkpoint.",
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
        help="Torch dtype for model inference.",
    )
    parser.add_argument(
        "--score-type",
        choices=("prob_hq", "logit_margin"),
        default="prob_hq",
        help="Reward scoring rule.",
    )
    parser.add_argument(
        "--disable-ddp",
        action="store_true",
        help=(
            "Disable server-side DDP. By default, CUDA inference starts one "
            "worker process per visible GPU when more than one GPU is available."
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
    return Image.open(BytesIO(binary)).convert("RGB")


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


def _count_media_payload(
    image_payload: Optional[list[str]],
    video_payload: Optional[list[list[str]]],
) -> int:
    """Count image or video samples without decoding image bytes."""
    if image_payload:
        return len(image_payload)

    if not video_payload:
        return 0

    for frames in video_payload:
        if not frames:
            raise ValueError("Encountered an empty video payload.")
    return len(video_payload)


class AestheticShadowDDPWorker:
    """DDP worker for one Aesthetic Shadow GPU rank."""

    def __init__(
        self,
        *,
        local_rank: int,
        world_size: int,
        model_path: Path,
        dtype: str,
        score_type: str,
    ) -> None:
        self.local_rank = local_rank
        self.world_size = world_size
        self.model_path = model_path
        self.device = torch.device(f"cuda:{local_rank}")
        self.dtype = getattr(torch, dtype)
        self.score_type = score_type

        self.processor = ViTImageProcessor.from_pretrained(str(model_path))
        model = ViTForImageClassification.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
        )
        model.to(device=self.device, dtype=self.dtype)
        model.eval()
        self.inference_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        self.inference_model.eval()

        label_to_id = getattr(model.config, "label2id", {}) or {}
        self.hq_index = int(label_to_id.get("hq", 0))
        self.lq_index = int(label_to_id.get("lq", 1 if self.hq_index == 0 else 0))

    def _get_autocast_context(self):
        """Create a fresh autocast context for one forward pass."""
        if self.dtype != torch.float32:
            return torch.autocast(device_type="cuda", dtype=self.dtype)
        return nullcontext()

    def compute(self, payload: dict[str, Any]) -> list[float]:
        """Compute one DDP rank shard."""
        images = _resolve_media_payload(payload.get("image"), payload.get("video"))
        if not images:
            raise ValueError("At least one image or video input is required.")

        with torch.inference_mode():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with self._get_autocast_context():
                logits = self.inference_model(**inputs, return_dict=False)[0]

            if self.score_type == "logit_margin":
                scores = logits[:, self.hq_index] - logits[:, self.lq_index]
            else:
                scores = logits.softmax(dim=-1)[:, self.hq_index]

            return scores.detach().cpu().float().tolist()

    def close(self) -> None:
        """Release CUDA memory held by this worker."""
        del self.inference_model
        torch.cuda.empty_cache()


class AestheticShadowService:
    """Wrapper around the Aesthetic Shadow model."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str,
        score_type: str,
        disable_ddp: bool,
    ) -> None:
        self.model_path = model_path
        self.device = self._resolve_device(device, disable_ddp)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.dtype_name = dtype if self.device.type == "cuda" else "float32"
        self.score_type = score_type
        self.device_ids = self._resolve_ddp_device_ids(disable_ddp)
        self.ddp_enabled = len(self.device_ids) > 1
        self.ddp_pool: Optional[DDPWorkerPool] = None
        self._condition = threading.Condition(threading.Lock())
        self._active_compute_count = 0
        self._loaded_on_device = False

        config = AutoConfig.from_pretrained(str(model_path))
        label_to_id = getattr(config, "label2id", {}) or {}
        self.hq_index = int(label_to_id.get("hq", 0))
        self.lq_index = int(label_to_id.get("lq", 1 if self.hq_index == 0 else 0))

        if self.ddp_enabled:
            self.processor = None
            self.model = None
            self.inference_model = None
            self.ddp_pool = DDPWorkerPool(
                worker_factory=AestheticShadowDDPWorker,
                worker_kwargs={
                    "model_path": model_path,
                    "dtype": self.dtype_name,
                    "score_type": score_type,
                },
                world_size=len(self.device_ids),
                batch_keys=("image", "video"),
                logger=LOGGER,
            )
        else:
            self.processor = ViTImageProcessor.from_pretrained(str(model_path))
            self.model = ViTForImageClassification.from_pretrained(
                str(model_path),
                torch_dtype=self.dtype,
            )
            self.model.eval()
            self.inference_model: Optional[torch.nn.Module] = self.model

        LOGGER.info(
            "Aesthetic Shadow device config: device=%s dtype=%s visible_cuda_devices=%s "
            "ddp=%s device_ids=%s",
            self.device,
            self.dtype,
            torch.cuda.device_count() if torch.cuda.is_available() else 0,
            self.ddp_enabled,
            self.device_ids,
        )

    @staticmethod
    def _resolve_device(device: str, disable_ddp: bool) -> torch.device:
        """Resolve the primary inference device."""
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return resolved

        if not torch.cuda.is_available():
            raise ValueError("CUDA device was requested, but CUDA is not available.")

        if not disable_ddp and torch.cuda.device_count() > 1 and resolved.index not in {None, 0}:
            raise ValueError(
                "DDP shadow inference uses all visible CUDA devices and "
                "requires --device cuda or --device cuda:0. Use "
                "--disable-ddp for a single indexed CUDA device."
            )

        return torch.device("cuda:0" if resolved.index is None else resolved)

    def _resolve_ddp_device_ids(
        self,
        disable_ddp: bool,
    ) -> list[int]:
        """Return CUDA device ids used by DDP workers."""
        if self.device.type != "cuda" or disable_ddp:
            return []

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return []
        return list(range(device_count))

    def ensure_on_device(self) -> None:
        """Move the Aesthetic Shadow model to its inference device."""
        with self._condition:
            self._ensure_on_device_locked()

    def _ensure_on_device_locked(self) -> None:
        """Move the model to its inference device while holding the condition."""
        if self._loaded_on_device:
            return
        if self.ddp_enabled:
            if self.ddp_pool is None:
                raise RuntimeError("DDP worker pool is not initialized.")
            self.ddp_pool.start()
            self._loaded_on_device = True
            return
        if self.model is None:
            raise RuntimeError("Aesthetic Shadow model is not initialized.")
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()
        self.inference_model = self.model
        self._loaded_on_device = True

    def offload_to_cpu(self) -> None:
        """Move the Aesthetic Shadow model to CPU and release CUDA cache."""
        with self._condition:
            while self._active_compute_count > 0:
                self._condition.wait()
            self._offload_to_cpu_locked()

    def _offload_to_cpu_locked(self) -> None:
        """Move the model to CPU while holding the condition."""
        if not self._loaded_on_device:
            return
        if self.ddp_enabled:
            if self.ddp_pool is not None:
                self.ddp_pool.shutdown()
            self._loaded_on_device = False
            return
        if self.model is None:
            raise RuntimeError("Aesthetic Shadow model is not initialized.")
        self.inference_model = self.model
        self.model.to("cpu")
        self.model.eval()
        self._loaded_on_device = False
        if self.device.type == "cuda":
            for device_id in [self.device.index or 0]:
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

    def compute_rewards(
        self,
        image_payload: Optional[list[str]],
        video_payload: Optional[list[list[str]]],
    ) -> list[float]:
        """Compute rewards for a batch of images or videos."""
        start_time = time.perf_counter()
        sample_count = _count_media_payload(image_payload, video_payload)
        if sample_count <= 0:
            raise ValueError("At least one image or video input is required.")

        LOGGER.info(
            "Shadow compute request: samples=%s score_type=%s ddp=%s active_gpus=%s",
            sample_count,
            self.score_type,
            self.ddp_enabled,
            len(self.device_ids) if self.ddp_enabled else 1,
        )

        self._enter_compute()
        try:
            if self.ddp_enabled:
                if self.ddp_pool is None:
                    raise RuntimeError("DDP worker pool is not initialized.")
                rewards = self.ddp_pool.compute(
                    payload={"image": image_payload, "video": video_payload},
                    total_size=sample_count,
                )
            else:
                if self.processor is None or self.inference_model is None:
                    raise RuntimeError("Aesthetic Shadow model is not initialized.")
                images = _resolve_media_payload(image_payload, video_payload)
                with torch.inference_mode():
                    inputs = self.processor(images=images, return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}

                    with self._get_autocast_context():
                        logits = self.inference_model(**inputs, return_dict=False)[0]

                    if self.score_type == "logit_margin":
                        scores = logits[:, self.hq_index] - logits[:, self.lq_index]
                    else:
                        scores = logits.softmax(dim=-1)[:, self.hq_index]

                    rewards = scores.detach().cpu().float().tolist()
        finally:
            self._exit_compute()

        LOGGER.info(
            "Shadow compute complete: samples=%s ddp=%s active_gpus=%s elapsed_s=%.3f",
            sample_count,
            self.ddp_enabled,
            len(self.device_ids) if self.ddp_enabled else 1,
            time.perf_counter() - start_time,
        )
        return rewards

    def close(self) -> None:
        """Release server resources."""
        with self._condition:
            self._offload_to_cpu_locked()


class RewardRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler bound to a reward service instance."""

    service: AestheticShadowService

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
                    "ddp": self.service.ddp_enabled,
                    "parallel_mode": "ddp" if self.service.ddp_enabled else "single",
                    "active_gpus": len(self.service.device_ids) if self.service.ddp_enabled else 1,
                    "device_ids": self.service.device_ids,
                    "loaded": self.service._loaded_on_device,
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
                LOGGER.exception("Failed to load shadow model.")
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if self.path == "/offload":
            try:
                self.service.offload_to_cpu()
                self._send_json({"status": "offloaded"})
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to offload shadow model.")
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
            LOGGER.exception("Failed to compute shadow rewards.")
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Redirect HTTP logs through the standard logger."""
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    """Start the reward server."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    RewardRequestHandler.service = AestheticShadowService(
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        score_type=args.score_type,
        disable_ddp=args.disable_ddp,
    )

    server = ThreadingHTTPServer((args.host, args.port), RewardRequestHandler)
    LOGGER.info(
        "Starting Aesthetic Shadow reward server on http://%s:%s using %s",
        args.host,
        args.port,
        args.model_path,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping Aesthetic Shadow reward server.")
    finally:
        RewardRequestHandler.service.close()
        server.server_close()


if __name__ == "__main__":
    main()
