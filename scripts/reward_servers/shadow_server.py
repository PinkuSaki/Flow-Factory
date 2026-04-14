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
    - ``POST /compute`` -> ``{"rewards": [...]}``

The request schema matches ``flow_factory.rewards.my_reward_remote``.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import importlib.machinery
import sys
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

from transformers import ViTForImageClassification, ViTImageProcessor


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
        "--max-batch-size",
        type=int,
        default=16,
        help="Maximum batch size per forward pass.",
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


class AestheticShadowService:
    """Wrapper around the Aesthetic Shadow model."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        dtype: str,
        score_type: str,
        max_batch_size: int,
    ) -> None:
        self.model_path = model_path
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.score_type = score_type
        self.max_batch_size = max_batch_size

        self.processor = ViTImageProcessor.from_pretrained(str(model_path))
        self.model = ViTForImageClassification.from_pretrained(
            str(model_path),
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        label_to_id = getattr(self.model.config, "label2id", {}) or {}
        self.hq_index = int(label_to_id.get("hq", 0))
        self.lq_index = int(label_to_id.get("lq", 1 if self.hq_index == 0 else 0))

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
        images = _resolve_media_payload(image_payload, video_payload)
        if not images:
            raise ValueError("At least one image or video input is required.")

        rewards: list[float] = []

        with torch.inference_mode():
            for start in range(0, len(images), self.max_batch_size):
                batch_images = images[start : start + self.max_batch_size]
                inputs = self.processor(images=batch_images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                with self._get_autocast_context():
                    logits = self.model(**inputs).logits

                if self.score_type == "logit_margin":
                    batch_scores = logits[:, self.hq_index] - logits[:, self.lq_index]
                else:
                    batch_scores = logits.softmax(dim=-1)[:, self.hq_index]

                rewards.extend(batch_scores.detach().cpu().float().tolist())

        return rewards


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
            self._send_json({"status": "ok"})
            return
        self._send_json({"error": f"Unsupported path: {self.path}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST requests."""
        if self.path != "/compute":
            self._send_json({"error": f"Unsupported path: {self.path}"}, status=HTTPStatus.NOT_FOUND)
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
        max_batch_size=args.max_batch_size,
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
        server.server_close()


if __name__ == "__main__":
    main()
