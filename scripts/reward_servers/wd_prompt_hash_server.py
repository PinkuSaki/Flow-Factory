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

"""Serve WD EVA02 prompt-hash reference similarity from a prebuilt cache.

The server exposes:
    - ``GET /health``  -> ``{"status": "ok", "reference_count": ...}``
    - ``POST /load``   -> move the WD model to the inference device
    - ``POST /offload`` -> move the WD model back to CPU
    - ``POST /compute`` -> ``{"rewards": [...]}``

Reference embeddings must be built before training with
``build_wd_prompt_hash_cache.py``. This training-time server only loads the
cache, hashes incoming prompts, encodes generated images, and returns cosine
similarity rewards.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
from ddp_worker_pool import DDPWorkerPool
from PIL import Image
from wd_prompt_hash_common import (
    WDEVA02EmbeddingModel,
    load_reference_cache,
    prompt_sha256,
)

LOGGER = logging.getLogger("wd_prompt_hash_server")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=18082, help="Port to bind.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("../models/wd-eva02-large-tagger-v3"),
        help="Path to the local WD EVA02 tagger checkpoint directory.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Prebuilt prompt-hash reference embedding cache.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for generated image embedding inference.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
        help="Torch dtype for CUDA inference. CPU inference always uses float32.",
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
    """Resolve image or video payloads into a list of generated images."""
    if image_payload:
        return [_decode_base64_image(item) for item in image_payload]

    if not video_payload:
        return []

    images: list[Image.Image] = []
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


class WDEVA02PromptHashDDPWorker:
    """DDP worker for one WD prompt-hash GPU rank."""

    def __init__(
        self,
        *,
        local_rank: int,
        world_size: int,
        model_path: Path,
        cache_path: Path,
        dtype: str,
    ) -> None:
        self.local_rank = local_rank
        self.world_size = world_size
        self.reference_embeddings = load_reference_cache(cache_path)
        self.encoder = WDEVA02EmbeddingModel(
            model_path=model_path,
            device=f"cuda:{local_rank}",
            dtype=dtype,
        )
        self.encoder.wrap_ddp(local_rank)

    def compute(self, payload: dict) -> list[float]:
        """Compute one DDP rank shard."""
        prompts = payload.get("prompt") or []
        images = _resolve_media_payload(
            image_payload=payload.get("image"),
            video_payload=payload.get("video"),
        )
        if not images:
            raise ValueError("At least one image or video input is required.")
        if len(prompts) != len(images):
            raise ValueError(
                f"Prompt/image length mismatch: prompts={len(prompts)} images={len(images)}"
            )

        prompt_hashes = [prompt_sha256(prompt) for prompt in prompts]
        missing_hashes = [
            prompt_hash
            for prompt_hash in prompt_hashes
            if prompt_hash not in self.reference_embeddings
        ]
        if missing_hashes:
            raise KeyError(
                f"Missing reference embeddings for {len(missing_hashes)} prompt hash(es), "
                f"first missing hash: {missing_hashes[0]}"
            )

        generated_embeddings = self.encoder.encode_images(images)
        reference_embeddings = torch.stack(
            [self.reference_embeddings[prompt_hash] for prompt_hash in prompt_hashes]
        )
        rewards = (generated_embeddings * reference_embeddings).sum(dim=-1)
        return rewards.float().tolist()

    def close(self) -> None:
        """Release CUDA memory held by this worker."""
        self.encoder.offload_to_cpu()


class WDEVA02PromptHashService:
    """Compute generated-reference similarity rewards with prompt-hash lookup."""

    def __init__(
        self,
        model_path: Path,
        cache_path: Path,
        device: str,
        dtype: str,
        disable_ddp: bool,
    ) -> None:
        self.model_path = model_path
        self.cache_path = cache_path
        self.device = self._resolve_device(device, disable_ddp)
        self.dtype = getattr(torch, dtype) if self.device.type == "cuda" else torch.float32
        self.dtype_name = dtype if self.device.type == "cuda" else "float32"
        self.device_ids = self._resolve_ddp_device_ids(disable_ddp)
        self.ddp_enabled = len(self.device_ids) > 1
        self.ddp_pool: Optional[DDPWorkerPool] = None
        self._condition = threading.Condition(threading.Lock())
        self._active_compute_count = 0
        self._loaded_on_device = False

        self.reference_embeddings = load_reference_cache(cache_path)
        if self.ddp_enabled:
            self.encoder: Optional[WDEVA02EmbeddingModel] = None
            self.ddp_pool = DDPWorkerPool(
                worker_factory=WDEVA02PromptHashDDPWorker,
                worker_kwargs={
                    "model_path": model_path,
                    "cache_path": cache_path,
                    "dtype": self.dtype_name,
                },
                world_size=len(self.device_ids),
                batch_keys=("prompt", "image", "video"),
                logger=LOGGER,
            )
        else:
            self.encoder = WDEVA02EmbeddingModel(
                model_path=model_path,
                device=str(self.device),
                dtype=self.dtype_name,
            )

        LOGGER.info(
            "Loaded %s WD reference embeddings from %s",
            len(self.reference_embeddings),
            cache_path,
        )
        LOGGER.info(
            "WD prompt-hash device config: device=%s dtype=%s visible_cuda_devices=%s "
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
                "DDP WD inference uses all visible CUDA devices and requires "
                "--device cuda or --device cuda:0. Use --disable-ddp for a "
                "single indexed CUDA device."
            )

        return torch.device("cuda:0" if resolved.index is None else resolved)

    def _resolve_ddp_device_ids(self, disable_ddp: bool) -> list[int]:
        """Return CUDA device ids used by DDP workers."""
        if self.device.type != "cuda" or disable_ddp:
            return []

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            return []
        return list(range(device_count))

    def ensure_on_device(self) -> None:
        """Move the WD model to its inference device."""
        with self._condition:
            self._ensure_on_device_locked()

    def _ensure_on_device_locked(self) -> None:
        """Move the WD model to its inference device while holding the condition."""
        if self._loaded_on_device:
            return
        if self.ddp_enabled:
            if self.ddp_pool is None:
                raise RuntimeError("DDP worker pool is not initialized.")
            self.ddp_pool.start()
            self._loaded_on_device = True
            return
        if self.encoder is None:
            raise RuntimeError("WD encoder is not initialized.")
        self.encoder.ensure_on_device()
        self._loaded_on_device = True

    def offload_to_cpu(self) -> None:
        """Move the WD model to CPU after in-flight compute requests finish."""
        with self._condition:
            while self._active_compute_count > 0:
                self._condition.wait()
            self._offload_to_cpu_locked()

    def _offload_to_cpu_locked(self) -> None:
        """Move the WD model to CPU while holding the condition."""
        if not self._loaded_on_device:
            return
        if self.ddp_enabled:
            if self.ddp_pool is not None:
                self.ddp_pool.shutdown()
            self._loaded_on_device = False
            return
        if self.encoder is None:
            raise RuntimeError("WD encoder is not initialized.")
        self.encoder.offload_to_cpu()
        self._loaded_on_device = False

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

    def compute_rewards(
        self,
        prompts: list[str],
        image_payload: Optional[list[str]],
        video_payload: Optional[list[list[str]]],
    ) -> list[float]:
        """Compute cosine similarity rewards for generated images."""
        start_time = time.perf_counter()
        sample_count = _count_media_payload(
            image_payload=image_payload,
            video_payload=video_payload,
        )
        if sample_count <= 0:
            raise ValueError("At least one image or video input is required.")
        if len(prompts) != sample_count:
            raise ValueError(
                f"Prompt/image length mismatch: prompts={len(prompts)} images={sample_count}"
            )

        prompt_hashes = [prompt_sha256(prompt) for prompt in prompts]
        missing_hashes = [
            prompt_hash
            for prompt_hash in prompt_hashes
            if prompt_hash not in self.reference_embeddings
        ]
        if missing_hashes:
            raise KeyError(
                f"Missing reference embeddings for {len(missing_hashes)} prompt hash(es), "
                f"first missing hash: {missing_hashes[0]}"
            )

        LOGGER.info(
            "WD compute request: samples=%s ddp=%s active_gpus=%s",
            sample_count,
            self.ddp_enabled,
            len(self.device_ids) if self.ddp_enabled else 1,
        )
        self._enter_compute()
        try:
            if self.ddp_enabled:
                if self.ddp_pool is None:
                    raise RuntimeError("DDP worker pool is not initialized.")
                rewards = self.ddp_pool.compute(
                    payload={
                        "prompt": prompts,
                        "image": image_payload,
                        "video": video_payload,
                    },
                    total_size=sample_count,
                )
            else:
                if self.encoder is None:
                    raise RuntimeError("WD encoder is not initialized.")
                images = _resolve_media_payload(
                    image_payload=image_payload,
                    video_payload=video_payload,
                )
                generated_embeddings = self.encoder.encode_images(images)

                reference_embeddings = torch.stack(
                    [self.reference_embeddings[prompt_hash] for prompt_hash in prompt_hashes]
                )
                rewards_tensor = (generated_embeddings * reference_embeddings).sum(dim=-1)
                rewards = rewards_tensor.float().tolist()
        finally:
            self._exit_compute()

        LOGGER.info(
            "WD compute complete: samples=%s ddp=%s active_gpus=%s elapsed_s=%.3f",
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
    """HTTP request handler bound to the WD prompt-hash reward service."""

    service: WDEVA02PromptHashService

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
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
                    "reference_count": len(self.service.reference_embeddings),
                    "cache_path": str(self.service.cache_path),
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
                LOGGER.exception("Failed to load WD prompt-hash model.")
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if self.path == "/offload":
            try:
                self.service.offload_to_cpu()
                self._send_json({"status": "offloaded"})
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to offload WD prompt-hash model.")
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
                prompts=payload.get("prompt") or [],
                image_payload=payload.get("image"),
                video_payload=payload.get("video"),
            )
            self._send_json({"rewards": rewards})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to compute WD prompt-hash rewards.")
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args) -> None:
        """Redirect HTTP logs through the standard logger."""
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    """Start the reward server."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    RewardRequestHandler.service = WDEVA02PromptHashService(
        model_path=args.model_path,
        cache_path=args.cache_path,
        device=args.device,
        dtype=args.dtype,
        disable_ddp=args.disable_ddp,
    )

    server = ThreadingHTTPServer((args.host, args.port), RewardRequestHandler)
    LOGGER.info(
        "Starting WD prompt-hash reward server on http://%s:%s using %s",
        args.host,
        args.port,
        args.model_path,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping WD prompt-hash reward server.")
    finally:
        RewardRequestHandler.service.close()
        server.server_close()


if __name__ == "__main__":
    main()
