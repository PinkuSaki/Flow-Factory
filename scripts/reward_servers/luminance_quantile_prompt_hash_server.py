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

"""Serve prompt-hash reference similarity with luminance quantile curves.

The server exposes:
    - ``GET /health``  -> ``{"status": "ok", "reference_count": ...}``
    - ``POST /load``   -> lifecycle-compatible no-op
    - ``POST /offload`` -> lifecycle-compatible no-op
    - ``POST /compute`` -> ``{"rewards": [...]}``

Reference descriptors must be built before training with
``build_luminance_quantile_prompt_hash_cache.py``. This training-time server
only loads the cache, hashes incoming prompts, extracts generated-image
luminance descriptors, and returns similarity rewards.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import signal
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from luminance_quantile_prompt_hash_common import (
    LuminanceQuantileConfig,
    LuminanceQuantileOutputs,
    extract_luminance_quantile_outputs,
    load_reference_cache_payload,
    luminance_quantile_reward,
    prompt_sha256,
)

LOGGER = logging.getLogger("luminance_quantile_prompt_hash_server")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=18086, help="Port to bind.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Prebuilt prompt-hash reference luminance quantile cache.",
    )
    parser.add_argument(
        "--curve-weight",
        type=float,
        default=0.75,
        help="Weight for the luminance quantile curve similarity term.",
    )
    parser.add_argument(
        "--contrast-weight",
        type=float,
        default=0.15,
        help="Weight for the P95-P05 contrast similarity term.",
    )
    parser.add_argument(
        "--clip-weight",
        type=float,
        default=0.10,
        help="Weight for the black/white clipping fraction similarity term.",
    )
    parser.add_argument(
        "--reward-temperature",
        "--curve-temperature",
        dest="reward_temperature",
        type=float,
        default=0.08,
        help="Temperature for converting quantile-curve L1 distance into exp(-d/tau).",
    )
    parser.add_argument(
        "--contrast-temperature",
        type=float,
        default=0.35,
        help="Temperature for converting contrast log-ratio distance into exp(-d/tau).",
    )
    parser.add_argument(
        "--clip-temperature",
        type=float,
        default=0.05,
        help="Temperature for converting clipping fraction distance into exp(-d/tau).",
    )
    parser.add_argument(
        "--tail-weight",
        type=float,
        default=0.0,
        help=(
            "Additional endpoint emphasis for quantile distance. 0 disables explicit "
            "tail weighting; the default quantile grid is already denser near tails."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of CPU worker processes used inside each compute request.",
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


def _build_non_empty_spans(total_size: int, worker_count: int) -> list[tuple[int, int]]:
    """Split a batch into non-empty contiguous spans."""
    if total_size <= 0:
        return []
    if worker_count <= 0:
        raise ValueError(f"worker_count must be positive, got {worker_count}.")

    shard_count = min(total_size, worker_count)
    base_size = total_size // shard_count
    remainder = total_size % shard_count
    spans = []
    start = 0
    for shard_index in range(shard_count):
        end = start + base_size + (1 if shard_index < remainder else 0)
        spans.append((start, end))
        start = end
    return spans


def _ignore_sigint_in_worker() -> None:
    """Let the parent process handle Ctrl-C and shut worker processes down cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _slice_media_payload(
    image_payload: Optional[list[str]],
    video_payload: Optional[list[list[str]]],
    start: int,
    end: int,
) -> tuple[Optional[list[str]], Optional[list[list[str]]]]:
    """Slice one media payload while preserving image-or-video exclusivity."""
    if image_payload:
        return image_payload[start:end], None
    if video_payload:
        return None, video_payload[start:end]
    return None, None


def _extract_luminance_output_shard(
    image_payload: Optional[list[str]],
    video_payload: Optional[list[list[str]]],
    descriptor_config: LuminanceQuantileConfig,
) -> LuminanceQuantileOutputs:
    """Decode and extract luminance descriptors for one process worker shard."""
    images = _resolve_media_payload(
        image_payload=image_payload,
        video_payload=video_payload,
    )
    return extract_luminance_quantile_outputs(images, descriptor_config)


def _concat_luminance_outputs(
    outputs: list[LuminanceQuantileOutputs],
) -> LuminanceQuantileOutputs:
    """Concatenate luminance output shards in request order."""
    if not outputs:
        raise ValueError("At least one luminance output shard is required.")
    return LuminanceQuantileOutputs(
        quantiles=torch.cat([output.quantiles for output in outputs], dim=0),
        contrasts=torch.cat([output.contrasts for output in outputs], dim=0),
        black_fractions=torch.cat([output.black_fractions for output in outputs], dim=0),
        white_fractions=torch.cat([output.white_fractions for output in outputs], dim=0),
    )


class LuminanceQuantilePromptHashService:
    """Compute generated-reference luminance rewards with prompt-hash lookup."""

    def __init__(
        self,
        cache_path: Path,
        curve_weight: float,
        contrast_weight: float,
        clip_weight: float,
        reward_temperature: float,
        contrast_temperature: float,
        clip_temperature: float,
        tail_weight: float,
        num_workers: int,
    ) -> None:
        self.cache_path = cache_path
        self.curve_weight = curve_weight
        self.contrast_weight = contrast_weight
        self.clip_weight = clip_weight
        self.reward_temperature = reward_temperature
        self.contrast_temperature = contrast_temperature
        self.clip_temperature = clip_temperature
        self.tail_weight = tail_weight
        self.num_workers = num_workers
        self.reference_cache = load_reference_cache_payload(cache_path)
        self.descriptor_config = LuminanceQuantileConfig.from_metadata(
            self.reference_cache.metadata
        )
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}.")
        if any(
            weight < 0.0 for weight in (self.curve_weight, self.contrast_weight, self.clip_weight)
        ):
            raise ValueError(
                "Reward weights must be non-negative, got "
                f"curve={curve_weight}, contrast={contrast_weight}, clip={clip_weight}."
            )
        if self.curve_weight + self.contrast_weight + self.clip_weight <= 0.0:
            raise ValueError("At least one reward weight must be positive.")
        if self.reward_temperature <= 0.0:
            raise ValueError(
                f"reward_temperature must be positive, got {self.reward_temperature}."
            )
        if self.contrast_temperature <= 0.0:
            raise ValueError(
                f"contrast_temperature must be positive, got {self.contrast_temperature}."
            )
        if self.clip_temperature <= 0.0:
            raise ValueError(f"clip_temperature must be positive, got {self.clip_temperature}.")
        if self.tail_weight < 0.0:
            raise ValueError(f"tail_weight must be non-negative, got {self.tail_weight}.")

        self.process_pool: Optional[ProcessPoolExecutor] = None
        if self.num_workers > 1:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_ignore_sigint_in_worker,
            )

        self._condition = threading.Condition(threading.Lock())
        self._active_compute_count = 0
        self._loaded = False

        LOGGER.info(
            "Loaded %s luminance quantile reference entries from %s",
            len(self.reference_cache.quantiles),
            cache_path,
        )
        LOGGER.info(
            "Luminance quantile prompt-hash config: image_size=%s luminance_space=%s "
            "quantiles=%s black_threshold=%s white_threshold=%s "
            "weights=(%s,%s,%s) temperatures=(%s,%s,%s) tail_weight=%s num_workers=%s",
            self.descriptor_config.image_size,
            self.descriptor_config.luminance_space,
            len(self.descriptor_config.quantile_levels),
            self.descriptor_config.black_threshold,
            self.descriptor_config.white_threshold,
            self.curve_weight,
            self.contrast_weight,
            self.clip_weight,
            self.reward_temperature,
            self.contrast_temperature,
            self.clip_temperature,
            self.tail_weight,
            self.num_workers,
        )

    def ensure_on_device(self) -> None:
        """Mark the model lifecycle as loaded for remote reward compatibility."""
        with self._condition:
            self._loaded = True

    def offload_to_cpu(self) -> None:
        """Wait for in-flight requests and mark the lifecycle as offloaded."""
        with self._condition:
            while self._active_compute_count > 0:
                self._condition.wait()
            self._loaded = False

    def _enter_compute(self) -> None:
        """Mark a compute request active."""
        with self._condition:
            self._active_compute_count += 1

    def _exit_compute(self) -> None:
        """Mark a compute request complete."""
        with self._condition:
            self._active_compute_count -= 1
            self._condition.notify_all()

    def _build_reference_outputs(self, prompt_hashes: list[str]) -> LuminanceQuantileOutputs:
        """Stack cached reference descriptors for one compute request."""
        return LuminanceQuantileOutputs(
            quantiles=torch.stack(
                [self.reference_cache.quantiles[prompt_hash] for prompt_hash in prompt_hashes]
            ),
            contrasts=torch.stack(
                [self.reference_cache.contrasts[prompt_hash] for prompt_hash in prompt_hashes]
            ),
            black_fractions=torch.stack(
                [
                    self.reference_cache.black_fractions[prompt_hash]
                    for prompt_hash in prompt_hashes
                ]
            ),
            white_fractions=torch.stack(
                [
                    self.reference_cache.white_fractions[prompt_hash]
                    for prompt_hash in prompt_hashes
                ]
            ),
        )

    def _extract_generated_outputs(
        self,
        image_payload: Optional[list[str]],
        video_payload: Optional[list[list[str]]],
        sample_count: int,
    ) -> LuminanceQuantileOutputs:
        """Extract generated descriptors with optional process-pool parallelism."""
        if self.process_pool is None or sample_count <= 1:
            images = _resolve_media_payload(
                image_payload=image_payload,
                video_payload=video_payload,
            )
            return extract_luminance_quantile_outputs(images, self.descriptor_config)

        futures = []
        spans = _build_non_empty_spans(sample_count, self.num_workers)
        for start, end in spans:
            shard_image_payload, shard_video_payload = _slice_media_payload(
                image_payload=image_payload,
                video_payload=video_payload,
                start=start,
                end=end,
            )
            futures.append(
                self.process_pool.submit(
                    _extract_luminance_output_shard,
                    shard_image_payload,
                    shard_video_payload,
                    self.descriptor_config,
                )
            )

        return _concat_luminance_outputs([future.result() for future in futures])

    def compute_rewards(
        self,
        prompts: list[str],
        image_payload: Optional[list[str]],
        video_payload: Optional[list[list[str]]],
    ) -> list[float]:
        """Compute luminance quantile similarity rewards for generated images."""
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
            if prompt_hash not in self.reference_cache.quantiles
        ]
        if missing_hashes:
            raise KeyError(
                f"Missing reference data for {len(missing_hashes)} prompt hash(es), "
                f"first missing hash: {missing_hashes[0]}"
            )

        LOGGER.info(
            "Luminance quantile compute request: samples=%s num_workers=%s",
            sample_count,
            self.num_workers,
        )
        self._enter_compute()
        try:
            generated_outputs = self._extract_generated_outputs(
                image_payload=image_payload,
                video_payload=video_payload,
                sample_count=sample_count,
            )
            reference_outputs = self._build_reference_outputs(prompt_hashes)
            rewards = luminance_quantile_reward(
                generated_outputs=generated_outputs,
                reference_outputs=reference_outputs,
                quantile_levels=self.descriptor_config.quantile_levels,
                curve_weight=self.curve_weight,
                contrast_weight=self.contrast_weight,
                clip_weight=self.clip_weight,
                reward_temperature=self.reward_temperature,
                contrast_temperature=self.contrast_temperature,
                clip_temperature=self.clip_temperature,
                tail_weight=self.tail_weight,
            )
            reward_values = rewards.float().tolist()
        finally:
            self._exit_compute()

        LOGGER.info(
            "Luminance quantile compute complete: samples=%s num_workers=%s elapsed_s=%.3f",
            sample_count,
            self.num_workers,
            time.perf_counter() - start_time,
        )
        return reward_values

    def close(self) -> None:
        """Release server resources."""
        self.offload_to_cpu()
        if self.process_pool is not None:
            self.process_pool.shutdown(wait=True, cancel_futures=True)
            self.process_pool = None


class RewardRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler bound to the luminance quantile reward service."""

    service: LuminanceQuantilePromptHashService

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
            descriptor_dim = next(iter(self.service.reference_cache.quantiles.values())).numel()
            self._send_json(
                {
                    "status": "ok",
                    "reference_count": len(self.service.reference_cache.quantiles),
                    "cache_path": str(self.service.cache_path),
                    "curve_weight": self.service.curve_weight,
                    "contrast_weight": self.service.contrast_weight,
                    "clip_weight": self.service.clip_weight,
                    "reward_temperature": self.service.reward_temperature,
                    "contrast_temperature": self.service.contrast_temperature,
                    "clip_temperature": self.service.clip_temperature,
                    "tail_weight": self.service.tail_weight,
                    "num_workers": self.service.num_workers,
                    "process_pool": self.service.process_pool is not None,
                    "descriptor": "luminance_quantile_curve",
                    "image_size": self.service.descriptor_config.image_size,
                    "luminance_space": self.service.descriptor_config.luminance_space,
                    "black_threshold": self.service.descriptor_config.black_threshold,
                    "white_threshold": self.service.descriptor_config.white_threshold,
                    "descriptor_dim": descriptor_dim,
                    "loaded": self.service._loaded,
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
                LOGGER.exception("Failed to load luminance quantile prompt-hash service.")
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if self.path == "/offload":
            try:
                self.service.offload_to_cpu()
                self._send_json({"status": "offloaded"})
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to offload luminance quantile prompt-hash service.")
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
            LOGGER.exception("Failed to compute luminance quantile prompt-hash rewards.")
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args) -> None:
        """Redirect HTTP logs through the standard logger."""
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    """Start the reward server."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    RewardRequestHandler.service = LuminanceQuantilePromptHashService(
        cache_path=args.cache_path,
        curve_weight=args.curve_weight,
        contrast_weight=args.contrast_weight,
        clip_weight=args.clip_weight,
        reward_temperature=args.reward_temperature,
        contrast_temperature=args.contrast_temperature,
        clip_temperature=args.clip_temperature,
        tail_weight=args.tail_weight,
        num_workers=args.num_workers,
    )

    server = ThreadingHTTPServer((args.host, args.port), RewardRequestHandler)
    LOGGER.info(
        "Starting luminance quantile prompt-hash reward server on http://%s:%s "
        "num_workers=%s",
        args.host,
        args.port,
        args.num_workers,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping luminance quantile prompt-hash reward server.")
    finally:
        RewardRequestHandler.service.close()
        server.server_close()


if __name__ == "__main__":
    main()
