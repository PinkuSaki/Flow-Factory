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

"""Serve prompt-hash reference similarity with hierarchical Haar wavelet energy.

The server exposes:
    - ``GET /health``  -> ``{"status": "ok", "reference_count": ...}``
    - ``POST /load``   -> lifecycle-compatible no-op
    - ``POST /offload`` -> lifecycle-compatible no-op
    - ``POST /compute`` -> ``{"rewards": [...]}``

Reference descriptors must be built before training with
``build_wavelet_prompt_hash_cache.py``. This training-time server only loads the
cache, hashes incoming prompts, extracts generated-image wavelet descriptors,
and returns similarity rewards.
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
from wavelet_prompt_hash_common import (
    WaveletDescriptorConfig,
    WaveletImageOutputs,
    extract_wavelet_outputs,
    load_reference_cache_payload,
    prompt_sha256,
    wavelet_similarity_reward,
)

LOGGER = logging.getLogger("wavelet_prompt_hash_server")


def _ignore_sigint_in_worker() -> None:
    """Let the parent process handle Ctrl-C and shut worker processes down cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=18084, help="Port to bind.")
    parser.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Prebuilt prompt-hash reference wavelet cache.",
    )
    parser.add_argument(
        "--score-type",
        choices=("bhattacharyya", "soft_jaccard", "cosine_log_energy"),
        default="bhattacharyya",
        help="Wavelet descriptor similarity score to compute.",
    )
    parser.add_argument(
        "--distribution-weight",
        type=float,
        default=0.8,
        help=(
            "Weight for the descriptor distribution score. The remaining weight is applied "
            "to total wavelet energy similarity."
        ),
    )
    parser.add_argument(
        "--total-energy-tau",
        type=float,
        default=1.0,
        help="Temperature for total wavelet energy log-ratio similarity.",
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


def _extract_wavelet_output_shard(
    image_payload: Optional[list[str]],
    video_payload: Optional[list[list[str]]],
    descriptor_config: WaveletDescriptorConfig,
) -> WaveletImageOutputs:
    """Decode and extract wavelet descriptors for one process worker shard."""
    images = _resolve_media_payload(
        image_payload=image_payload,
        video_payload=video_payload,
    )
    return extract_wavelet_outputs(images, descriptor_config)


def _concat_wavelet_outputs(outputs: list[WaveletImageOutputs]) -> WaveletImageOutputs:
    """Concatenate wavelet output shards in request order."""
    if not outputs:
        raise ValueError("At least one wavelet output shard is required.")
    return WaveletImageOutputs(
        distributions=torch.cat([output.distributions for output in outputs], dim=0),
        log_energies=torch.cat([output.log_energies for output in outputs], dim=0),
        total_energies=torch.cat([output.total_energies for output in outputs], dim=0),
    )


class WaveletPromptHashService:
    """Compute generated-reference similarity rewards with prompt-hash lookup."""

    def __init__(
        self,
        cache_path: Path,
        score_type: str,
        distribution_weight: float,
        total_energy_tau: float,
        num_workers: int,
    ) -> None:
        self.cache_path = cache_path
        self.score_type = score_type
        self.distribution_weight = distribution_weight
        self.total_energy_tau = total_energy_tau
        self.num_workers = num_workers
        self.reference_cache = load_reference_cache_payload(cache_path)
        self.descriptor_config = WaveletDescriptorConfig.from_metadata(
            self.reference_cache.metadata
        )
        if self.num_workers <= 0:
            raise ValueError(f"num_workers must be positive, got {self.num_workers}.")
        self.process_pool: Optional[ProcessPoolExecutor] = None
        if self.num_workers > 1:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_ignore_sigint_in_worker,
            )

        self._condition = threading.Condition(threading.Lock())
        self._active_compute_count = 0
        self._loaded = False

        if not 0.0 <= self.distribution_weight <= 1.0:
            raise ValueError(
                f"distribution_weight must be in [0, 1], got {self.distribution_weight}."
            )
        if self.total_energy_tau <= 0:
            raise ValueError(f"total_energy_tau must be positive, got {self.total_energy_tau}.")

        LOGGER.info(
            "Loaded %s wavelet reference entries from %s",
            len(self.reference_cache.distributions),
            cache_path,
        )
        LOGGER.info(
            "Wavelet prompt-hash config: image_size=%s levels=%s color_space=%s "
            "include_approximation=%s level_weight_decay=%s score_type=%s "
            "distribution_weight=%s total_energy_tau=%s num_workers=%s",
            self.descriptor_config.image_size,
            self.descriptor_config.levels,
            self.descriptor_config.color_space,
            self.descriptor_config.include_approximation,
            self.descriptor_config.level_weight_decay,
            self.score_type,
            self.distribution_weight,
            self.total_energy_tau,
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

    def _build_reference_outputs(self, prompt_hashes: list[str]) -> WaveletImageOutputs:
        """Stack cached reference descriptors for one compute request."""
        return WaveletImageOutputs(
            distributions=torch.stack(
                [self.reference_cache.distributions[prompt_hash] for prompt_hash in prompt_hashes]
            ),
            log_energies=torch.stack(
                [self.reference_cache.log_energies[prompt_hash] for prompt_hash in prompt_hashes]
            ),
            total_energies=torch.stack(
                [self.reference_cache.total_energies[prompt_hash] for prompt_hash in prompt_hashes]
            ),
        )

    def _extract_generated_outputs(
        self,
        image_payload: Optional[list[str]],
        video_payload: Optional[list[list[str]]],
        sample_count: int,
    ) -> WaveletImageOutputs:
        """Extract generated descriptors with optional process-pool parallelism."""
        if self.process_pool is None or sample_count <= 1:
            images = _resolve_media_payload(
                image_payload=image_payload,
                video_payload=video_payload,
            )
            return extract_wavelet_outputs(images, self.descriptor_config)

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
                    _extract_wavelet_output_shard,
                    shard_image_payload,
                    shard_video_payload,
                    self.descriptor_config,
                )
            )

        return _concat_wavelet_outputs([future.result() for future in futures])

    def compute_rewards(
        self,
        prompts: list[str],
        image_payload: Optional[list[str]],
        video_payload: Optional[list[list[str]]],
    ) -> list[float]:
        """Compute wavelet similarity rewards for generated images."""
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
            if prompt_hash not in self.reference_cache.distributions
        ]
        if missing_hashes:
            raise KeyError(
                f"Missing reference data for {len(missing_hashes)} prompt hash(es), "
                f"first missing hash: {missing_hashes[0]}"
            )

        LOGGER.info(
            "Wavelet compute request: samples=%s score_type=%s num_workers=%s",
            sample_count,
            self.score_type,
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
            rewards = wavelet_similarity_reward(
                generated_outputs=generated_outputs,
                reference_outputs=reference_outputs,
                score_type=self.score_type,
                distribution_weight=self.distribution_weight,
                total_energy_tau=self.total_energy_tau,
            )
            reward_values = rewards.float().tolist()
        finally:
            self._exit_compute()

        LOGGER.info(
            "Wavelet compute complete: samples=%s score_type=%s num_workers=%s elapsed_s=%.3f",
            sample_count,
            self.score_type,
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
    """HTTP request handler bound to the wavelet prompt-hash reward service."""

    service: WaveletPromptHashService

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
            descriptor_dim = next(iter(self.service.reference_cache.distributions.values())).numel()
            self._send_json(
                {
                    "status": "ok",
                    "reference_count": len(self.service.reference_cache.distributions),
                    "cache_path": str(self.service.cache_path),
                    "score_type": self.service.score_type,
                    "distribution_weight": self.service.distribution_weight,
                    "total_energy_tau": self.service.total_energy_tau,
                    "num_workers": self.service.num_workers,
                    "process_pool": self.service.process_pool is not None,
                    "wavelet": "haar",
                    "image_size": self.service.descriptor_config.image_size,
                    "levels": self.service.descriptor_config.levels,
                    "color_space": self.service.descriptor_config.color_space,
                    "include_approximation": (self.service.descriptor_config.include_approximation),
                    "level_weight_decay": self.service.descriptor_config.level_weight_decay,
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
                LOGGER.exception("Failed to load wavelet prompt-hash service.")
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if self.path == "/offload":
            try:
                self.service.offload_to_cpu()
                self._send_json({"status": "offloaded"})
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Failed to offload wavelet prompt-hash service.")
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
            LOGGER.exception("Failed to compute wavelet prompt-hash rewards.")
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args) -> None:
        """Redirect HTTP logs through the standard logger."""
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    """Start the reward server."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    RewardRequestHandler.service = WaveletPromptHashService(
        cache_path=args.cache_path,
        score_type=args.score_type,
        distribution_weight=args.distribution_weight,
        total_energy_tau=args.total_energy_tau,
        num_workers=args.num_workers,
    )

    server = ThreadingHTTPServer((args.host, args.port), RewardRequestHandler)
    LOGGER.info(
        "Starting wavelet prompt-hash reward server on http://%s:%s "
        "score_type=%s num_workers=%s",
        args.host,
        args.port,
        args.score_type,
        args.num_workers,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping wavelet prompt-hash reward server.")
    finally:
        RewardRequestHandler.service.close()
        server.server_close()


if __name__ == "__main__":
    main()
