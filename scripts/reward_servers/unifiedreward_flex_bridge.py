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

"""Bridge UnifiedReward-Flex pairwise judging into a groupwise reward service.

The bridge exposes:
    - ``GET /health``  -> ``{"status": "ok"}``
    - ``POST /compute`` -> ``{"rewards": [...]}``

Each ``/compute`` request compares every pair in a group and converts pairwise
wins into win-rate rewards in ``[0, 1]``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import re
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from openai import OpenAI


LOGGER = logging.getLogger("unifiedreward_flex_bridge")

PAIRWISE_PROMPT_TEMPLATE = """You are an expert judge for anime image generation.

Compare Image 1 and Image 2 for the prompt below using:
1. Prompt alignment
2. Visual quality and artifact control
3. Overall aesthetics and composition

Prompt:
{prompt}

Return exactly one JSON object with no extra text:
{{"winner": 1, "reason": "short explanation"}}

Rules:
- winner must be 1, 2, or 0
- use 0 only for an actual tie
- keep reason under 20 words
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=18083, help="Port to bind.")
    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8080/v1",
        help="Base URL of the local OpenAI-compatible vLLM endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the local OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--model",
        default="UnifiedReward",
        help="Served model name exposed by vLLM.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for one pairwise comparison.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Maximum completion tokens for one pairwise comparison.",
    )
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=600,
        help="Maximum prompt characters forwarded to the judge model.",
    )
    parser.add_argument(
        "--tie-value",
        type=float,
        default=0.5,
        help="Reward assigned to each image when the judge returns a tie.",
    )
    parser.add_argument(
        "--max-parallel-pairs",
        type=int,
        default=1,
        help="Maximum number of pairwise requests to submit concurrently per group.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def _ensure_data_url(image_payload: str) -> str:
    """Ensure an image payload is a valid data URL."""
    if image_payload.startswith("data:image/"):
        return image_payload
    return f"data:image/png;base64,{image_payload}"


def _normalize_prompt(prompt: str, max_chars: int) -> str:
    """Normalize and truncate prompts before forwarding them to the judge."""
    normalized = re.sub(r"\s+", " ", prompt).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 16].rstrip() + " ... [truncated]"


def _resolve_group_images(
    image_payload: Optional[list[str]],
    video_payload: Optional[list[list[str]]],
) -> list[str]:
    """Resolve image or video payloads into image data URLs."""
    if image_payload:
        return [_ensure_data_url(item) for item in image_payload]

    if not video_payload:
        return []

    resolved: list[str] = []
    for frames in video_payload:
        if not frames:
            raise ValueError("Encountered an empty video payload.")
        resolved.append(_ensure_data_url(frames[0]))
    return resolved


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a text response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not find a JSON object in response: {text!r}") from None
        return json.loads(match.group(0))


def _extract_winner_fallback(text: str) -> Optional[int]:
    """Extract the winner from a partially truncated response."""
    match = re.search(
        r'"winner"\s*:\s*(?:"([^"]+)"|([0-2]))',
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    value = (match.group(1) or match.group(2) or "").strip().lower()
    if value in {"0", "tie", "draw"}:
        return 0
    if value in {"1", "image 1", "image1"}:
        return 1
    if value in {"2", "image 2", "image2"}:
        return 2
    return None


def _parse_winner(response_text: str) -> int:
    """Parse the pairwise winner from the model response."""
    try:
        payload = _extract_json_object(response_text)
    except ValueError:
        fallback_winner = _extract_winner_fallback(response_text)
        if fallback_winner is not None:
            return fallback_winner
        raise

    winner = payload.get("winner")
    winner_name = payload.get("winner_name")

    if isinstance(winner, int) and winner in {0, 1, 2}:
        return winner

    if isinstance(winner, str):
        normalized = winner.strip().lower()
        if normalized in {"0", "tie", "draw"}:
            return 0
        if normalized in {"1", "image 1", "image1"} or "image 1" in normalized:
            return 1
        if normalized in {"2", "image 2", "image2"} or "image 2" in normalized:
            return 2

    if isinstance(winner_name, str):
        normalized = winner_name.strip().lower()
        if normalized in {"0", "tie", "draw"}:
            return 0
        if normalized in {"1", "image 1", "image1"} or "image 1" in normalized:
            return 1
        if normalized in {"2", "image 2", "image2"} or "image 2" in normalized:
            return 2

    raise ValueError(f"Unsupported winner field in response: {payload!r}")


class UnifiedRewardFlexBridgeService:
    """Groupwise reward service backed by pairwise UnifiedReward-Flex calls."""

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        model: str,
        request_timeout: float,
        max_retries: int,
        max_completion_tokens: int,
        max_prompt_chars: int,
        tie_value: float,
        max_parallel_pairs: int,
    ) -> None:
        http_client = None
        if self._should_bypass_env_proxy(api_base_url):
            # Local vLLM endpoints must bypass global proxy variables.
            http_client = httpx.Client(trust_env=False, timeout=request_timeout)

        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
            timeout=request_timeout,
            http_client=http_client,
        )
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.max_completion_tokens = max_completion_tokens
        self.max_prompt_chars = max_prompt_chars
        self.tie_value = tie_value
        self.max_parallel_pairs = max(1, max_parallel_pairs)

    @staticmethod
    def _should_bypass_env_proxy(api_base_url: str) -> bool:
        """Return True when the target URL points to a loopback endpoint."""
        hostname = (urlparse(api_base_url).hostname or "").lower()
        return hostname in {"127.0.0.1", "localhost", "::1"}

    def compare_pair(self, prompt: str, left_image: str, right_image: str) -> int:
        """Compare one image pair and return the winner index."""
        prompt_text = PAIRWISE_PROMPT_TEMPLATE.format(
            prompt=_normalize_prompt(prompt, self.max_prompt_chars)
        )
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": left_image}},
                                {"type": "image_url", "image_url": {"url": right_image}},
                            ],
                        }
                    ],
                    temperature=0.0,
                    max_completion_tokens=self.max_completion_tokens,
                )
                content = completion.choices[0].message.content or ""
                return _parse_winner(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                LOGGER.warning(
                    "UnifiedReward-Flex pairwise request failed on attempt %s/%s: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))

        raise RuntimeError(f"Failed to score pair after {self.max_retries} attempts: {last_error}")

    def compute_rewards(
        self,
        prompts: list[str],
        image_payload: Optional[list[str]],
        video_payload: Optional[list[list[str]]],
    ) -> list[float]:
        """Convert pairwise comparison results into win-rate rewards."""
        images = _resolve_group_images(image_payload=image_payload, video_payload=video_payload)
        if not images:
            raise ValueError("At least one image or video input is required.")
        if len(prompts) != len(images):
            raise ValueError(
                f"Prompt/image length mismatch: prompts={len(prompts)} images={len(images)}"
            )
        if len(images) == 1:
            return [0.0]

        canonical_prompt = prompts[0]
        wins = [0.0 for _ in images]

        pair_indices = list(combinations(range(len(images)), 2))
        pair_results: list[tuple[int, int, int]] = []

        if self.max_parallel_pairs == 1 or len(pair_indices) == 1:
            for left_index, right_index in pair_indices:
                winner = self.compare_pair(
                    prompt=canonical_prompt,
                    left_image=images[left_index],
                    right_image=images[right_index],
                )
                pair_results.append((left_index, right_index, winner))
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_parallel_pairs, len(pair_indices))) as executor:
                future_map = {
                    executor.submit(
                        self.compare_pair,
                        canonical_prompt,
                        images[left_index],
                        images[right_index],
                    ): (left_index, right_index)
                    for left_index, right_index in pair_indices
                }
                for future in as_completed(future_map):
                    left_index, right_index = future_map[future]
                    pair_results.append((left_index, right_index, future.result()))

        for left_index, right_index, winner in pair_results:
            if winner == 1:
                wins[left_index] += 1.0
            elif winner == 2:
                wins[right_index] += 1.0
            else:
                wins[left_index] += self.tie_value
                wins[right_index] += self.tie_value

        max_wins = float(len(images) - 1)
        return [win_count / max_wins for win_count in wins]


class RewardRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler bound to the bridge service."""

    service: UnifiedRewardFlexBridgeService

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
                prompts=payload.get("prompt") or [],
                image_payload=payload.get("image"),
                video_payload=payload.get("video"),
            )
            self._send_json({"rewards": rewards})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to compute UnifiedReward-Flex rewards.")
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Redirect HTTP logs through the standard logger."""
        LOGGER.info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    """Start the bridge server."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    RewardRequestHandler.service = UnifiedRewardFlexBridgeService(
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        model=args.model,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        max_completion_tokens=args.max_completion_tokens,
        max_prompt_chars=args.max_prompt_chars,
        tie_value=args.tie_value,
        max_parallel_pairs=args.max_parallel_pairs,
    )

    server = ThreadingHTTPServer((args.host, args.port), RewardRequestHandler)
    LOGGER.info(
        "Starting UnifiedReward-Flex bridge on http://%s:%s via %s",
        args.host,
        args.port,
        args.api_base_url,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping UnifiedReward-Flex bridge.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
