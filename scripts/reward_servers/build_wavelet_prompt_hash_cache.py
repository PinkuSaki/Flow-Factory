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

"""Build a Haar wavelet prompt-hash reference cache.

This script is intended to run before training. The training-time reward server
loads the resulting cache and never reads reference images.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from PIL import Image
from tqdm import tqdm
from wavelet_prompt_hash_common import (
    WaveletDescriptorConfig,
    WaveletImageOutputs,
    extract_wavelet_outputs,
    prompt_sha256,
    read_jsonl,
    resolve_reference_image_path,
    save_reference_cache,
)

LOGGER = logging.getLogger("build_wavelet_prompt_hash_cache")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        required=True,
        help="JSONL containing prompts and reference image paths.",
    )
    parser.add_argument(
        "--reference-image-dir",
        type=Path,
        default=None,
        help="Base directory for relative reference image paths.",
    )
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="JSONL field containing the prompt text.",
    )
    parser.add_argument(
        "--image-field",
        default="source_image",
        help="JSONL field containing the reference image path.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        required=True,
        help="Destination torch cache path for prompt-hash wavelet descriptors.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cache-path if it already exists.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Square image size used before Haar decomposition.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Number of hierarchical Haar decomposition levels.",
    )
    parser.add_argument(
        "--color-space",
        choices=("y", "rgb", "ycbcr"),
        default="y",
        help="Color representation used for wavelet energy extraction.",
    )
    parser.add_argument(
        "--include-approximation",
        action="store_true",
        help="Append final LL approximation energy to the descriptor.",
    )
    parser.add_argument(
        "--level-weight-decay",
        type=float,
        default=1.0,
        help=(
            "Per-level energy multiplier decay. Level 1 uses 1.0, level 2 uses this "
            "value, and so on."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level.",
    )
    return parser.parse_args()


def _build_hash_to_path(
    reference_jsonl: Path,
    reference_image_dir: Path | None,
    prompt_field: str,
    image_field: str,
) -> dict[str, Path]:
    """Build a prompt-hash to reference image path mapping with duplicate checks."""
    records = read_jsonl(reference_jsonl)
    if not records:
        raise ValueError(f"No records found in {reference_jsonl}.")

    hash_to_path: dict[str, Path] = {}
    hash_to_line: dict[str, int] = {}
    for line_number, record in records:
        if prompt_field not in record:
            raise KeyError(
                f"Missing prompt field {prompt_field!r} at {reference_jsonl}:{line_number}."
            )
        if image_field not in record:
            raise KeyError(
                f"Missing image field {image_field!r} at {reference_jsonl}:{line_number}."
            )

        prompt_hash = prompt_sha256(str(record[prompt_field]))
        if prompt_hash in hash_to_line:
            raise ValueError(
                f"Duplicate prompt hash {prompt_hash} at lines "
                f"{hash_to_line[prompt_hash]} and {line_number}. "
                "Prompt-hash lookup requires globally unique prompts."
            )

        hash_to_line[prompt_hash] = line_number
        hash_to_path[prompt_hash] = resolve_reference_image_path(
            jsonl_path=reference_jsonl,
            reference_image_dir=reference_image_dir,
            raw_path=str(record[image_field]),
        )

    return hash_to_path


def build_cache(args: argparse.Namespace) -> None:
    """Build and save the reference wavelet descriptor cache."""
    if args.cache_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Cache already exists: {args.cache_path}. Pass --overwrite to replace it."
        )

    config = WaveletDescriptorConfig(
        image_size=args.image_size,
        levels=args.levels,
        color_space=args.color_space,
        include_approximation=args.include_approximation,
        level_weight_decay=args.level_weight_decay,
    )
    config.validate()

    hash_to_path = _build_hash_to_path(
        reference_jsonl=args.reference_jsonl,
        reference_image_dir=args.reference_image_dir,
        prompt_field=args.prompt_field,
        image_field=args.image_field,
    )
    prompt_hashes = list(hash_to_path.keys())
    outputs_by_hash: dict[str, WaveletImageOutputs] = {}

    LOGGER.info(
        "Building %s Haar wavelet reference descriptors from %s",
        len(prompt_hashes),
        args.reference_jsonl,
    )
    for prompt_hash in tqdm(prompt_hashes, desc="Wavelet reference cache"):
        with Image.open(hash_to_path[prompt_hash]) as image:
            outputs_by_hash[prompt_hash] = extract_wavelet_outputs([image.convert("RGB")], config)

    save_reference_cache(
        cache_path=args.cache_path,
        prompt_hashes=prompt_hashes,
        outputs_by_hash=outputs_by_hash,
        metadata={
            **config.to_metadata(),
            "prompt_field": args.prompt_field,
            "image_field": args.image_field,
            "reference_jsonl": str(args.reference_jsonl),
            "cache_fields": ["distributions", "log_energies", "total_energies"],
        },
    )
    LOGGER.info("Saved Haar wavelet reference cache to %s", args.cache_path)


def main() -> None:
    """Run the cache builder."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    build_cache(args)


if __name__ == "__main__":
    main()
