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

"""Build a WD ConvNeXt prompt-hash preprocessed-image cache.

This script stores reference images after the exact WD ConvNeXt input resizing
step. The training-time server loads the resulting uint8 tensor cache into CPU
memory and computes ConvNeXt features on demand.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image
from wd_convnext_perceptual_common import (
    load_convnext_preprocess_config,
    preprocess_image_to_uint8,
    save_preprocessed_image_cache,
)
from wd_prompt_hash_common import prompt_sha256, read_jsonl, resolve_reference_image_path

LOGGER = logging.getLogger("build_wd_convnext_perceptual_cache")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("../models/wd-convnext-tagger-v3"),
        help="Path to the local WD ConvNeXt tagger checkpoint directory.",
    )
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
        help="Destination torch cache path for prompt-hash preprocessed images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cache-path if it already exists.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress after this many images.",
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
    """Build and save the reference image cache."""
    if args.cache_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Cache already exists: {args.cache_path}. Pass --overwrite to replace it."
        )
    if args.log_every <= 0:
        raise ValueError(f"log_every must be positive, got {args.log_every}.")

    preprocess_config = load_convnext_preprocess_config(args.model_path)
    hash_to_path = _build_hash_to_path(
        reference_jsonl=args.reference_jsonl,
        reference_image_dir=args.reference_image_dir,
        prompt_field=args.prompt_field,
        image_field=args.image_field,
    )
    prompt_hashes = list(hash_to_path.keys())
    images = torch.empty(
        (
            len(prompt_hashes),
            3,
            preprocess_config.image_size,
            preprocess_config.image_size,
        ),
        dtype=torch.uint8,
    )

    LOGGER.info(
        "Building %s WD ConvNeXt preprocessed reference images from %s",
        len(prompt_hashes),
        args.reference_jsonl,
    )
    for index, prompt_hash in enumerate(prompt_hashes):
        with Image.open(hash_to_path[prompt_hash]) as image:
            images[index] = preprocess_image_to_uint8(
                image=image,
                image_size=preprocess_config.image_size,
            )
        if (index + 1) % args.log_every == 0 or index + 1 == len(prompt_hashes):
            LOGGER.info("Preprocessed %s/%s reference images", index + 1, len(prompt_hashes))

    save_preprocessed_image_cache(
        cache_path=args.cache_path,
        prompt_hashes=prompt_hashes,
        images=images,
        metadata={
            "prompt_field": args.prompt_field,
            "image_field": args.image_field,
            "model_path": str(args.model_path),
            "reference_jsonl": str(args.reference_jsonl),
            "image_size": preprocess_config.image_size,
            "image_mean": list(preprocess_config.image_mean),
            "image_std": list(preprocess_config.image_std),
            "cache_fields": ["images"],
            "image_dtype": "uint8",
            "image_layout": "NCHW",
            "preprocess": "pad_square_white_resize_bicubic",
        },
    )
    LOGGER.info(
        "Saved WD ConvNeXt preprocessed image cache to %s with tensor shape %s",
        args.cache_path,
        tuple(images.shape),
    )


def main() -> None:
    """Run the cache builder."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    build_cache(args)


if __name__ == "__main__":
    main()
