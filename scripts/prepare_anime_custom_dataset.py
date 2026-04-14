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

"""Prepare the anime_custom dataset for Flow-Factory training.

This script converts ``anime10k.json`` into ``train.jsonl`` and ``test.jsonl`` while
preserving the original Markdown caption layout. The only content transformation is
removing the ``Artist`` section from each caption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("dataset/anime_custom/anime10k.json"),
        help="Path to the source JSON mapping image names to captions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/anime_custom"),
        help="Directory where train/test JSONL files will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=256,
        help="Number of prompts to place in the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )
    return parser.parse_args()


def _normalize_newlines(text: str) -> str:
    """Normalize line endings to Unix style."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_artist_section(markdown_text: str) -> str:
    """Remove the ``Artist`` section while preserving other Markdown blocks."""
    lines = _normalize_newlines(markdown_text).split("\n")
    kept_lines: list[str] = []
    skipping_artist = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].strip().rstrip(":").strip().lower()
            skipping_artist = heading == "artist"
            if skipping_artist:
                continue

        if not skipping_artist:
            kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def load_source_records(input_json: Path) -> list[dict[str, Any]]:
    """Load and normalize source records from the raw JSON file."""
    with input_json.open("r", encoding="utf-8") as handle:
        raw_mapping = json.load(handle)

    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Expected a JSON object in {input_json}, got {type(raw_mapping).__name__}")

    records: list[dict[str, Any]] = []
    for source_image, payload in sorted(raw_mapping.items()):
        if not isinstance(payload, dict):
            raise ValueError(
                f"Expected payload for {source_image} to be a JSON object, "
                f"got {type(payload).__name__}"
            )

        prompt = strip_artist_section(str(payload.get("caption", "")))
        if not prompt:
            continue

        records.append(
            {
                "prompt": prompt,
                "source_image": source_image,
                "image_size": payload.get("image_size"),
            }
        )

    return records


def split_records(
    records: list[dict[str, Any]],
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into train and test subsets deterministically."""
    if test_size < 0:
        raise ValueError(f"test_size must be non-negative, got {test_size}")
    if test_size >= len(records):
        raise ValueError(
            f"test_size({test_size}) must be smaller than the dataset size({len(records)})"
        )

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    test_records = shuffled[:test_size]
    train_records = shuffled[test_size:]
    return train_records, test_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to a JSON Lines file."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_manifest(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Build manifest entries with prompt hashes for traceability."""
    manifest: list[dict[str, Any]] = []
    for record in records:
        manifest.append(
            {
                "split": split,
                "source_image": record["source_image"],
                "image_size": record.get("image_size"),
                "prompt_sha256": hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest(),
            }
        )
    return manifest


def main() -> None:
    """Run the dataset conversion pipeline."""
    args = parse_args()
    records = load_source_records(args.input_json)
    train_records, test_records = split_records(records, test_size=args.test_size, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.output_dir / "train.jsonl"
    test_path = args.output_dir / "test.jsonl"
    manifest_path = args.output_dir / "split_manifest.jsonl"

    write_jsonl(train_path, train_records)
    write_jsonl(test_path, test_records)
    write_jsonl(
        manifest_path,
        build_manifest(train_records, "train") + build_manifest(test_records, "test"),
    )

    print(
        json.dumps(
            {
                "input_json": str(args.input_json),
                "output_dir": str(args.output_dir),
                "train_size": len(train_records),
                "test_size": len(test_records),
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
