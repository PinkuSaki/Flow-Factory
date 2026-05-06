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

"""CLI for converting Flow-Factory Anima LoRA checkpoints to sd-scripts format."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flow_factory.utils.anima_lora_conversion import (  # noqa: E402
    ConversionSummary,
    convert_flow_factory_anima_lora,
    describe_summary,
)

ADAPTER_WEIGHTS_CANDIDATES = ("adapter_model.safetensors", "adapter_model.bin")


def _has_adapter_weights(directory: Path) -> bool:
    """Return whether a directory contains PEFT adapter weights."""
    return any((directory / filename).exists() for filename in ADAPTER_WEIGHTS_CANDIDATES)


def _checkpoint_sort_key(path: Path) -> list[int | str]:
    """Sort checkpoint paths naturally, e.g. checkpoint-2 before checkpoint-10."""
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def _discover_checkpoint_dirs(input_path: Path) -> list[Path]:
    """Discover direct checkpoint subdirectories that contain adapter weights."""
    if not input_path.is_dir():
        return []
    return sorted(
        (
            child
            for child in input_path.glob("checkpoint-*")
            if child.is_dir() and _has_adapter_weights(child)
        ),
        key=_checkpoint_sort_key,
    )


def _should_batch_convert(input_path: Path, checkpoint_dirs: list[Path]) -> bool:
    """Return whether the input path should be treated as a checkpoint collection."""
    if not checkpoint_dirs:
        return False
    if _has_adapter_weights(input_path):
        return False
    if any(
        _has_adapter_weights(input_path / component)
        for component in ("transformer", "text_encoder")
    ):
        return False
    return input_path.name == "checkpoints" or len(checkpoint_dirs) > 1


def _convert_checkpoint_collection(
    checkpoint_dirs: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[ConversionSummary]:
    """Convert every checkpoint directory into one sd-scripts safetensors file."""
    if output_dir.suffix == ".safetensors":
        raise ValueError(
            "Batch checkpoint conversion requires output_path to be a directory, "
            "not a .safetensors file."
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"Batch output path exists and is not a directory: {output_dir}")

    summaries: list[ConversionSummary] = []
    for checkpoint_dir in checkpoint_dirs:
        output_path = output_dir / f"{checkpoint_dir.name}.safetensors"
        summary = convert_flow_factory_anima_lora(
            input_path=checkpoint_dir,
            output_path=output_path,
            component_arg=args.component_type,
            save_dtype_name=args.save_dtype,
            default_alpha=args.default_alpha,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        print(describe_summary(summary))
    return summaries


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert Flow-Factory Anima LoRA checkpoint directories into "
            "sd-scripts-compatible .safetensors files."
        )
    )
    parser.add_argument(
        "input_path",
        help=(
            "Flow-Factory LoRA directory, component directory, adapter file, or a "
            "directory containing checkpoint-* subdirectories."
        ),
    )
    parser.add_argument(
        "output_path",
        help=(
            "Destination .safetensors file for single-checkpoint conversion, or "
            "destination directory for checkpoint collection conversion."
        ),
    )
    parser.add_argument(
        "--component-type",
        choices=["auto", "transformer", "text_encoder"],
        default="auto",
        help="Component type hint when the input points to a single adapter file or directory.",
    )
    parser.add_argument(
        "--save-dtype",
        choices=["keep", "fp16", "bf16", "fp32"],
        default="keep",
        help="Optional dtype cast for LoRA weights before saving.",
    )
    parser.add_argument(
        "--default-alpha",
        type=float,
        default=None,
        help="Fallback alpha when adapter_config.json is unavailable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    return parser


def main() -> int:
    """Run the CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    checkpoint_dirs = _discover_checkpoint_dirs(input_path)
    if _should_batch_convert(input_path, checkpoint_dirs):
        summaries = _convert_checkpoint_collection(
            checkpoint_dirs=checkpoint_dirs,
            output_dir=output_path,
            args=args,
        )
        print(f"Converted {len(summaries)} checkpoints into {output_path}.")
        return 0

    summary = convert_flow_factory_anima_lora(
        input_path=input_path,
        output_path=output_path,
        component_arg=args.component_type,
        save_dtype_name=args.save_dtype,
        default_alpha=args.default_alpha,
        overwrite=args.overwrite,
    )
    print(describe_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
