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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flow_factory.utils.anima_lora_conversion import (  # noqa: E402
    convert_flow_factory_anima_lora,
    describe_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Flow-Factory Anima LoRA checkpoint directory into a single "
            "sd-scripts-compatible .safetensors file."
        )
    )
    parser.add_argument("input_path", help="Flow-Factory LoRA directory, component directory, or adapter file.")
    parser.add_argument("output_path", help="Destination .safetensors file for sd-scripts.")
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

    summary = convert_flow_factory_anima_lora(
        input_path=args.input_path,
        output_path=args.output_path,
        component_arg=args.component_type,
        save_dtype_name=args.save_dtype,
        default_alpha=args.default_alpha,
        overwrite=args.overwrite,
    )
    print(describe_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
