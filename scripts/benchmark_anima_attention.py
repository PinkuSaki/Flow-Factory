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

"""Benchmark one Anima LoRA attention backend configuration."""

from __future__ import annotations

import argparse
import gc
import json
import time
from copy import deepcopy
from typing import Any, Dict

import torch
import yaml

from flow_factory.hparams import Arguments
from flow_factory.trainers import load_trainer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["flash", "sageattn"],
        required=True,
        help="Anima runtime attention backend.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Per-device batch size. The benchmark also uses this as group_size.",
    )
    parser.add_argument(
        "--base-config",
        default="examples/grpo/lora/anima.yaml",
        help="Base YAML config for the benchmark.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Square resolution used for training and sampling.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=10,
        help="Number of rollout denoising steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="CFG scale used in rollout.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for the benchmarked train step.",
    )
    return parser.parse_args()


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base``."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str) -> Dict[str, Any]:
    """Load one YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def snapshot_lora_parameters(trainer) -> Dict[str, torch.Tensor]:
    """Snapshot all LoRA parameters for delta-based validation."""
    snapshots: Dict[str, torch.Tensor] = {}
    for name, param in trainer.adapter.transformer.named_parameters():
        if "lora_" in name:
            snapshots[name] = param.detach().clone().cpu()
    if not snapshots:
        raise RuntimeError("Could not find any Anima LoRA parameters for benchmarking.")
    return snapshots


def summarize_parameter_updates(
    before: Dict[str, torch.Tensor],
    after: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Find the parameter with the largest absolute update."""
    best_name = None
    best_delta = -1.0
    changed_count = 0

    for name, before_tensor in before.items():
        after_tensor = after[name]
        delta = float((after_tensor - before_tensor).abs().sum().item())
        if delta > 0:
            changed_count += 1
        if delta > best_delta:
            best_name = name
            best_delta = delta

    assert best_name is not None
    return {
        "tracked_parameter": best_name,
        "tracked_parameter_abs_delta_sum": best_delta,
        "tracked_parameter_norm_before": float(before[best_name].norm().item()),
        "tracked_parameter_norm_after": float(after[best_name].norm().item()),
        "changed_parameter_count": changed_count,
    }


def cuda_peak_gib() -> float:
    """Return the current CUDA peak allocation in GiB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def synchronize() -> None:
    """Synchronize CUDA work if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(fn) -> tuple[Any, float, float]:
    """Run one callable and return ``(result, seconds, peak_gib)``."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    started_at = time.perf_counter()
    result = fn()
    synchronize()
    elapsed_seconds = time.perf_counter() - started_at
    return result, elapsed_seconds, cuda_peak_gib()


def build_config(args: argparse.Namespace) -> Arguments:
    """Build the benchmark configuration from the base YAML."""
    config_dict = load_yaml(args.base_config)
    benchmark_overrides = {
        "num_processes": 1,
        "main_process_port": 29620,
        "data": {
            "dataset_dir": "dataset/pickscore",
            "preprocessing_batch_size": 1,
            "dataloader_num_workers": 0,
            "force_reprocess": False,
            "max_dataset_size": 1,
            "sampler_type": "group_contiguous",
        },
        "model": {
            "attn_mode": args.backend,
            "split_attn": False,
        },
        "log": {
            "run_name": f"anima_{args.backend}_bs{args.batch_size}_bench",
            "project": "Flow-Factory-Benchmark",
            "logging_backend": "none",
            "save_dir": "saves/anima_attention_bench",
            "save_freq": 0,
            "save_model_only": True,
            "verbose": False,
        },
        "train": {
            "max_epochs": 1,
            "resolution": args.resolution,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "per_device_batch_size": args.batch_size,
            "group_size": args.batch_size,
            "unique_sample_num_per_epoch": 1,
            "gradient_step_per_epoch": 1,
            "num_inner_epochs": 1,
            "enable_gradient_checkpointing": args.gradient_checkpointing,
            "ema_device": "cpu",
            "ref_param_device": "cpu",
        },
        "eval": {
            "resolution": args.resolution,
            "per_device_batch_size": 1,
            "guidance_scale": args.guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "eval_freq": 0,
        },
        "rewards": [
            {
                "name": "synthetic_rank",
                "reward_model": "flow_factory.rewards.my_reward.MyGroupwiseRewardModel",
                "weight": 1.0,
                "batch_size": args.batch_size,
                "device": "cpu",
                "dtype": "float32",
            }
        ],
    }
    deep_update(config_dict, benchmark_overrides)
    return Arguments.from_dict(config_dict)


def cleanup_trainer(trainer) -> None:
    """Release trainer-related memory aggressively."""
    if trainer is not None:
        del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Benchmark one backend/batch-size pair."""
    config = build_config(args)
    setup_started_at = time.perf_counter()
    trainer = load_trainer(config)
    setup_seconds = time.perf_counter() - setup_started_at

    metrics: Dict[str, Any] = {
        "backend": args.backend,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "gradient_checkpointing": args.gradient_checkpointing,
        "setup_seconds": round(setup_seconds, 3),
        "attn_mode": config.model_args.attn_mode,
        "split_attn": config.model_args.split_attn,
        "trainer_type": config.training_args.trainer_type,
    }

    tracked_before = snapshot_lora_parameters(trainer)

    samples, sample_seconds, sample_peak_gib = timed_call(trainer.sample)
    _, feedback_seconds, feedback_peak_gib = timed_call(lambda: trainer.prepare_feedback(samples))
    _, optimize_seconds, optimize_peak_gib = timed_call(lambda: trainer.optimize(samples))

    tracked_after = snapshot_lora_parameters(trainer)

    metrics.update(
        {
            "sample_count": len(samples),
            "sample_seconds": round(sample_seconds, 3),
            "prepare_feedback_seconds": round(feedback_seconds, 3),
            "optimize_seconds": round(optimize_seconds, 3),
            "total_step_seconds": round(sample_seconds + feedback_seconds + optimize_seconds, 3),
            "sample_peak_gib": round(sample_peak_gib, 3),
            "prepare_feedback_peak_gib": round(feedback_peak_gib, 3),
            "optimize_peak_gib": round(optimize_peak_gib, 3),
            "sample_images_per_second": round(len(samples) / sample_seconds, 3),
            "train_images_per_second": round(
                len(samples) / (sample_seconds + feedback_seconds + optimize_seconds), 3
            ),
            "optimizer_steps": trainer.step,
            "image_shape": list(samples[0].image.shape),
        }
    )
    metrics.update(summarize_parameter_updates(tracked_before, tracked_after))

    trainer.accelerator.wait_for_everyone()
    cleanup_trainer(trainer)
    return metrics


def main() -> None:
    """Run the benchmark and print JSON metrics."""
    args = parse_args()
    try:
        result = run_benchmark(args)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            cleanup_trainer(None)
            print(
                json.dumps(
                    {
                        "backend": args.backend,
                        "batch_size": args.batch_size,
                        "resolution": args.resolution,
                        "status": "oom",
                        "error": str(exc),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
