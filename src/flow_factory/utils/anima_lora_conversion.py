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

"""Convert Flow-Factory Anima LoRA checkpoints into sd-scripts format."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch
from safetensors.torch import load_file, save_file

ComponentType = Literal["transformer", "text_encoder"]
ComponentArg = Literal["auto", "transformer", "text_encoder"]

ADAPTER_WEIGHTS_CANDIDATES = ("adapter_model.safetensors", "adapter_model.bin")
KNOWN_COMPONENT_DIRS: tuple[ComponentType, ...] = ("transformer", "text_encoder")
SAVE_DTYPE_MAP: dict[str, Optional[torch.dtype]] = {
    "keep": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
COMPONENT_PREFIX = {
    "transformer": "lora_unet",
    "text_encoder": "lora_te",
}
PEFT_LORA_PATTERN = re.compile(
    r"^(?:base_model\.model\.)?(?P<module>.+)\.lora_(?P<kind>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$"
)


@dataclass(frozen=True)
class LoadedComponent:
    """Loaded Flow-Factory LoRA component."""

    component_type: ComponentType
    weights_path: Path
    config_path: Optional[Path]
    state_dict: Dict[str, torch.Tensor]
    config: Dict[str, Any]


@dataclass(frozen=True)
class ConversionSummary:
    """Summary of a completed conversion."""

    input_path: Path
    output_path: Path
    module_count: int
    tensor_count: int
    component_module_count: Dict[str, int]


def _load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    """Load a safetensors or PyTorch state dict onto CPU."""
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _find_weights_file(directory: Path) -> Optional[Path]:
    """Find the adapter weights file inside a PEFT directory."""
    for filename in ADAPTER_WEIGHTS_CANDIDATES:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    safetensors_files = sorted(directory.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        return safetensors_files[0]

    bin_files = sorted(directory.glob("*.bin"))
    if len(bin_files) == 1:
        return bin_files[0]

    return None


def _normalize_module_path(module_path: str, component_type: ComponentType) -> str:
    """Normalize a PEFT module path to the raw module name used by sd-scripts."""
    prefixes = [f"{component_type}."]
    if component_type == "transformer":
        prefixes.extend(("model.", "module."))

    for prefix in prefixes:
        if module_path.startswith(prefix):
            module_path = module_path[len(prefix) :]

    return module_path


def _guess_component_type(state_dict: Dict[str, torch.Tensor]) -> ComponentType:
    """Infer component type from module paths when the input is a single PEFT directory."""
    text_encoder_hits = 0
    transformer_hits = 0

    for key in state_dict.keys():
        match = PEFT_LORA_PATTERN.match(key)
        if match is None:
            continue

        module_path = match.group("module")
        if module_path.startswith(("model.layers.", "embed_tokens.", "lm_head.", "norm.")):
            text_encoder_hits += 1
        elif module_path.startswith(
            (
                "blocks.",
                "x_embedder.",
                "final_layer.",
                "t_embedder.",
                "y_embedder.",
                "context_embedder.",
                "llm_adapter.",
                "transformer.",
            )
        ):
            transformer_hits += 1

    if text_encoder_hits > transformer_hits:
        return "text_encoder"
    return "transformer"


def _load_component(
    weights_path: Path,
    config_path: Optional[Path],
    component_arg: ComponentArg,
) -> LoadedComponent:
    """Load one LoRA component bundle from disk."""
    state_dict = _load_state_dict(weights_path)
    config: Dict[str, Any] = {}
    if config_path is not None and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    component_type: ComponentType
    if component_arg == "auto":
        component_type = _guess_component_type(state_dict)
    else:
        component_type = component_arg

    return LoadedComponent(
        component_type=component_type,
        weights_path=weights_path,
        config_path=config_path,
        state_dict=state_dict,
        config=config,
    )


def discover_components(input_path: os.PathLike[str] | str, component_arg: ComponentArg) -> list[LoadedComponent]:
    """Discover Flow-Factory LoRA component bundles from a file or directory."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        config_path = path.with_name("adapter_config.json")
        config_path = config_path if config_path.exists() else None
        return [_load_component(path, config_path, component_arg)]

    direct_weights = _find_weights_file(path)
    if direct_weights is not None:
        config_path = path / "adapter_config.json"
        return [_load_component(direct_weights, config_path if config_path.exists() else None, component_arg)]

    discovered: list[LoadedComponent] = []
    for component_name in KNOWN_COMPONENT_DIRS:
        component_dir = path / component_name
        weights_path = _find_weights_file(component_dir)
        if weights_path is None:
            continue

        config_path = component_dir / "adapter_config.json"
        discovered.append(
            _load_component(
                weights_path=weights_path,
                config_path=config_path if config_path.exists() else None,
                component_arg=component_name,
            )
        )

    if discovered:
        return discovered

    subdirs_with_weights = []
    for child in sorted(path.iterdir()):
        if not child.is_dir():
            continue
        weights_path = _find_weights_file(child)
        if weights_path is not None:
            subdirs_with_weights.append(child)

    if len(subdirs_with_weights) == 1:
        child = subdirs_with_weights[0]
        weights_path = _find_weights_file(child)
        config_path = child / "adapter_config.json"
        return [_load_component(weights_path, config_path if config_path.exists() else None, component_arg)]

    raise FileNotFoundError(
        f"Could not find a Flow-Factory LoRA adapter under {path}. "
        "Expected an adapter_model.safetensors/.bin file or transformer/text_encoder subdirectories."
    )


def _resolve_global_alpha(component: LoadedComponent, default_alpha: Optional[float]) -> float:
    """Resolve the alpha value for a component."""
    if default_alpha is not None:
        return float(default_alpha)

    if "lora_alpha" in component.config and component.config["lora_alpha"] is not None:
        return float(component.config["lora_alpha"])

    raise ValueError(
        f"Cannot determine LoRA alpha for {component.weights_path}. "
        "Provide --default-alpha or keep adapter_config.json next to the weights."
    )


def _resolve_alpha_pattern(component: LoadedComponent) -> Dict[str, float]:
    """Resolve per-module alpha overrides from adapter_config.json when available."""
    alpha_pattern = component.config.get("alpha_pattern")
    if not isinstance(alpha_pattern, dict):
        return {}

    normalized_pattern: Dict[str, float] = {}
    for key, value in alpha_pattern.items():
        if value is None:
            continue
        normalized_pattern[str(key)] = float(value)

    return normalized_pattern


def _resolve_module_alpha(
    module_path: str,
    component_type: ComponentType,
    global_alpha: float,
    alpha_pattern: Dict[str, float],
) -> float:
    """Resolve the alpha value for a single module."""
    candidates = [
        module_path,
        f"{component_type}.{module_path}",
        f"base_model.model.{module_path}",
        f"base_model.model.{component_type}.{module_path}",
    ]
    for candidate in candidates:
        if candidate in alpha_pattern:
            return alpha_pattern[candidate]
    return global_alpha


def _convert_tensor_dtype(tensor: torch.Tensor, save_dtype: Optional[torch.dtype]) -> torch.Tensor:
    """Convert one tensor to the requested save dtype while keeping it on CPU."""
    tensor = tensor.detach().cpu().contiguous()
    if save_dtype is not None and tensor.is_floating_point():
        tensor = tensor.to(dtype=save_dtype)
    return tensor


def convert_component_state_dict(
    component: LoadedComponent,
    save_dtype: Optional[torch.dtype],
    default_alpha: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Convert one Flow-Factory component state dict into sd-scripts key layout."""
    if component.config.get("use_dora"):
        raise ValueError(
            f"DoRA adapters are not supported by this converter: {component.weights_path}"
        )

    global_alpha = _resolve_global_alpha(component, default_alpha)
    alpha_pattern = _resolve_alpha_pattern(component)
    sd_scripts_state_dict: Dict[str, torch.Tensor] = {}
    unsupported_keys: list[str] = []

    for key, tensor in component.state_dict.items():
        match = PEFT_LORA_PATTERN.match(key)
        if match is None:
            if "lora_" in key or key.endswith(".alpha"):
                unsupported_keys.append(key)
            continue

        module_path = _normalize_module_path(match.group("module"), component.component_type)
        lora_name = f"{COMPONENT_PREFIX[component.component_type]}_{module_path.replace('.', '_')}"
        kind = match.group("kind")
        suffix = "lora_down.weight" if kind == "A" else "lora_up.weight"
        new_key = f"{lora_name}.{suffix}"
        if new_key in sd_scripts_state_dict:
            raise ValueError(f"Duplicate converted key detected: {new_key}")

        sd_scripts_state_dict[new_key] = _convert_tensor_dtype(tensor, save_dtype)

        alpha_key = f"{lora_name}.alpha"
        if alpha_key not in sd_scripts_state_dict:
            alpha_value = _resolve_module_alpha(
                module_path=module_path,
                component_type=component.component_type,
                global_alpha=global_alpha,
                alpha_pattern=alpha_pattern,
            )
            sd_scripts_state_dict[alpha_key] = torch.tensor(alpha_value, dtype=torch.float32)

    if unsupported_keys:
        unsupported_text = ", ".join(sorted(unsupported_keys[:5]))
        raise ValueError(
            f"Unsupported LoRA keys found in {component.weights_path}: {unsupported_text}"
        )

    return sd_scripts_state_dict


def validate_sd_scripts_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Validate the converted sd-scripts state dict and return component-wise counts."""
    modules: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_down.weight"):
            module_name = key[: -len(".lora_down.weight")]
            modules.setdefault(module_name, {})["down"] = tensor
        elif key.endswith(".lora_up.weight"):
            module_name = key[: -len(".lora_up.weight")]
            modules.setdefault(module_name, {})["up"] = tensor
        elif key.endswith(".alpha"):
            module_name = key[: -len(".alpha")]
            modules.setdefault(module_name, {})["alpha"] = tensor
        else:
            raise ValueError(f"Unexpected sd-scripts key: {key}")

    if not modules:
        raise ValueError("No LoRA modules were converted.")

    component_counts = {"transformer": 0, "text_encoder": 0}
    for module_name, tensors in modules.items():
        missing = {"down", "up", "alpha"} - set(tensors.keys())
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Converted module {module_name} is missing tensors: {missing_text}")

        down = tensors["down"]
        up = tensors["up"]
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError(f"Only Linear LoRA tensors are supported, got {module_name}")
        if down.shape[0] != up.shape[1]:
            raise ValueError(
                f"Rank mismatch for {module_name}: down={tuple(down.shape)}, up={tuple(up.shape)}"
            )

        if module_name.startswith("lora_unet_"):
            component_counts["transformer"] += 1
        elif module_name.startswith("lora_te_"):
            component_counts["text_encoder"] += 1
        else:
            raise ValueError(f"Unexpected module prefix: {module_name}")

    return {key: value for key, value in component_counts.items() if value > 0}


def convert_flow_factory_anima_lora(
    input_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    component_arg: ComponentArg = "auto",
    save_dtype_name: str = "keep",
    default_alpha: Optional[float] = None,
    overwrite: bool = False,
) -> ConversionSummary:
    """Convert a Flow-Factory Anima LoRA checkpoint into sd-scripts safetensors format."""
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output}")
    if output.suffix != ".safetensors":
        raise ValueError("Output file must end with .safetensors")

    if save_dtype_name not in SAVE_DTYPE_MAP:
        valid = ", ".join(sorted(SAVE_DTYPE_MAP))
        raise ValueError(f"Invalid save dtype: {save_dtype_name}. Valid options: {valid}")

    components = discover_components(input_path, component_arg)
    save_dtype = SAVE_DTYPE_MAP[save_dtype_name]

    converted_state_dict: Dict[str, torch.Tensor] = {}
    for component in components:
        component_state_dict = convert_component_state_dict(
            component=component,
            save_dtype=save_dtype,
            default_alpha=default_alpha,
        )
        overlap = set(converted_state_dict).intersection(component_state_dict)
        if overlap:
            duplicate_key = sorted(overlap)[0]
            raise ValueError(f"Duplicate key produced across components: {duplicate_key}")
        converted_state_dict.update(component_state_dict)

    component_module_count = validate_sd_scripts_state_dict(converted_state_dict)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "sd-scripts-anima-lora",
        "converted_from": "flow-factory-anima-lora",
        "source_path": str(Path(input_path).expanduser().resolve()),
        "components": ",".join(component.component_type for component in components),
    }
    save_file(converted_state_dict, str(output), metadata=metadata)

    return ConversionSummary(
        input_path=Path(input_path).expanduser().resolve(),
        output_path=output,
        module_count=sum(component_module_count.values()),
        tensor_count=len(converted_state_dict),
        component_module_count=component_module_count,
    )


def describe_summary(summary: ConversionSummary) -> str:
    """Format a human-readable conversion summary."""
    component_text = ", ".join(
        f"{name}={count}" for name, count in sorted(summary.component_module_count.items())
    )
    return (
        f"Converted {summary.module_count} LoRA modules "
        f"({component_text}) into {summary.output_path} "
        f"with {summary.tensor_count} tensors."
    )
