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

import logging
import math

# src/flow_factory/hparams/model_args.py
import os
from dataclasses import asdict, dataclass, field
from typing import Any, List, Literal, Optional, Union

import torch
import yaml

from .abc import ArgABC

dtype_map = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass
class ModelArguments(ArgABC):
    r"""Arguments pertaining to model configuration."""

    model_name_or_path: str = field(
        default="black-forest-labs/FLUX.1-dev",
        metadata={
            "help": "Path to pre-trained model or model identifier from huggingface.co/models"
        },
    )

    finetune_type: Literal["full", "lora"] = field(
        default="full", metadata={"help": "Fine-tuning type. Options are ['full', 'lora']"}
    )

    trainable_parameters_dtype: Union[
        Literal["fp32", "bf16", "fp16", "float16", "bfloat16", "float32"], torch.dtype
    ] = field(
        default="bfloat16",
        metadata={
            "help": "Torch dtype for all trainable parameters (`requires_grad=True`) -- i.e. the "
            "optimizer 'master weight' precision. (Renamed from the misnamed "
            "`master_weight_dtype`, which despite its name only ever set the *trainable* "
            "parameter dtype.)"
        },
    )
    frozen_parameters_dtype: Optional[
        Union[Literal["fp32", "bf16", "fp16", "float16", "bfloat16", "float32"], torch.dtype]
    ] = field(
        default=None,
        metadata={
            "help": "Torch dtype for frozen (`requires_grad=False`) parameters and floating-point "
            "buffers. `None` (default) preserves each frozen component's original "
            "`from_pretrained` dtype and never downcasts -- released checkpoints deliberately "
            "ship components in different dtypes (e.g. Z-Image: transformer fp32, text encoder "
            "bf16), so forcing one uniform frozen dtype would override those choices. Set an "
            "explicit dtype (e.g. 'bf16') to cast all frozen params to it, e.g. to save memory."
        },
    )

    target_components: Union[str, List[str]] = field(
        default="transformer",
        metadata={
            "help": "Which components to fine-tune. Options are like ['transformer', 'transformer_2', ['transformer', 'transformer_2']]"
        },
    )
    target_modules: Union[str, List[str]] = field(
        default="all",
        metadata={
            "help": "Which layers to fine-tune. Options are like ['all',  'default', 'to_q', ['to_q', 'to_k', 'to_v']]"
        },
    )

    model_type: Literal[
        "sd3-5",
        "flux1",
        "flux1-kontext",
        "flux2",
        "flux2-klein",
        "qwen-image",
        "qwen-image-edit-plus",
        "z-image",
        "wan2_t2v",
        "wan2_i2v",
        "wan2_v2v",
        "bagel",
        "ltx2_t2av",
        "ltx2_i2av",
        "anima",
    ] = field(
        default="flux1",
        metadata={
            "help": "Registered model adapter key (see models/registry.py), or a custom 'pkg.module.Adapter' python path."
        },
    )

    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank for LoRA adapters."},
    )

    lora_alpha: Optional[int] = field(
        default=None,
        metadata={
            "help": "Alpha scaling factor for LoRA adapters. Default to `2 * lora_rank` if None."
        },
    )

    resume_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Resume from checkpoint. Accepts either a local directory or a "
            "Hugging Face repo spec ('owner/repo[/subfolder][@revision]', or "
            "explicit 'hf://owner/repo[/subfolder][@revision]'). When a local "
            "path doesn't exist, falls back to Hugging Face Hub download. "
            "Multi-node: HF_TOKEN must be set on every node; downloads happen "
            "once per node; consider HF_HUB_ENABLE_HF_TRANSFER=1 for large "
            "checkpoints to avoid NCCL watchdog timeouts."
        },
    )

    resume_type: Optional[Literal["lora", "full", "state"]] = field(
        default=None,
        metadata={
            "help": "Type of checkpoint to load from resume_path. "
            "'lora': Load LoRA adapters only. "
            "'full': Load full model weights. "
            "'state': Load full training state (model + optimizer). "
            "If None, auto-detect based on finetune_type."
        },
    )

    attn_backend: Optional[str] = field(
        default=None,
        metadata={
            "help": "Attention backend for transformers. "
            "Options: 'native', 'flash', 'flash_hub', '_flash_3', '_flash_3_hub', 'sage', 'xformers'. "
            "None means use diffusers default."
            "See https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends for all details."
        },
    )

    qwen3: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the Qwen3 text encoder checkpoint used by Anima."},
    )
    vae: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the VAE checkpoint used by Anima."},
    )
    llm_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to a standalone Anima LLM adapter checkpoint."},
    )
    t5_tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the T5 tokenizer directory used by Anima."},
    )
    qwen3_max_token_length: int = field(
        default=512,
        metadata={"help": "Maximum Qwen3 token length for Anima prompt encoding."},
    )
    t5_max_token_length: int = field(
        default=512,
        metadata={"help": "Maximum T5 token length for Anima LLM adapter inputs."},
    )
    sd_scripts_root: str = field(
        default="~/sd-scripts",
        metadata={"help": "Root directory of the sd-scripts repository used by Anima."},
    )
    attn_mode: Optional[Literal["torch", "xformers", "flash", "sageattn", "sdpa"]] = field(
        default="torch",
        metadata={"help": "Attention implementation used by Anima runtime modules."},
    )
    split_attn: bool = field(
        default=True,
        metadata={"help": "Whether to enable split attention for Anima."},
    )
    vae_chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "Optional VAE spatial chunk size for Anima memory reduction."},
    )
    vae_disable_cache: bool = field(
        default=False,
        metadata={"help": "Disable the internal Qwen-Image VAE cache for Anima."},
    )
    discrete_flow_shift: float = field(
        default=1.0,
        metadata={"help": "Shift factor used to initialize the Anima flow scheduler."},
    )

    def __post_init__(self):
        if isinstance(self.trainable_parameters_dtype, str):
            self.trainable_parameters_dtype = dtype_map[self.trainable_parameters_dtype]
        if isinstance(self.frozen_parameters_dtype, str):
            self.frozen_parameters_dtype = dtype_map[self.frozen_parameters_dtype]

        # Normalize target_components to list
        if isinstance(self.target_components, str):
            self.target_components = [self.target_components]

        if isinstance(self.target_modules, str):
            if self.target_modules not in ["all", "default"]:
                self.target_modules = [self.target_modules]

        if self.lora_alpha is None:
            self.lora_alpha = 2 * self.lora_rank

        self.model_name_or_path = os.path.expanduser(self.model_name_or_path)
        self.qwen3 = os.path.expanduser(self.qwen3) if self.qwen3 is not None else None
        self.vae = os.path.expanduser(self.vae) if self.vae is not None else None
        self.llm_adapter_path = (
            os.path.expanduser(self.llm_adapter_path) if self.llm_adapter_path is not None else None
        )
        self.t5_tokenizer_path = (
            os.path.expanduser(self.t5_tokenizer_path)
            if self.t5_tokenizer_path is not None
            else None
        )
        self.sd_scripts_root = os.path.expanduser(self.sd_scripts_root)
        self.resume_path = (
            os.path.expanduser(self.resume_path) if self.resume_path is not None else None
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["trainable_parameters_dtype"] = str(self.trainable_parameters_dtype).split(".")[-1]
        if self.frozen_parameters_dtype is not None:
            d["frozen_parameters_dtype"] = str(self.frozen_parameters_dtype).split(".")[-1]
        return d

    def __str__(self) -> str:
        """Pretty print configuration as YAML."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False, indent=2)

    def __repr__(self) -> str:
        """Same as __str__ for consistency."""
        return self.__str__()
