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

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import DiffusionPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler as DiffusersFlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from safetensors.torch import load_file

from ...hparams import Arguments
from ...samples import T2ISample
from ...scheduler import (
    FlowMatchEulerDiscreteSDEScheduler,
    FlowMatchEulerDiscreteSDESchedulerOutput,
    SDESchedulerOutput,
)
from ...utils.logger_utils import setup_logger
from ...utils.trajectory_collector import (
    CallbackCollector,
    TrajectoryIndicesType,
    create_callback_collector,
    create_trajectory_collector,
)
from ..abc import BaseAdapter

logger = setup_logger(__name__)

ANIMA_DEFAULT_LORA_EXCLUDE_PATTERN = r"^llm_adapter(?:\..*)?$"


def _resolve_sd_scripts_root(sd_scripts_root: str) -> str:
    """Resolve and validate the sd-scripts root directory."""
    resolved = os.path.abspath(os.path.expanduser(sd_scripts_root))
    if not os.path.isdir(resolved):
        raise FileNotFoundError(
            f"sd_scripts_root does not exist: {resolved}. "
            "Set `model.sd_scripts_root` to a valid sd-scripts checkout."
        )
    return resolved


def _load_anima_runtime_modules(sd_scripts_root: str) -> tuple[Any, Any]:
    """Import Anima runtime helpers from the external sd-scripts checkout."""
    resolved_root = _resolve_sd_scripts_root(sd_scripts_root)
    if resolved_root not in sys.path:
        sys.path.insert(0, resolved_root)

    try:
        anima_utils = importlib.import_module("library.anima_utils")
        qwen_image_autoencoder_kl = importlib.import_module("library.qwen_image_autoencoder_kl")
    except ImportError as exc:
        raise ImportError(
            f"Failed to import Anima runtime modules from sd-scripts at {resolved_root}."
        ) from exc

    return anima_utils, qwen_image_autoencoder_kl


def _load_optional_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a state dict from a safetensors or PyTorch checkpoint."""
    if path.endswith(".safetensors"):
        return load_file(path, device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _get_anima_param_groups(
    transformer: torch.nn.Module,
    base_lr: float,
    self_attn_lr: Optional[float] = None,
    cross_attn_lr: Optional[float] = None,
    mlp_lr: Optional[float] = None,
    mod_lr: Optional[float] = None,
    llm_adapter_lr: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Build Anima full-finetune optimizer groups with per-submodule LRs."""
    self_attn_lr = base_lr if self_attn_lr is None else self_attn_lr
    cross_attn_lr = base_lr if cross_attn_lr is None else cross_attn_lr
    mlp_lr = base_lr if mlp_lr is None else mlp_lr
    mod_lr = base_lr if mod_lr is None else mod_lr
    llm_adapter_lr = base_lr if llm_adapter_lr is None else llm_adapter_lr

    buckets: Dict[str, List[torch.nn.Parameter]] = {
        "base": [],
        "self_attn": [],
        "cross_attn": [],
        "mlp": [],
        "mod": [],
        "llm_adapter": [],
    }
    lr_map = {
        "base": base_lr,
        "self_attn": self_attn_lr,
        "cross_attn": cross_attn_lr,
        "mlp": mlp_lr,
        "mod": mod_lr,
        "llm_adapter": llm_adapter_lr,
    }

    for name, param in transformer.named_parameters():
        if not param.requires_grad:
            continue
        if "llm_adapter" in name:
            buckets["llm_adapter"].append(param)
        elif ".self_attn" in name:
            buckets["self_attn"].append(param)
        elif ".cross_attn" in name:
            buckets["cross_attn"].append(param)
        elif ".mlp" in name:
            buckets["mlp"].append(param)
        elif ".adaln_modulation" in name:
            buckets["mod"].append(param)
        else:
            buckets["base"].append(param)

    param_groups = []
    for group_name, params in buckets.items():
        lr = lr_map[group_name]
        if lr == 0:
            for param in params:
                param.requires_grad_(False)
            logger.info(f"Frozen Anima parameter group `{group_name}` ({len(params)} tensors).")
            continue
        if params:
            param_groups.append({"params": params, "lr": lr})
            logger.info(f"Prepared Anima parameter group `{group_name}` with {len(params)} tensors at lr={lr}.")

    return param_groups


class AnimaPipeline(DiffusionPipeline):
    """Minimal pseudo-pipeline wrapping Anima runtime components."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        text_encoder: torch.nn.Module,
        vae: torch.nn.Module,
        tokenizer: Any,
        t5_tokenizer: Any,
        scheduler: DiffusersFlowMatchEulerDiscreteScheduler,
    ) -> None:
        super().__init__()
        self.register_modules(
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            t5_tokenizer=t5_tokenizer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = getattr(vae, "spatial_compression_ratio", 8)


@dataclass
class AnimaSample(T2ISample):
    """Per-sample output container for Anima text-to-image generation."""

    _shared_fields: ClassVar[frozenset[str]] = frozenset({})
    prompt_embeds_mask: Optional[torch.Tensor] = None
    t5_input_ids: Optional[torch.Tensor] = None
    t5_attn_mask: Optional[torch.Tensor] = None
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None
    negative_t5_input_ids: Optional[torch.Tensor] = None
    negative_t5_attn_mask: Optional[torch.Tensor] = None


class AnimaAdapter(BaseAdapter):
    """Flow-Factory adapter for the Anima preview text-to-image model."""

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: AnimaPipeline
        self.scheduler: FlowMatchEulerDiscreteSDEScheduler

    @property
    def tokenizer(self) -> Any:
        """Return the Qwen3 tokenizer used for prompt id bookkeeping."""
        return self.pipeline.tokenizer

    @property
    def default_target_modules(self) -> List[str]:
        """Default LoRA targets matching the reference Anima LoRA recipe."""
        return ["q_proj", "k_proj", "v_proj", "output_proj", "layer1", "layer2"]

    def _resolve_default_lora_exclude_modules(
        self,
        target_modules: Union[str, List[str]],
        component_name: str,
    ) -> Optional[str]:
        """Return the PEFT exclusion pattern for the default Anima LoRA recipe.

        The default Anima LoRA recipe should target only the top-level DiT blocks. PEFT
        list-based matching is suffix-based, so names like ``q_proj`` would also match
        ``llm_adapter.*.q_proj`` unless that subtree is explicitly excluded.

        Args:
            target_modules: Raw target module configuration passed to ``apply_lora``.
            component_name: Component currently receiving LoRA adapters.

        Returns:
            Regex pattern for ``exclude_modules`` when the default Anima LoRA recipe
            should exclude ``llm_adapter``. Returns ``None`` for all other cases.
        """
        if component_name != "transformer":
            return None

        raw_modules = [target_modules] if isinstance(target_modules, str) else target_modules
        default_aliases = {"default", f"{component_name}.default"}
        requested_default = any(module in default_aliases for module in raw_modules)
        requested_llm_adapter = any(
            module not in default_aliases and "llm_adapter" in module for module in raw_modules
        )

        if requested_default and not requested_llm_adapter:
            return ANIMA_DEFAULT_LORA_EXCLUDE_PATTERN
        return None

    def _create_lora_config(
        self,
        target_modules: Union[str, List[str]],
        exclude_modules: Optional[str] = None,
    ) -> LoraConfig:
        """Create a LoRA config for the current Anima adapter settings.

        Args:
            target_modules: PEFT target module spec for the current component.
            exclude_modules: Optional PEFT exclusion regex.

        Returns:
            LoRA config initialized from the current model arguments.
        """
        return LoraConfig(
            r=self.model_args.lora_rank,
            lora_alpha=self.model_args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            exclude_modules=exclude_modules,
        )

    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = "transformer",
        overwrite: bool = False,
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        """Apply LoRA adapters while keeping the default Anima recipe off ``llm_adapter``.

        Args:
            target_modules: Module patterns with optional component prefix.
            components: Component or components to receive LoRA adapters.
            overwrite: Whether to replace an existing ``default`` adapter.

        Returns:
            The wrapped PEFT model, or a mapping when multiple components are updated.
        """
        if isinstance(components, str):
            components = [components]

        component_modules = self._parse_target_modules(target_modules, components)
        results = {}
        for comp in components:
            modules = component_modules.get(comp)

            if modules == "default":
                modules = self.default_target_modules
            elif modules == "all":
                modules = "all"
            elif not modules:
                logger.warning(f"No target modules for {comp}, skipping LoRA")
                continue

            exclude_modules = self._resolve_default_lora_exclude_modules(target_modules, comp)
            lora_config = self._create_lora_config(modules, exclude_modules=exclude_modules)
            model_component = self.get_component(comp)

            if isinstance(model_component, PeftModel):
                has_default = "default" in model_component.peft_config
                if has_default and not overwrite:
                    logger.info(
                        f"Component {comp} already has 'default' adapter. "
                        "Skipping initialization but enabling gradients."
                    )
                    for name, param in model_component.named_parameters():
                        if any(key in name for key in self.lora_keys):
                            param.requires_grad = True
                    results[comp] = model_component
                    continue

                if has_default and overwrite:
                    logger.info(f"Overwriting existing 'default' adapter for {comp}")
                    model_component.delete_adapter("default")

                model_component.add_adapter("default", lora_config)
            else:
                model_component = get_peft_model(model_component, lora_config)
                self.set_component(comp, model_component)

            model_component.set_adapter("default")
            results[comp] = model_component

            if exclude_modules is None:
                logger.info(f"Applied LoRA to {comp} with modules: {modules}")
            else:
                logger.info(
                    f"Applied LoRA to {comp} with modules: {modules} "
                    f"(exclude_modules={exclude_modules})"
                )

        if not results:
            logger.warning("No LoRA adapters were applied")
            return {}

        return next(iter(results.values())) if len(results) == 1 else results

    def load_pipeline(self) -> AnimaPipeline:
        """Load the Anima runtime components through sd-scripts helpers."""
        if self.model_args.qwen3 is None:
            raise ValueError("Anima requires `model.qwen3` to point to the Qwen3 text encoder checkpoint.")
        if self.model_args.vae is None:
            raise ValueError("Anima requires `model.vae` to point to the Qwen-Image VAE checkpoint.")

        anima_utils, qwen_image_autoencoder_kl = _load_anima_runtime_modules(self.model_args.sd_scripts_root)
        self._anima_utils = anima_utils
        self._qwen_image_autoencoder_kl = qwen_image_autoencoder_kl

        attn_mode = self.model_args.attn_mode or "torch"
        if attn_mode == "sdpa":
            attn_mode = "torch"
        if attn_mode == "sageattn" and self.model_args.finetune_type != "lora":
            raise ValueError("Anima training does not support `attn_mode=sageattn`.")

        load_dtype = self._inference_dtype
        text_encoder, tokenizer = anima_utils.load_qwen3_text_encoder(
            self.model_args.qwen3,
            dtype=load_dtype,
            device="cpu",
        )
        t5_tokenizer = anima_utils.load_t5_tokenizer(self.model_args.t5_tokenizer_path)
        vae = qwen_image_autoencoder_kl.load_vae(
            self.model_args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=self.model_args.vae_chunk_size,
            disable_cache=self.model_args.vae_disable_cache,
        )
        vae.to(dtype=load_dtype)

        transformer = anima_utils.load_anima_model(
            device="cpu",
            dit_path=self.model_args.model_name_or_path,
            attn_mode=attn_mode,
            split_attn=self.model_args.split_attn,
            loading_device="cpu",
            dit_weight_dtype=load_dtype,
        )
        if self.model_args.llm_adapter_path is not None:
            llm_adapter_state_dict = _load_optional_state_dict(self.model_args.llm_adapter_path)
            stripped_state_dict = {
                key[len("llm_adapter.") :] if key.startswith("llm_adapter.") else key: value
                for key, value in llm_adapter_state_dict.items()
            }
            missing, unexpected = transformer.llm_adapter.load_state_dict(stripped_state_dict, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "Failed to load `model.llm_adapter_path` into the Anima transformer. "
                    f"Missing keys: {missing[:5]}, unexpected keys: {unexpected[:5]}"
                )

        scheduler = DiffusersFlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=self.model_args.discrete_flow_shift,
        )
        return AnimaPipeline(
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            t5_tokenizer=t5_tokenizer,
            scheduler=scheduler,
        )

    def get_optimizer_param_groups(self) -> List[Union[torch.nn.Parameter, Dict[str, Any]]]:
        """Use Anima-specific full-finetune parameter groups when requested."""
        if (
            self.model_args.finetune_type == "full"
            and self.model_args.target_components == ["transformer"]
            and self.target_module_map.get("transformer") == "all"
        ):
            return _get_anima_param_groups(
                self.transformer,
                base_lr=self.training_args.learning_rate,
                self_attn_lr=self.training_args.self_attn_lr,
                cross_attn_lr=self.training_args.cross_attn_lr,
                mlp_lr=self.training_args.mlp_lr,
                mod_lr=self.training_args.mod_lr,
                llm_adapter_lr=self.training_args.llm_adapter_lr,
            )
        return super().get_optimizer_param_groups()

    def _encode_prompt_batch(
        self,
        prompt: List[str],
        device: torch.device,
        dtype: torch.dtype,
        qwen3_max_length: int,
        t5_max_length: int,
    ) -> Dict[str, torch.Tensor]:
        """Encode prompt text into Qwen3 hidden states and T5 token ids."""
        qwen3_encoding = self.tokenizer.batch_encode_plus(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=qwen3_max_length,
        )
        prompt_ids = qwen3_encoding["input_ids"].to(device)
        prompt_embeds_mask = qwen3_encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self.text_encoder(input_ids=prompt_ids, attention_mask=prompt_embeds_mask)
        prompt_embeds = outputs.last_hidden_state.to(device=device, dtype=dtype)
        prompt_embeds = prompt_embeds * prompt_embeds_mask.unsqueeze(-1).to(dtype=prompt_embeds.dtype)

        t5_encoding = self.pipeline.t5_tokenizer.batch_encode_plus(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=t5_max_length,
        )
        t5_input_ids = t5_encoding["input_ids"].to(device=device, dtype=torch.long)
        t5_attn_mask = t5_encoding["attention_mask"].to(device)

        return {
            "prompt_ids": prompt_ids,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "t5_input_ids": t5_input_ids,
            "t5_attn_mask": t5_attn_mask,
        }

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qwen3_max_length: Optional[int] = None,
        t5_max_length: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Encode prompt text using the Anima Qwen3 encoder and T5 tokenizer."""
        del kwargs
        device = device or self.text_encoder.device
        dtype = dtype or self.text_encoder.dtype
        qwen3_max_length = qwen3_max_length or self.model_args.qwen3_max_token_length
        t5_max_length = t5_max_length or self.model_args.t5_max_token_length

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is None:
            prompt = [""]

        results = self._encode_prompt_batch(
            prompt=prompt,
            device=device,
            dtype=dtype,
            qwen3_max_length=qwen3_max_length,
            t5_max_length=t5_max_length,
        )

        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * len(prompt)
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            if len(negative_prompt) == 1 and len(prompt) > 1:
                negative_prompt = negative_prompt * len(prompt)
            if len(negative_prompt) != len(prompt):
                raise ValueError("The number of negative prompts must match the number of prompts.")

            negative_results = self._encode_prompt_batch(
                prompt=negative_prompt,
                device=device,
                dtype=dtype,
                qwen3_max_length=qwen3_max_length,
                t5_max_length=t5_max_length,
            )
            results.update(
                {
                    "negative_prompt_ids": negative_results["prompt_ids"],
                    "negative_prompt_embeds": negative_results["prompt_embeds"],
                    "negative_prompt_embeds_mask": negative_results["prompt_embeds_mask"],
                    "negative_t5_input_ids": negative_results["t5_input_ids"],
                    "negative_t5_attn_mask": negative_results["t5_attn_mask"],
                }
            )

        return results

    def encode_image(self, images: Any, **kwargs) -> Optional[Dict[str, Any]]:
        """Anima text-to-image does not use image conditioning."""
        del images, kwargs
        return None

    def encode_video(self, videos: Any, **kwargs) -> Optional[Dict[str, Any]]:
        """Anima text-to-image does not use video conditioning."""
        del videos, kwargs
        return None

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: Literal["pil", "pt", "np"] = "pil",
        **kwargs,
    ) -> Union[List[Image.Image], torch.Tensor, np.ndarray]:
        """Decode Qwen-Image latents back to pixel space."""
        del kwargs
        pixels = self.vae.decode_to_pixels(latents.to(device=self.vae.device, dtype=self.vae.dtype))
        pixels = torch.clamp((pixels.float() + 1.0) / 2.0, min=0.0, max=1.0)

        if output_type == "pt":
            return pixels

        images_np = (pixels.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        if output_type == "np":
            return images_np
        return [Image.fromarray(image) for image in images_np]

    @staticmethod
    def _as_batch_tensor(
        value: Union[float, torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Broadcast a scalar or vector timestep-like input to batch shape."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=device)
        value = value.to(device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch_size)
        elif value.ndim == 1 and value.shape[0] == 1:
            value = value.expand(batch_size)
        elif value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(
                f"Expected a scalar or batch-sized 1D tensor, got shape {tuple(value.shape)}."
            )
        return value

    def forward(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        t5_input_ids: torch.Tensor,
        t5_attn_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_t5_input_ids: Optional[torch.Tensor] = None,
        negative_t5_attn_mask: Optional[torch.Tensor] = None,
        guidance_scale: float = 4.0,
        t_next: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "noise_pred",
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
        ],
        **kwargs,
    ) -> FlowMatchEulerDiscreteSDESchedulerOutput:
        """Run a single Anima denoising step and optional scheduler transition."""
        del kwargs
        batch_size = latents.shape[0]
        model_dtype = self.transformer.dtype
        model_device = next(self.transformer.parameters()).device

        timestep = self._as_batch_tensor(t, batch_size, model_device, model_dtype) / 1000.0
        model_latents = latents.to(device=model_device, dtype=model_dtype).unsqueeze(2)
        prompt_embeds = prompt_embeds.to(device=model_device, dtype=model_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(device=model_device)
        t5_input_ids = t5_input_ids.to(device=model_device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(device=model_device)

        padding_mask = torch.zeros(
            batch_size,
            1,
            latents.shape[-2],
            latents.shape[-1],
            dtype=model_dtype,
            device=model_device,
        )

        noise_pred = self.transformer(
            model_latents,
            timestep,
            prompt_embeds,
            padding_mask=padding_mask,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=prompt_embeds_mask,
        ).squeeze(2)

        do_classifier_free_guidance = (
            guidance_scale > 1.0
            and negative_prompt_embeds is not None
            and negative_prompt_embeds_mask is not None
            and negative_t5_input_ids is not None
            and negative_t5_attn_mask is not None
        )
        if do_classifier_free_guidance:
            neg_noise_pred = self.transformer(
                model_latents,
                timestep,
                negative_prompt_embeds.to(device=model_device, dtype=model_dtype),
                padding_mask=padding_mask,
                target_input_ids=negative_t5_input_ids.to(device=model_device, dtype=torch.long),
                target_attention_mask=negative_t5_attn_mask.to(device=model_device),
                source_attention_mask=negative_prompt_embeds_mask.to(device=model_device),
            ).squeeze(2)
            noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

        return self.scheduler.step(
            noise_pred=noise_pred,
            timestep=t,
            latents=latents,
            timestep_next=t_next,
            next_latents=next_latents,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
            noise_level=noise_level,
        )

    @torch.no_grad()
    def inference(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        t5_input_ids: Optional[torch.Tensor] = None,
        t5_attn_mask: Optional[torch.Tensor] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_t5_input_ids: Optional[torch.Tensor] = None,
        negative_t5_attn_mask: Optional[torch.Tensor] = None,
        compute_log_prob: bool = True,
        extra_call_back_kwargs: List[str] = [],
        trajectory_indices: TrajectoryIndicesType = "all",
        **kwargs,
    ) -> List[AnimaSample]:
        """Generate Anima rollouts while retaining trajectory tensors for RL training."""
        del kwargs
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Anima requires height/width divisible by 16, got {height}x{width}.")

        device = self.device
        dtype = self.transformer.dtype

        if prompt_embeds is None or prompt_embeds_mask is None or t5_input_ids is None or t5_attn_mask is None:
            encoded = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_scale > 1.0,
                device=device,
                dtype=dtype,
            )
            prompt_ids = encoded["prompt_ids"]
            prompt_embeds = encoded["prompt_embeds"]
            prompt_embeds_mask = encoded["prompt_embeds_mask"]
            t5_input_ids = encoded["t5_input_ids"]
            t5_attn_mask = encoded["t5_attn_mask"]
            negative_prompt_ids = encoded.get("negative_prompt_ids")
            negative_prompt_embeds = encoded.get("negative_prompt_embeds")
            negative_prompt_embeds_mask = encoded.get("negative_prompt_embeds_mask")
            negative_t5_input_ids = encoded.get("negative_t5_input_ids")
            negative_t5_attn_mask = encoded.get("negative_t5_attn_mask")
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            prompt_embeds_mask = prompt_embeds_mask.to(device=device)
            t5_input_ids = t5_input_ids.to(device=device, dtype=torch.long)
            t5_attn_mask = t5_attn_mask.to(device=device)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds_mask is not None:
                negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device=device)
            if negative_t5_input_ids is not None:
                negative_t5_input_ids = negative_t5_input_ids.to(device=device, dtype=torch.long)
            if negative_t5_attn_mask is not None:
                negative_t5_attn_mask = negative_t5_attn_mask.to(device=device)

        batch_size = prompt_embeds.shape[0]
        latent_height = height // self.pipeline.vae_scale_factor
        latent_width = width // self.pipeline.vae_scale_factor
        latents = randn_tensor(
            (batch_size, 16, latent_height, latent_width),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        self.scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latents = self.cast_latents(latents, default_dtype=dtype)
        latent_collector.collect(latents, step_idx=0)
        log_prob_collector = None
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        for i, current_t in enumerate(timesteps):
            current_noise_level = self.scheduler.get_noise_level_for_timestep(current_t)
            next_t = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
            return_kwargs = list(set(["next_latents", "log_prob", "noise_pred"] + extra_call_back_kwargs))
            current_compute_log_prob = compute_log_prob and current_noise_level > 0

            output = self.forward(
                t=current_t,
                t_next=next_t,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                t5_input_ids=t5_input_ids,
                t5_attn_mask=t5_attn_mask,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                negative_t5_input_ids=negative_t5_input_ids,
                negative_t5_attn_mask=negative_t5_attn_mask,
                guidance_scale=guidance_scale,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=return_kwargs,
                noise_level=current_noise_level,
            )

            latents = self.cast_latents(output.next_latents, default_dtype=dtype)
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob and log_prob_collector is not None:
                log_prob_collector.collect(output.log_prob, i)

            callback_collector.collect_step(
                step_idx=i,
                output=output,
                keys=extra_call_back_kwargs,
                capturable={"noise_level": current_noise_level},
            )

        decoded_images = self.decode_latents(latents, output_type="pt")
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        negative_prompt_list = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        extra_call_back_res = callback_collector.get_result()
        callback_index_map = callback_collector.get_index_map()
        all_latents = latent_collector.get_result()
        latent_index_map = latent_collector.get_index_map()
        all_log_probs = log_prob_collector.get_result() if log_prob_collector is not None else None
        log_prob_index_map = log_prob_collector.get_index_map() if log_prob_collector is not None else None

        return [
            AnimaSample(
                timesteps=timesteps,
                all_latents=(
                    torch.stack([latent[b] for latent in all_latents], dim=0)
                    if all_latents is not None
                    else None
                ),
                log_probs=(
                    torch.stack([log_prob[b] for log_prob in all_log_probs], dim=0)
                    if all_log_probs is not None
                    else None
                ),
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
                height=height,
                width=width,
                image=decoded_images[b],
                prompt=prompt_list[b] if prompt_list is not None else None,
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b],
                prompt_embeds_mask=prompt_embeds_mask[b],
                t5_input_ids=t5_input_ids[b],
                t5_attn_mask=t5_attn_mask[b],
                negative_prompt=negative_prompt_list[b] if negative_prompt_list is not None else None,
                negative_prompt_ids=negative_prompt_ids[b] if negative_prompt_ids is not None else None,
                negative_prompt_embeds=(
                    negative_prompt_embeds[b] if negative_prompt_embeds is not None else None
                ),
                negative_prompt_embeds_mask=(
                    negative_prompt_embeds_mask[b] if negative_prompt_embeds_mask is not None else None
                ),
                negative_t5_input_ids=(
                    negative_t5_input_ids[b] if negative_t5_input_ids is not None else None
                ),
                negative_t5_attn_mask=negative_t5_attn_mask[b] if negative_t5_attn_mask is not None else None,
                extra_kwargs={
                    **{key: value[b] for key, value in extra_call_back_res.items()},
                    "callback_index_map": callback_index_map,
                },
            )
            for b in range(batch_size)
        ]
