# Anima Support Notes

## Overview

This document records the implementation and validation status of `model_type: anima` in Flow-Factory as of April 11, 2026.

The integration targets the local Anima runtime layout used in this workspace:

- DiT checkpoint: `models/animaOfficial_preview2.safetensors`
- Qwen3 text encoder: `models/qwen_3_06b_base.safetensors`
- Qwen-Image VAE: `models/qwen_image_vae.safetensors`
- Tokenizers: `tokenizer/qwen3_06b`, `tokenizer/t5_old`
- External runtime reference: `~/sd-scripts`

Flow-Factory does not depend on a diffusers-native Anima pipeline. Instead, it wraps the Anima runtime exposed by `sd-scripts` through a custom adapter and a lightweight pseudo-pipeline.

## Implemented Changes

### Adapter and runtime

The Anima integration is implemented in `src/flow_factory/models/anima/anima.py` and registered under `model_type: anima`.

Key behaviors:

- Loads the Anima DiT, Qwen3 text encoder, T5 tokenizer, and Qwen-Image VAE.
- Encodes prompts into:
  - Qwen3 hidden states
  - Qwen3 attention mask
  - T5 token ids
  - T5 attention mask
- Uses 5D model inputs for the DiT path: `(B, C, 1, H, W)`.
- Normalizes timesteps to `[0, 1]` before calling the Anima transformer.
- Supports classifier-free guidance with separate negative prompt encodings.
- Reuses Flow-Factory's SDE scheduler and trajectory collection path.

### Configuration surface

The following Anima-specific model fields were added to `ModelArguments`:

- `qwen3`
- `vae`
- `llm_adapter_path`
- `t5_tokenizer_path`
- `qwen3_max_token_length`
- `t5_max_token_length`
- `sd_scripts_root`
- `attn_mode`
- `split_attn`
- `vae_chunk_size`
- `vae_disable_cache`
- `discrete_flow_shift`

The following Anima-specific full-finetune learning-rate overrides were added to `TrainingArguments`:

- `self_attn_lr`
- `cross_attn_lr`
- `mlp_lr`
- `mod_lr`
- `llm_adapter_lr`

### Full-finetune parameter grouping

Anima full finetuning uses six optimizer groups:

1. `base`
2. `self_attn`
3. `cross_attn`
4. `mlp`
5. `mod`
6. `llm_adapter`

This mirrors the reference training layout from `sd-scripts` closely enough for practical use inside Flow-Factory.

## BF16 Guidance

The Anima examples are configured to prefer BF16 end to end:

- Top-level `mixed_precision: "bf16"`
- `model.master_weight_dtype: "bf16"`
- `train.latent_storage_dtype: "bf16"`

The `latent_storage_dtype` override matters for Anima because the rollout path stores trajectory latents between sampling and optimization. Using BF16 avoids unintentional downcasts during BF16 training and inference.

## Example Config Updates

Updated example files:

- `examples/grpo/lora/anima.yaml`
- `examples/grpo/full/anima.yaml`

Notable changes:

- `mixed_precision` switched from FP16 to BF16
- `master_weight_dtype` switched from FP16 to BF16
- `sd_scripts_root` set to `~/sd-scripts`
- `train.latent_storage_dtype` set to `bf16`
- Full finetune example now uses `target_modules: "all"`

## Validation Performed

### 1. Adapter load smoke

Date: April 11, 2026

Validated:

- `Arguments.load_from_yaml("examples/grpo/lora/anima.yaml")`
- `load_model(...)` successfully returns `AnimaAdapter`
- LoRA attaches to the expected default target modules

Observed LoRA target list:

- `q_proj`
- `k_proj`
- `v_proj`
- `output_proj`
- `layer1`
- `layer2`

Default attachment scope:

- top-level `blocks.*` modules only
- `llm_adapter.*` is excluded from the default LoRA recipe

### 2. BF16 inference smoke

Date: April 11, 2026

Validated with:

- 1 prompt
- 1 negative prompt
- `height=64`
- `width=64`
- `num_inference_steps=2`
- `guidance_scale=2.0`
- `compute_log_prob=True`

Observed results:

- `adapter.on_load(cuda)` correctly moved the LoRA-wrapped transformer to `cuda:0`
- output sample count: `1`
- decoded image shape: `(3, 64, 64)`
- trajectory latent shape: `(3, 16, 8, 8)`
- `log_probs` shape: `(1,)`
- stored latent dtype: `torch.bfloat16`

### 3. GRPO LoRA training-flow smoke

Date: April 11, 2026

Validated with a minimized single-rank GRPO loop:

- dataset limited to 1 prompt
- `group_size=2`
- `num_inference_steps=2`
- `resolution=64`
- one synthetic groupwise reward model (`MyGroupwiseRewardModel`)

Executed path:

1. `trainer.sample()`
2. `trainer.prepare_feedback(samples)`
3. `trainer.optimize(samples)`

Observed results:

- generated sample count: `2`
- generated image shape: `(3, 64, 64)`
- reward mean/std: `0.5 / 0.5`
- advantage range: `[-1, 1]`
- non-zero gradient norm observed on the first optimization step: `7.4845`
- optimizer steps completed: `2`
- tracked LoRA parameter:
  - `base_model.model.blocks.0.self_attn.q_proj.lora_B.default.weight`
- tracked parameter norm changed from `0.0` to `0.12700527906417847`
- tracked parameter absolute delta sum: `32.485313415527344`

This confirms that the LoRA training path is not only executable, but also updates trainable Anima parameters.

### 4. Full-finetune instantiation smoke

Date: April 11, 2026

Validated with `examples/grpo/full/anima.yaml`:

- adapter instantiation succeeds
- all transformer parameters are unfrozen
- the Anima-specific optimizer grouping path returns 6 parameter groups

Observed group learning rates:

- `[1e-05, 1e-05, 1e-05, 1e-05, 5e-06, 5e-06]`

Observed trainable parameter count:

- `2091068928`

### 5. Attention backend inference compatibility

Date: April 11, 2026

Validated with the same minimal BF16 inference setup used above:

- `height=64`
- `width=64`
- `num_inference_steps=2`
- `compute_log_prob=True`

Observed environment:

- `flash_attn==2.8.3`
- `xformers==0.0.35`
- `sageattention` import available

Observed results:

- `attn_mode=flash`, `split_attn=false`: success
- `attn_mode=xformers`, `split_attn=true`: success
- `attn_mode=sageattn`, `split_attn=false`: success
- all three runs returned 1 decoded image with shape `(3, 64, 64)` and `log_probs` shape `(1,)`

### 6. Default reward GRPO LoRA smoke

Date: April 11, 2026

Validated with a minimized single-rank GRPO loop using the default reward stack from the Anima LoRA example:

- `pickscore`
- `clip`
- dataset limited to 1 prompt
- `group_size=4`
- `num_inference_steps=2`
- `resolution=64`

Executed path:

1. `trainer.sample()`
2. `trainer.prepare_feedback(samples)`
3. `trainer.optimize(samples)`

Observed results:

- generated sample count: `4`
- generated image shape: `(3, 64, 64)`
- PickScore mean/std: `0.5869439840316772 / 0.007971200160682201`
- CLIP text-alignment mean/std: `0.158203125 / 0.016673214733600616`
- normalized advantages:
  - `[-0.4706340432167053, 1.3029879331588745, -1.347391128540039, 0.5150371789932251]`
- optimization logs remained finite across 4 steps:
  - step 0 policy loss: `-1.3030`
  - step 1 policy loss: `0.4706`
  - step 2 policy loss: `-0.0001`
  - step 3 policy loss: `1.3473`
- tracked LoRA parameter absolute delta sum: `48.64039611816406`

This confirms that the default reward configuration produces non-constant reward variance, valid normalized advantages, and actual LoRA parameter updates.

## Notes and Current Limits

- The adapter implementation is shared across GRPO, NFT, and AWM through the common `BaseAdapter` interface.
- Only the GRPO LoRA training flow was executed end to end in this session.
- `sageattn` was only validated on the inference path in this workspace.
- Advanced `sd-scripts` memory features such as block swapping and custom offload behavior were not ported.
- The local environment initially missed the `datasets` package, but the dependency is already declared in `pyproject.toml`.
