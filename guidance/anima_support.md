# Anima Support Notes

## Overview

This document records the implementation and validation status of `model_type: anima` in Flow-Factory as of April 26, 2026.

The integration targets the local Anima runtime layout used in this workspace:

- GRPO/NFT DiT checkpoint: `models/animaOfficial_preview2.safetensors`
- AWM scaling DiT checkpoint: `models/animaOfficial_preview3Base.safetensors`
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
- `model.trainable_parameters_dtype: "bf16"`
- `train.latent_storage_dtype: "bf16"`

The `latent_storage_dtype` override matters for Anima because the rollout path stores trajectory latents between sampling and optimization. Using BF16 avoids unintentional downcasts during BF16 training and inference.

## Example Config Updates

Updated example files:

- `examples/grpo/lora/anima/default.yaml`
- `examples/grpo/full/anima/default.yaml`
- `examples/nft/lora/anima/default.yaml`
- `examples/awm/lora/anima/default.yaml`
- `examples/awm/lora/anima/single_gpu_smoke.yaml`
- `examples/awm/lora/anima/multi_gpu_deepspeed_zero2_smoke.yaml`
- `examples/awm/lora/anima/multi_gpu_fsdp2_smoke.yaml`
- `examples/grpo/lora/anima/anime_custom.yaml`

Notable changes:

- `mixed_precision` switched from FP16 to BF16
- `trainable_parameters_dtype` switched from FP16 to BF16
- `sd_scripts_root` set to `~/sd-scripts`
- `train.latent_storage_dtype` set to `bf16`
- Full finetune example now uses `target_modules: "all"`
- The AWM custom route uses `dataset/anime_custom_single_gpu_eval16`, Aesthetic Shadow, and WD prompt-hash reference similarity.

## Custom AWM Scaling Route

The current custom anime scaling route is documented in `anima_scaling_plan.md`.

Primary runtime config:

- `examples/awm/lora/anima/default.yaml`

Dataset split:

- Source: `dataset/anime_custom_single_gpu_eval16/anime4k.jsonl`
- Source records: `4049`
- Train split: `dataset/anime_custom_single_gpu_eval16/train.jsonl`, `4033` records
- Test split: `dataset/anime_custom_single_gpu_eval16/test.jsonl`, `16` records
- Split rule: random sample 16 test records with seed `42`; use the remaining records for train

Reward stack:

- `aesthetic_shadow`: remote Aesthetic Shadow pointwise reward at `http://127.0.0.1:18081`
- `wd_reference_similarity`: remote WD EVA02 prompt-hash similarity reward at `http://127.0.0.1:18082`
- WD reference embedding cache: `dataset/anime_custom_single_gpu_eval16/wd_prompt_hash_cache.pt`
- WD cache coverage: `4049` prompt hashes, matching the full source JSONL

Operational notes:

- The WD cache does not need to be rebuilt after the train/test split because it covers all source prompts.
- `examples/awm/lora/anima/default.yaml` currently omits `train.max_epochs`; this means training runs until interrupted. Set a finite value before unattended formal runs.

## Reproducible Smoke Runner

The following utility script was added to keep the Anima LoRA smoke checks reproducible:

- `scripts/validate_anima_lora_smoke.py`

It executes a minimized single-rank training flow:

1. `trainer.sample()`
2. `trainer.prepare_feedback(samples)`
3. `trainer.optimize(samples)`

The script prints a JSON summary with:

- sample count
- decoded image shape
- reward statistics
- advantage statistics
- optimizer step count
- the LoRA parameter with the largest absolute update

## Attention Benchmark Utility

The following utility script was added for single-backend attention smoke benchmarks:

- `scripts/benchmark_anima_attention.py`

It runs one minimized single-rank flow and reports:

- setup time
- sampling, reward, and optimization time
- CUDA peak allocation for each phase
- sample and train throughput
- LoRA parameter update summary

## Validation Performed

### 1. Adapter load smoke

Date: April 11, 2026

Validated:

- `Arguments.load_from_yaml("examples/grpo/lora/anima/default.yaml")`
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

Validated with `examples/grpo/full/anima/default.yaml`:

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

### 7. SageAttention GRPO LoRA training-flow smoke

Date: April 26, 2026

Validated with:

- `python scripts/validate_anima_lora_smoke.py --scenario grpo-sageattn`
- base config: `examples/grpo/lora/anima/default.yaml`
- runtime overrides:
  - `attn_mode=sageattn`
  - `split_attn=false`
  - dataset limited to 1 prompt
  - `group_size=2`
  - `resolution=64`
  - `num_inference_steps=2`
  - one synthetic groupwise reward model (`MyGroupwiseRewardModel`)

Observed results:

- generated sample count: `2`
- generated image shape: `(3, 64, 64)`
- reward mean/std: `0.5 / 0.5`
- advantage range: `[-1, 1]`
- ratio min/max/mean: `1.0 / 1.0 / 1.0`
- logged grad norm: `1.5340`
- optimizer steps completed: `1`
- changed LoRA parameter count: `112`
- tracked LoRA parameter:
  - `base_model.model.blocks.0.mlp.layer1.lora_B.default.weight`
- tracked parameter norm changed from `0.0` to `0.216796875`
- tracked parameter absolute delta sum: `157.0`

This confirms that the Anima LoRA training path remains executable when the runtime uses `attn_mode=sageattn`.

### 8. NFT LoRA training-flow smoke

Date: April 26, 2026

Validated with:

- `python scripts/validate_anima_lora_smoke.py --scenario nft`
- base config: `examples/nft/lora/anima/default.yaml`
- runtime overrides:
  - dataset limited to 1 prompt
  - `group_size=2`
  - `resolution=64`
  - `num_inference_steps=2`
  - one synthetic groupwise reward model (`MyGroupwiseRewardModel`)

Observed results:

- generated sample count: `2`
- generated image shape: `(3, 64, 64)`
- reward mean/std: `0.5 / 0.5`
- advantage range: `[-1, 1]`
- optimization logs remained finite:
  - policy loss: `4.0613`
  - grad norm: `46.9312`
- optimizer steps completed: `1`
- changed LoRA parameter count: `280`
- tracked LoRA parameter:
  - `base_model.model.blocks.8.mlp.layer1.lora_B.default.weight`
- tracked parameter norm changed from `0.0` to `0.072265625`
- tracked parameter absolute delta sum: `52.25`

This confirms that Anima `forward()` outputs are compatible with the decoupled NFT trainer path without adding trainer-specific model branches.

### 9. AWM LoRA training-flow smoke

Date: April 26, 2026

Validated with:

- `python scripts/validate_anima_lora_smoke.py --scenario awm`
- base config: `examples/awm/lora/anima/default.yaml`
- runtime overrides:
  - dataset limited to 1 prompt
  - `group_size=2`
  - `resolution=64`
  - `num_inference_steps=2`
  - one synthetic groupwise reward model (`MyGroupwiseRewardModel`)

Observed results:

- generated sample count: `2`
- generated image shape: `(3, 64, 64)`
- reward mean/std: `0.5 / 0.5`
- advantage range: `[-1, 1]`
- optimization logs remained finite:
  - policy loss: `0.0`
  - KL loss: `0.0`
  - EMA KL loss: `0.0`
  - grad norm: `53.4453`
- optimizer steps completed: `1`
- changed LoRA parameter count: `280`
- tracked LoRA parameter:
  - `base_model.model.blocks.10.mlp.layer1.lora_B.default.weight`
- tracked parameter norm changed from `0.0` to `0.01025390625`
- tracked parameter absolute delta sum: `5.25`

This confirms that Anima `forward()` also satisfies the AWM loss path, including the optional KL / EMA-KL branches when their coefficients are kept at the smoke defaults.

### 10. Remote reward service smoke

Date: April 26, 2026

Validated local CPU-mode reward services:

- Aesthetic Shadow:
  - command: `python scripts/reward_servers/shadow_server.py --host 127.0.0.1 --port 18081 --model-path /root/reward_models/aesthetic-shadow-v2-backup --device cpu --dtype float32`
  - `/health`: `200 {"status": "ok"}`
  - `/load`: success
  - `/compute` on a dummy image: returned one finite reward, `0.15886734426021576`
  - `/offload`: success
- WD prompt-hash reference similarity:
  - command: `python scripts/reward_servers/wd_prompt_hash_server.py --host 127.0.0.1 --port 18082 --model-path /root/reward_models/wd-eva02-large-tagger-v3 --cache-path dataset/anime_custom_single_gpu_eval16/wd_prompt_hash_cache.pt --device cpu --dtype float32`
  - `/health`: `200`, `reference_count=4049`
  - `/load`: success
  - `/compute` with a prompt from the split dataset and a dummy image: returned one finite reward, `0.042261261492967606`
  - `/offload`: success

This confirms that the two local reward servers can load, score, and offload using the same endpoints configured in `examples/awm/lora/anima/default.yaml`.

### 11. AWM real remote-reward training-flow smoke

Date: April 26, 2026

Validated with:

- base config: `examples/awm/lora/anima/default.yaml`
- runtime overrides:
  - dataset limited to 1 prompt from `dataset/anime_custom_single_gpu_eval16`
  - `group_size=2`
  - `resolution=64`
  - `num_inference_steps=2`
  - `num_train_timesteps=1`
  - Aesthetic Shadow remote reward
  - WD prompt-hash reference similarity remote reward

Observed results:

- generated sample count: `2`
- generated image shape: `(3, 64, 64)`
- `aesthetic_shadow` reward mean/std: `0.5521295070648193 / 0.12267257273197174`
- `wd_reference_similarity` reward mean/std: `0.1760590374469757 / 0.020329244434833527`
- advantage range: `[-1, 1]`
- ratio min/max/mean: `1.0 / 1.0 / 1.0`
- logged grad norm: `0.0173`
- optimizer steps completed: `1`

This validates the `sample -> remote rewards -> advantages -> optimize` path for the current AWM custom route.

### 12. Attention benchmark smoke

Date: April 26, 2026

Validated with:

- `CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_anima_attention.py --backend flash --batch-size 2 --resolution 64 --num-inference-steps 2`
- `CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_anima_attention.py --backend sageattn --batch-size 2 --resolution 64 --num-inference-steps 2`

Observed results:

- `flash`:
  - generated sample count: `2`
  - total step time: `1.498s`
  - train throughput: `1.335 img/s`
  - sample peak allocation: `4.399 GiB`
  - optimize peak allocation: `5.505 GiB`
  - changed LoRA parameter count: `280`
  - tracked parameter absolute delta sum: `157.0`
- `sageattn`:
  - generated sample count: `2`
  - total step time: `2.675s`
  - train throughput: `0.748 img/s`
  - sample peak allocation: `4.399 GiB`
  - optimize peak allocation: `4.733 GiB`
  - changed LoRA parameter count: `112`
  - tracked parameter absolute delta sum: `157.0`

These are smoke-level benchmark numbers only. They validate script behavior and backend compatibility, not maximum achievable throughput.

### 13. Dataset split validation

Date: April 26, 2026

Validated:

- `dataset/anime_custom_single_gpu_eval16/anime4k.jsonl`: `4049` valid JSONL records
- `dataset/anime_custom_single_gpu_eval16/train.jsonl`: `4033` valid JSONL records
- `dataset/anime_custom_single_gpu_eval16/test.jsonl`: `16` valid JSONL records
- train/test overlap: `0`
- train/test union matches the source record set
- `GeneralDataset(..., split="train", enable_preprocess=False)` loads `4033` records
- `GeneralDataset(..., split="test", enable_preprocess=False)` loads `16` records

### 14. Upstream merge AWM smoke and multi-GPU validation

Date: July 11, 2026

Validated on one NVIDIA RTX 4080 SUPER with:

- `ff-train examples/awm/lora/anima/single_gpu_smoke.yaml`
- Anima base, Qwen3, and Qwen-Image VAE checkpoints loaded from local safetensors
- Aesthetic Shadow served on CPU at `http://127.0.0.1:18081`
- one evaluation prompt plus `group_size=2` training samples at `resolution=256`
- two inference steps and one AWM training timestep

Observed results:

- generated sample count: `2`
- evaluation sample count: `1`
- evaluation Aesthetic Shadow reward: `0.5181`
- Aesthetic Shadow reward mean/std: `0.2570 / 0.1662`
- advantage range: `[-1, 1]`
- ratio min/max/mean: `1.0 / 1.0 / 1.0`
- logged grad norm: `0.0225`
- optimizer steps completed: `1`
- process exit code: `0`

The multi-GPU follow-up ran on two NVIDIA RTX 4080 SUPER GPUs with both supported
sharded backends.

DeepSpeed ZeRO-2 validation used
`examples/awm/lora/anima/multi_gpu_deepspeed_zero2_smoke.yaml` and the three remote
rewards on ports `18082`, `18084`, and `18085`:

- world size: `2`
- sampler: `group_contiguous`
- geometry: `unique_sample_num_per_epoch=2`, `group_size=2`,
  `per_device_batch_size=1`, `num_batches_per_epoch=2`,
  `gradient_accumulation_steps=2`
- epoch reward means: `1.3588`, `1.3759`
- epoch gradient norms: `0.0018`, `0.0038`
- optimizer steps completed: `2`
- changed LoRA tensors between checkpoint 0 and checkpoint 1: `280 / 560`
- total checkpoint absolute parameter delta: `93.5718`
- maximum single-element parameter delta: `2.0027e-5`
- process exit code: `0`

FSDP2 validation used `examples/awm/lora/anima/multi_gpu_fsdp2_smoke.yaml` and
Aesthetic Shadow on port `18081`:

- world size: `2`
- sampler: `group_contiguous`
- geometry: `unique_sample_num_per_epoch=4`, `group_size=2`,
  `per_device_batch_size=1`, `num_batches_per_epoch=4`,
  `gradient_accumulation_steps=4`
- generated sample count: `8`
- Aesthetic Shadow reward mean/std: `0.4691 / 0.2378`
- advantage range: `[-0.6817, 0.6817]`
- logged grad norm: `0.0049`
- optimizer steps completed: `1`
- process exit code: `0`

The adapter exposes Anima `Block` and `LLMAdapterTransformerBlock` as FSDP no-split
module classes, and the FSDP2 run confirmed that Accelerate can wrap and train the
repeated blocks without a collective mismatch or rank hang. All multi-process reward
services also completed load, compute, offload, and Ctrl-C shutdown without leaving
worker processes or GPU allocations behind.

## Notes and Current Limits

- The adapter implementation is shared across GRPO, NFT, and AWM through the common `BaseAdapter` interface.
- GRPO (`sageattn`), NFT, and AWM LoRA smoke flows were executed end to end on a single rank in this workspace.
- The current AWM custom route was also validated end to end with real remote Aesthetic Shadow and WD prompt-hash rewards.
- The WD prompt-hash embedding cache covers the full `anime4k.jsonl` source set and is independent of the train/test split.
- Advanced `sd-scripts` memory features such as block swapping and custom offload behavior were not ported.
- The local environment initially missed the `datasets` package, but the dependency is already declared in `pyproject.toml`.
