# Anima Custom Training Report

## Overview

- Date: `2026-04-11`
- Commit: `f293818`
- Operator: `Codex`
- Goal: Run Anima LoRA GRPO training on the custom anime Markdown prompt dataset with remote Aesthetic Shadow as the only active training reward.

## Current Status Update

- Date: `2026-04-26`
- Scope: update the custom Anima route from the earlier GRPO shadow-only report to the current AWM scaling plan with two remote pointwise rewards.
- Primary plan: `anima_scaling_plan.md`
- Primary config: `examples/awm/lora/anima.yaml`
- Current model checkpoint: `models/animaOfficial_preview3Base.safetensors`
- Current dataset directory: `dataset/anime_custom_single_gpu_eval16`
- Current validation host: one visible NVIDIA GeForce RTX 4080 with 32760 MiB VRAM
- Current active rewards:
  - `aesthetic_shadow`
  - `wd_reference_similarity`
- Current status: single-GPU smoke validation completed; formal long training has not been executed in this update.

The original April 11 sections below are retained as historical GRPO validation
records. The April 26 AWM route supersedes the old UnifiedReward-Flex route for
the current custom scaling plan.

## Environment

- Machine: `/root/Flow-Factory` workspace container
- GPU topology: `GPU0` for training, `GPU1` for reward services
- GPU model: `2 x NVIDIA GeForce RTX 4080 SUPER (32760 MiB)`
- Driver / CUDA: `580.105.08` / `13.0`
- Python environment: `Python 3.12.3`, `torch 2.10.0+cu128`, `accelerate 1.13.0`, `transformers 4.57.6`
- Proxy configuration: `ALL_PROXY=socks5://127.0.0.1:7891`; loopback reward calls explicitly bypass environment proxies in both the training-side client and the bridge.

## Inputs

### Model

- Base checkpoint: `models/animaOfficial_preview2.safetensors`
- Qwen3 text encoder: `models/qwen_3_06b_base.safetensors`
- VAE: `models/qwen_image_vae.safetensors`
- Tokenizers: `tokenizer/t5_old`
- `sd-scripts` root: `~/sd-scripts`

### Dataset

- Source dataset: local anime prompt JSON prepared outside Git
- Conversion entrypoint: `scripts/prepare_anime_custom_dataset.py`
- Conversion rule: preserve Markdown formatting and remove only the `Artist` section.
- Generated dataset artifacts are local-only and intentionally excluded from version control.

Current AWM scaling dataset:

- Source JSONL: `dataset/anime_custom_single_gpu_eval16/anime4k.jsonl`
- Source records: `4049`
- Train split: `dataset/anime_custom_single_gpu_eval16/train.jsonl`, `4033` records
- Test split: `dataset/anime_custom_single_gpu_eval16/test.jsonl`, `16` records
- Split rule: random 16-record test sample with seed `42`; all remaining records are used for train
- Train/test overlap: `0`
- WD prompt-hash cache: `dataset/anime_custom_single_gpu_eval16/wd_prompt_hash_cache.pt`
- WD cache coverage: `4049` prompt hashes, matching the full source JSONL

### Reward Services

- Aesthetic Shadow:
  - Endpoint: `http://127.0.0.1:18081`
  - Command: `CUDA_VISIBLE_DEVICES=0 python scripts/reward_servers/shadow_server.py --host 127.0.0.1 --port 18081 --device cuda --dtype bfloat16 --score-type prob_hq`
  - Score type: `P(hq)`
- WD prompt-hash reference similarity:
  - Endpoint: `http://127.0.0.1:18082`
  - Command: `python scripts/reward_servers/wd_prompt_hash_server.py --host 127.0.0.1 --port 18082 --model-path /root/reward_models/wd-eva02-large-tagger-v3 --cache-path dataset/anime_custom_single_gpu_eval16/wd_prompt_hash_cache.pt --device cpu --dtype float32`
  - Runtime behavior: hash each prompt, load the cached reference embedding, encode the generated image, and return cosine similarity
  - Health check result from April 26 validation: `reference_count=4049`
- UnifiedReward-Flex (historical validation only, not part of the current training route):
  - Historical validation backend used for the recorded smoke / short metrics:
    - vLLM endpoint: `http://127.0.0.1:18082/v1`
    - vLLM command: `CUDA_VISIBLE_DEVICES=1 TORCHDYNAMO_DISABLE=1 TOKENIZERS_PARALLELISM=false vllm serve /root/reward_models/UnifiedReward-Flex-qwen35-4b --host 127.0.0.1 --port 18082 --trust-remote-code --served-model-name UnifiedReward --gpu-memory-utilization 0.82 --tensor-parallel-size 1 --max-model-len 8192 --default-chat-template-kwargs '{"enable_thinking": false}' --enforce-eager --compilation-config '{"mode":0}' --skip-mm-profiling`
  - Current recommended faster backend as of `2026-04-12`:
    - vLLM endpoint: `http://127.0.0.1:8080/v1`
    - vLLM command:
      ```bash
      source .venv/bin/activate
      export VLLM_DISABLE_FLASHINFER_GDN_PREFILL=1
      export TOKENIZERS_PARALLELISM=false

      vllm serve /root/reward_models/UnifiedReward-Flex-qwen35-4b \
        --host 0.0.0.0 \
        --port 8080 \
        --trust-remote-code \
        --served-model-name UnifiedReward \
        --gpu-memory-utilization 0.95 \
        --mm-encoder-tp-mode data \
        --mm-processor-cache-type shm \
        --enable-prefix-caching \
        --tensor-parallel-size 1 \
        --default-chat-template-kwargs '{"enable_thinking": false}'
      ```
    - Health check: `http://127.0.0.1:8080/health` returned `200`
    - Note: this vLLM build reported `VLLM_DISABLE_FLASHINFER_GDN_PREFILL` as an unknown environment variable; the service still started normally, but the flag effect was not confirmed.
  - Bridge endpoint: `http://127.0.0.1:18083`
  - Bridge command: `python scripts/reward_servers/unifiedreward_flex_bridge.py --host 127.0.0.1 --port 18083 --api-base-url http://127.0.0.1:8080/v1 --model UnifiedReward --request-timeout 180 --max-parallel-pairs 2`
  - Bridge health check: `http://127.0.0.1:18083/health` returned `{"status": "ok"}`

## Training Configuration

- Baseline formal YAML: `examples/grpo/lora/anima_anime_custom.yaml`
- Primary shadow-only smoke YAML: `examples/grpo/lora/anima_anime_custom_shadow_smoke.yaml`
- Historical dual-reward smoke YAML: `examples/grpo/lora/anima_anime_custom_dual_smoke.yaml`
- 2-GPU short fallback YAML: `examples/grpo/lora/anima_anime_custom_short_2gpu.yaml`
- Formal target settings:
  - `num_processes: 4`
  - `group_size: 4`
  - `unique_sample_num_per_epoch: 64`
  - `max_epochs: 50`
- Effective completed short-run settings:
  - `num_processes: 1`
  - `group_size: 4`
  - `unique_sample_num_per_epoch: 8`
  - `max_epochs: 1`
- Current active route:
  - use `examples/awm/lora/anima.yaml`
  - use `aesthetic_shadow`
  - use `wd_reference_similarity`
  - do not start `UnifiedReward-Flex` for the current AWM scaling route
  - set a finite `train.max_epochs` before unattended formal training because the current AWM YAML omits it

## Validation Runs

### Smoke Test

- Date: `2026-04-11`
- Overrides: `max_dataset_size=4`, `resolution=64`, `num_inference_steps=2`, `group_size=2`, `unique_sample_num_per_epoch=4`, `max_epochs=1`, `num_processes=1`
- Outcome: Success. The full `sample -> reward -> optimize` loop completed with both rewards enabled.
- Reward summary:
  - `train/reward_aesthetic_shadow_mean=0.5170`
  - `train/reward_aesthetic_shadow_std=0.2906`
  - `train/reward_unifiedreward_flex_mean=0.5000`
  - `train/reward_unifiedreward_flex_std=0.4330`
  - `train/reward_unifiedreward_flex_zero_std_ratio=0.2500`
- Parameter update summary:
  - `train/adv_min=-1.4866`
  - `train/adv_max=1.4866`
  - `train/grad_norm=38.9422`
- Notes: This run validated the remote pointwise reward, the remote groupwise reward, and the training loop end-to-end.

### Short Run

- Date: `2026-04-11`
- Overrides: `examples/grpo/lora/anima_anime_custom_short_2gpu.yaml` with `num_processes=1`, `max_dataset_size=32`, `resolution=256`, `num_inference_steps=4`, `group_size=4`, `unique_sample_num_per_epoch=8`, `max_epochs=1`, `eval_freq=0`
- Outcome: Success. The training loop completed and saved `checkpoint-0`.
- Reward trend:
  - `train/reward_aesthetic_shadow_mean=0.5189`
  - `train/reward_aesthetic_shadow_std=0.1982`
  - `train/reward_unifiedreward_flex_mean=0.5000`
  - `train/reward_unifiedreward_flex_std=0.3560`
  - `train/reward_unifiedreward_flex_zero_std_ratio=0`
  - `train/reward_unifiedreward_flex_group_std_mean=0.3516`
- Parameter update summary:
  - `train/adv_min=-1.8970`
  - `train/adv_max=1.7624`
  - `train/grad_norm=0.7264`
- Image quality notes: Not assessed in this fallback run because `eval_freq=0` disabled periodic evaluation image export.
- Notes: The run finished without recurring `500` errors, reward timeouts, or deadlocks. Groupwise reward evaluation completed `8/8` successfully.
- Serving note: these recorded metrics were produced on the earlier `18082` backend. The faster `.venv` + `8080` backend was adopted later as the recommended runtime path for future runs.
- Current status: the active training plan has since been simplified to shadow-only; this dual-reward run is retained here as a historical benchmark.

### Formal Run

- Date: `N/A`
- Outcome: Not executed in this turn.
- Best checkpoint: `N/A`
- Final checkpoint: `N/A`
- Reward trend: `N/A`
- Image quality notes: `N/A`
- Notes: The formal YAML is ready, but the current turn stopped after the smoke test and a successful 2-GPU short fallback run.

### April 26 AWM Remote-Reward Smoke

- Date: `2026-04-26`
- Base config: `examples/awm/lora/anima.yaml`
- Runtime scale overrides: `max_dataset_size=1`, `resolution=64`, `num_inference_steps=2`, `group_size=2`, `unique_sample_num_per_epoch=1`, `num_train_timesteps=1`, `max_epochs=1`, `eval_freq=0`
- Active rewards:
  - `aesthetic_shadow`
  - `wd_reference_similarity`
- Outcome: Success. The full `sample -> remote rewards -> advantages -> optimize` loop completed on one GPU with both remote rewards enabled.
- Reward summary:
  - `aesthetic_shadow` mean/std: `0.5521295070648193 / 0.12267257273197174`
  - `wd_reference_similarity` mean/std: `0.1760590374469757 / 0.020329244434833527`
- Parameter update summary:
  - `train/adv_min=-1`
  - `train/adv_max=1`
  - `train/ratio_min=1`
  - `train/ratio_max=1`
  - `train/grad_norm=0.0173`
  - optimizer steps completed: `1`
- Notes: this run validated the exact remote reward endpoints and WD prompt-hash cache used by the current AWM scaling plan.

### April 26 Synthetic Compatibility Smokes

- `CUDA_VISIBLE_DEVICES=0 python scripts/validate_anima_lora_smoke.py --scenario awm`: pass, `2` samples, `1` optimizer step, `280` changed LoRA parameters.
- `CUDA_VISIBLE_DEVICES=0 python scripts/validate_anima_lora_smoke.py --scenario nft`: pass, `2` samples, `1` optimizer step, finite policy loss `4.0613`, `280` changed LoRA parameters.
- `CUDA_VISIBLE_DEVICES=0 python scripts/validate_anima_lora_smoke.py --scenario grpo-sageattn`: pass, `2` samples, `1` optimizer step, ratio stayed at `1.0`, `112` changed LoRA parameters.

### April 26 Attention Benchmark Smoke

- `flash`, batch size `2`, `64x64`, `2` inference steps:
  - total step time: `1.498s`
  - train throughput: `1.335 img/s`
  - sample peak allocation: `4.399 GiB`
  - optimize peak allocation: `5.505 GiB`
- `sageattn`, batch size `2`, `64x64`, `2` inference steps:
  - total step time: `2.675s`
  - train throughput: `0.748 img/s`
  - sample peak allocation: `4.399 GiB`
  - optimize peak allocation: `4.733 GiB`
- Notes: these are smoke-level benchmark numbers only, not target GPU capacity limits.

## Outputs

- Save directory: `saves/anima_anime_custom_short_2gpu`
- Checkpoint paths:
  - `saves/anima_anime_custom_short_2gpu/checkpoints/checkpoint-0/adapter_model.safetensors`
  - `saves/anima_anime_custom_short_2gpu/checkpoints/checkpoint-0/adapter_config.json`
  - `saves/anima_anime_custom_short_2gpu/checkpoints/checkpoint-0/README.md`
- Sample image paths: no periodic evaluation images were exported in the completed short fallback run
- TensorBoard path: `saves/tensorboard/anima_anime_custom_short_2gpu`

## Failures and Fixes

1. Failure: Anima evaluation crashed when trajectory latents were not collected.
   - Cause: `AnimaAdapter.inference()` unconditionally stacked `all_latents`, but evaluation paths may legitimately disable the trajectory collector and return `None`.
   - Fix: `src/flow_factory/models/anima/anima.py` now keeps `all_latents=None` when the collector is disabled, matching the behavior of other adapters.
2. Failure: Loopback reward calls were vulnerable to global proxy environment variables.
   - Cause: both `requests` and `httpx` inherited proxy settings even for `127.0.0.1`, which is unsafe for local reward services.
   - Fix: `src/flow_factory/rewards/my_reward_remote.py` and `scripts/reward_servers/unifiedreward_flex_bridge.py` now disable environment proxies for loopback targets.
3. Failure: the UnifiedReward-Flex bridge returned `500` on long Markdown prompts or verbose / truncated winner fields.
   - Cause: the judge could echo long prompts and return string-valued or partially truncated JSON winner payloads.
   - Fix: the bridge now normalizes and truncates prompts, increases the completion budget to `512`, and accepts integer, string, and truncated `winner` payloads.

## Conclusion

- Did the full training run complete? No formal long run has been executed. Historical GRPO smoke and short fallback runs completed; the current AWM remote-reward route completed a tiny single-GPU smoke.
- Did reward variance stay healthy? Yes in the current AWM tiny smoke. Both `aesthetic_shadow` and `wd_reference_similarity` returned non-constant rewards.
- Did image quality improve? Not evaluated in the April 26 update because the run was a functional smoke with evaluation disabled.
- Recommended next step: set a finite `train.max_epochs`, then run a short `512x512` AWM job with both remote rewards and evaluation enabled before scaling to the target RTX 5090 run.
