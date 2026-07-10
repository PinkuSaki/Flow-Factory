# Anima LoRA Scaling Plan

## Scope

Scale Anima LoRA training from the original small GRPO workflow toward a larger
single-GPU or two-GPU run with multiple image rewards.

This document is the current operational plan for the custom anime prompt route.
Implementation details and validation history are recorded in:

- `guidance/anima_support.md`
- `guidance/anima_custom_training_report.md`

## Hardware and Data

- Target hardware: RTX 5090 class single-GPU or two-GPU host.
- Current validation host: one visible NVIDIA GeForce RTX 4080 with 32760 MiB VRAM.
- Dataset directory: `dataset/anime_custom_single_gpu_eval16`.
- Source prompt file: `dataset/anime_custom_single_gpu_eval16/anime4k.jsonl`.
- Source record count: `4049`.
- Train split: `dataset/anime_custom_single_gpu_eval16/train.jsonl`, `4033` records.
- Test split: `dataset/anime_custom_single_gpu_eval16/test.jsonl`, `16` records.
- Split rule: random sample `16` source records for test with seed `42`; use the
  remaining records for train.
- Test indices from the source JSONL: `102,356,419,456,571,914,1003,1126,2233,2418,2619,2771,3016,3033,3037,3654`.

The WD reference embedding cache remains valid after this split because it covers
all `4049` source prompts:

- `dataset/anime_custom_single_gpu_eval16/wd_prompt_hash_cache.pt`
- health-check reference count: `4049`

## Training Config

Primary config:

- `examples/awm/lora/anima.yaml`

Current important settings:

- `trainer_type: awm`
- `model_type: anima`
- `model_name_or_path: models/animaOfficial_preview3Base.safetensors`
- `finetune_type: lora`
- `lora_rank: 32`
- `lora_alpha: 64`
- `attn_mode: flash`
- `split_attn: true`
- `num_processes: 1`
- `resolution: 512`
- `num_inference_steps: 10`
- `eval.num_inference_steps: 28`
- `group_size: 16`
- `unique_sample_num_per_epoch: 48`
- `per_device_batch_size: 6`
- `max_dataset_size: 4000`

Important operational note:

- `examples/awm/lora/anima.yaml` currently omits `train.max_epochs`.
- `max_epochs=None` means the trainer runs until interrupted.
- Set an explicit finite `max_epochs` before unattended formal runs.

## Reward Design

The active reward stack contains two remote pointwise rewards.

### Aesthetic Shadow

- Reward name: `aesthetic_shadow`
- Config class: `flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel`
- Local checkpoint: `/root/reward_models/aesthetic-shadow-v2-backup`
- Endpoint: `http://127.0.0.1:18081`
- Recommended training config options:
  - `remote_dispatch_mode: main_process`
  - `remote_max_concurrent_requests: 1`
  - `remote_offload_after_compute: true`

Reference command:

```bash
python scripts/reward_servers/shadow_server.py \
  --host 127.0.0.1 \
  --port 18081 \
  --model-path /root/reward_models/aesthetic-shadow-v2-backup \
  --device cpu \
  --dtype float32 \
  --max-batch-size 1
```

For production throughput, use a CUDA device and BF16 if there is enough free
VRAM for the reward service.

### WD Prompt-Hash Reference Similarity

- Reward name: `wd_reference_similarity`
- Config class: `flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel`
- Local checkpoint: `/root/reward_models/wd-eva02-large-tagger-v3`
- Reference embedding cache: `dataset/anime_custom_single_gpu_eval16/wd_prompt_hash_cache.pt`
- Endpoint: `http://127.0.0.1:18082`
- Runtime behavior: hash each prompt, look up the cached reference embedding,
  encode the generated image, and return cosine similarity.

Reference command:

```bash
python scripts/reward_servers/wd_prompt_hash_server.py \
  --host 127.0.0.1 \
  --port 18082 \
  --model-path /root/reward_models/wd-eva02-large-tagger-v3 \
  --cache-path dataset/anime_custom_single_gpu_eval16/wd_prompt_hash_cache.pt \
  --device cpu \
  --dtype float32 \
  --max-batch-size 1
```

## Single-GPU Validation Status

Completed on April 26, 2026.

### Static Checks

- `python -m compileall -q src scripts`: pass.
- `git diff --check`: pass.
- `black --check scripts/validate_anima_lora_smoke.py scripts/benchmark_anima_attention.py`: pass.
- `isort --check-only scripts/validate_anima_lora_smoke.py scripts/benchmark_anima_attention.py`: pass.
- Full-repository `black --check src ...` and `isort --check src ...` still fail
  on pre-existing formatting drift outside the current Anima scaling changes.

### Dataset Checks

- `train.jsonl`: `4033` valid JSONL records.
- `test.jsonl`: `16` valid JSONL records.
- Train/test overlap: `0`.
- Train/test union matches the `anime4k.jsonl` source set.
- `GeneralDataset(..., split="train", enable_preprocess=False)` loads `4033` records.
- `GeneralDataset(..., split="test", enable_preprocess=False)` loads `16` records.

### Synthetic Trainer Smokes

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/validate_anima_lora_smoke.py --scenario awm
CUDA_VISIBLE_DEVICES=0 python scripts/validate_anima_lora_smoke.py --scenario nft
CUDA_VISIBLE_DEVICES=0 python scripts/validate_anima_lora_smoke.py --scenario grpo-sageattn
```

Results:

- AWM smoke: `2` samples, image shape `(3, 64, 64)`, optimizer steps `1`,
  changed LoRA parameters `280`, advantage range `[-1, 1]`.
- NFT smoke: `2` samples, image shape `(3, 64, 64)`, optimizer steps `1`,
  changed LoRA parameters `280`, finite policy loss `4.0613`.
- GRPO SageAttention smoke: `2` samples, image shape `(3, 64, 64)`,
  optimizer steps `1`, changed LoRA parameters `112`, ratio stayed at `1.0`.

### Reward Server Smokes

- Aesthetic Shadow `/health`: pass.
- Aesthetic Shadow `/load`, `/compute`, `/offload`: pass.
- WD prompt-hash `/health`: pass, `reference_count=4049`.
- WD prompt-hash `/load`, `/compute`, `/offload`: pass.

### AWM Remote-Reward End-to-End Smoke

The tiny AWM run loaded `examples/awm/lora/anima.yaml`, replaced only runtime
scale fields, generated `2` images at `64x64`, computed both real remote rewards,
and completed one optimizer step.

Observed reward statistics:

- `aesthetic_shadow`: mean `0.5521295070648193`, std `0.12267257273197174`.
- `wd_reference_similarity`: mean `0.1760590374469757`, std `0.020329244434833527`.
- Advantage range: `[-1, 1]`.
- Optimizer steps: `1`.
- Logged grad norm: `0.0173`.

### Attention Benchmark Smoke

Commands:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_anima_attention.py --backend flash --batch-size 2 --resolution 64 --num-inference-steps 2
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_anima_attention.py --backend sageattn --batch-size 2 --resolution 64 --num-inference-steps 2
```

Results:

- `flash`: `2` samples, total step `1.498s`, train throughput `1.335 img/s`,
  sample peak `4.399 GiB`, optimize peak `5.505 GiB`, LoRA delta sum `157.0`.
- `sageattn`: `2` samples, total step `2.675s`, train throughput `0.748 img/s`,
  sample peak `4.399 GiB`, optimize peak `4.733 GiB`, LoRA delta sum `157.0`.

These are smoke-level numbers only. They validate script execution and backend
compatibility; they are not capacity limits for the target GPU.

## Next Steps

1. Add a finite `train.max_epochs` to `examples/awm/lora/anima.yaml` before a
   formal unattended run.
2. Decide whether reward services should run on CPU, on a second GPU, or in
   serialized CUDA offload mode on the training GPU.
3. Run a short `512x512` AWM job with the real remote reward stack and evaluation
   enabled.
4. If the short run is stable, scale `unique_sample_num_per_epoch`,
   `max_epochs`, and reward service batch sizes for the target RTX 5090 setup.
