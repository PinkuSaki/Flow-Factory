# Starting Anima Training After the Upstream Merge

## Purpose

This guide describes how to start a new Anima training run after merging
`upstream/main` at `40bc85c` into the local branch. It focuses on operational
differences from the pre-merge branch, the validated launch paths, and checks that
must be completed before a long-running job.

Validation baseline:

- upstream merge commit: `4507c4e`
- two-GPU validation commit: `8fe85eb`
- validation date: July 12, 2026
- validated hardware: two NVIDIA RTX 4080 SUPER GPUs
- validated Anima algorithms: AWM and DPO
- validated distributed backends: DeepSpeed ZeRO-2 and FSDP2

The recommended production baseline is Anima LoRA with AWM, BF16, two-GPU
DeepSpeed ZeRO-2, and main-process remote reward dispatch. Run the corresponding
smoke test before every new production configuration.

## What Changed

| Area | Previous usage | Current usage | Operational impact |
|---|---|---|---|
| Example layout | Flat paths such as `examples/awm/lora/anima48.yaml` | `examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml` | Old example paths no longer work. |
| Dataset configuration | Flat `data.dataset_dir` and top-level `eval_datasets` | Unified `data.datasets` entries with `train` and `eval` blocks | Use stable dataset names and do not mix legacy and unified forms. |
| Trainable dtype | `master_weight_dtype` | `model.trainable_parameters_dtype` | Replace the old key in custom configurations. |
| Training arguments | One broad argument surface | Algorithm-specific AWM, DPO, GRPO, DPPO, DGPO, NFT, CRD, and OPD argument classes | A field valid for one trainer may be ignored or invalid for another trainer. |
| Batch geometry | Commonly calculated manually | Aligned from world size, sampler, group size, batch size, and optimizer steps | Do not carry an old gradient accumulation value into a new distributed run. |
| Distributed preparation | Components prepared separately | One `ModelBundle` root is prepared with the optimizer | Checkpoint state depends on bundle and trainable-component composition. |
| Reward dispatch | Commonly one request stream per rank | `main_process` and `per_rank` dispatch modes | Use `main_process` for one local endpoint unless per-rank service capacity is intentional. |
| Dataset cache | Broad manual cache deletion | Fingerprinted, distributed Flow-Factory cache | Prefer `force_reprocess: true` or a targeted cache removal. |
| Launcher state | YAML values were treated as the complete launch state | CLI, environment, and YAML are reconciled | Explicit `accelerate launch` values can override YAML process settings. |

Canonical Anima examples now include:

- `examples/awm/lora/anima/default.yaml`
- `examples/awm/lora/anima/artist_multi_reward.yaml`
- `examples/awm/lora/anima/dual_reward.yaml`
- `examples/awm/lora/anima/single_gpu_smoke.yaml`
- `examples/awm/lora/anima/multi_gpu_deepspeed_zero2_smoke.yaml`
- `examples/awm/lora/anima/multi_gpu_fsdp2_smoke.yaml`
- `examples/grpo/lora/anima/default.yaml`
- `examples/nft/lora/anima/default.yaml`

## Required Anima Configuration

Verify every path before launching:

```yaml
model:
  model_type: "anima"
  model_name_or_path: "models/animaBase1_ex05.safetensors"
  qwen3: "models/qwen_3_06b_base.safetensors"
  vae: "models/qwen_image_vae.safetensors"
  t5_tokenizer_path: "tokenizer/t5_old"
  sd_scripts_root: "~/sd-scripts"
  target_components: "transformer"
  target_modules: "default"
  trainable_parameters_dtype: "bf16"
```

Important Anima-specific behavior:

- The external `sd-scripts` runtime remains a required dependency.
- Default LoRA targets the transformer blocks and excludes `llm_adapter`.
  Targeting `llm_adapter` must be explicit.
- LoRA rank, alpha, target modules, and base checkpoint form part of checkpoint
  compatibility. The rank-8 smoke checkpoint is not resumable as the rank-64
  production adapter.
- `guidance_scale > 1.0` activates classifier-free guidance and requires cached
  negative-prompt encodings. Keep rollout, training, and evaluation guidance
  intentional.
- Use BF16 consistently for `mixed_precision`,
  `model.trainable_parameters_dtype`, and `train.latent_storage_dtype`.
- Do not manually cast prepared model parameters.

The single-GPU and FSDP2 smoke examples use `anima-base-v1.0.safetensors`, while
the artist ZeRO-2 route uses `animaBase1_ex05.safetensors`. Confirm which base is
intended instead of treating the smoke examples as interchangeable checkpoints.

## Dataset Migration

Use the unified dataset schema for every new configuration:

```yaml
data:
  datasets:
    - name: anime_custom_artist
      dataset_dir: "dataset/anime_custom_artist_eval16"
      train:
        weight: 1
        max_dataset_size: 1968
      eval: {}
  preprocessing_batch_size: 16
  dataloader_num_workers: 8
  force_reprocess: false
  cache_dir: "~/.cache/flow_factory/datasets"
  sampler_type: "auto"
```

Each `name` is a routing identifier, not a display label. A reward restricted to
specific sources must reference exactly these names:

```yaml
rewards:
  - name: source_specific_reward
    applicable_datasets: ["anime_custom_artist"]
```

Top-level `eval_datasets` can still be migrated by the parser, but it is
deprecated. New configurations should place evaluation settings in each dataset's
`eval` block.

Preprocessing is fingerprinted from the dataset, model, preprocessing function,
and relevant arguments. Set `force_reprocess: true` when changing preprocessing
code without changing its fingerprinted inputs. A changed Anima guidance scale,
base model, or prompt encoding setup should be treated as a new preprocessing
state. Do not delete all Hugging Face and Flow-Factory caches by default; the
destructive commands in `command.txt` are intentionally commented out.

## Algorithm and Scheduler Compatibility

Flow-Factory now enforces two training paradigms:

| Paradigm | Algorithms | Allowed dynamics |
|---|---|---|
| Coupled | GRPO, GRPO-Guard, DPPO | SDE only: `Flow-SDE`, `Dance-SDE`, or `CPS` |
| Decoupled | AWM, DPO, NFT, DGPO, CRD | ODE or SDE |
| Distillation | DiffusionOPD | ODE or SDE |

The validated Anima AWM and DPO paths use:

```yaml
scheduler:
  dynamics_type: "ODE"
```

Do not start Anima GRPO by changing only `trainer_type` in an AWM file. GRPO must
also use an SDE scheduler and its algorithm-specific arguments.

DPO forms chosen/rejected pairs at the start of optimization. It requires
`group_size >= 2`, a reference model, and non-degenerate within-group rewards.

## Distributed Batch Geometry

Let:

- `M = unique_sample_num_per_epoch`
- `K = group_size`
- `W = distributed world size`
- `B = per_device_batch_size`
- `G = gradient_step_per_epoch`

With automatically derived gradient accumulation, the repeated sample count must
align with `W * B * G`. `group_contiguous` additionally requires `M` to be
divisible by `W`. The parser can increase `M` to satisfy these constraints.

Do not assume the YAML values are the final runtime values. Parse the file under
the intended world size first:

```bash
WORLD_SIZE=2 python - <<'PY'
from flow_factory.hparams import Arguments

path = "examples/awm/lora/anima/artist_multi_reward.yaml"
config = Arguments.load_from_yaml(path)
train = config.training_args
print("training args:", type(train).__name__)
print("sampler:", config.data_args.sampler_type)
print("unique samples:", train.unique_sample_num_per_epoch)
print("batches per epoch:", train.num_batches_per_epoch)
print("gradient accumulation:", train.gradient_accumulation_steps)
PY
```

The currently validated two-GPU results are:

| Configuration | M | K | B | Batches per epoch | Gradient accumulation |
|---|---:|---:|---:|---:|---:|
| ZeRO-2 smoke | 2 | 2 | 1 | 2 | 2 |
| FSDP2 smoke | 4 | 2 | 1 | 4 | 4 |
| Artist multi-reward | 48 | 16 | 8 | 48 | 192 |

The artist configuration generates `48 * 16 = 768` rollout images per epoch at
512 resolution and uses four training timesteps. It is a production workload, not
a smoke test.

## Reward Services

Start only the services referenced by the selected training YAML. The complete
validated service commands are in [`command.txt`](../command.txt).

Service mapping for the current examples:

| Port | Service | Used by |
|---:|---|---|
| 18081 | Aesthetic Shadow | Single-GPU, FSDP2, and DPO smoke routes |
| 18082 | WD prompt-hash similarity | Artist multi-reward route |
| 18084 | Wavelet prompt-hash similarity | Artist multi-reward route |
| 18085 | WD ConvNeXt perceptual reward | Artist multi-reward route |

Check health before training:

```bash
curl -fsS http://127.0.0.1:18082/health
curl -fsS http://127.0.0.1:18084/health
curl -fsS http://127.0.0.1:18085/health
```

For one local endpoint shared by all training ranks, use:

```yaml
remote_dispatch_mode: "main_process"
remote_offload_after_compute: true
```

`main_process` gathers applicable samples, sends one logical reward workload, and
scatters the results back to ranks. `remote_offload_after_compute: true` releases
reward-model GPU allocations between reward computation and policy optimization.
Model loading still creates transient VRAM pressure, so production sizing must
leave headroom.

## Validated Launch Sequence

### 1. Single-GPU preflight

Start the Aesthetic Shadow service on port `18081`, then run:

```bash
ff-train examples/awm/lora/anima/single_gpu_smoke.yaml
```

### 2. Two-GPU ZeRO-2 smoke

Start ports `18082`, `18084`, and `18085`, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file config/deepspeed/deepspeed_zero2.yaml \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    -m flow_factory.train \
    examples/awm/lora/anima/multi_gpu_deepspeed_zero2_smoke.yaml
```

### 3. Formal artist multi-reward run

After the smoke checkpoint has been verified, launch the formal configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file config/deepspeed/deepspeed_zero2.yaml \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    -m flow_factory.train \
    examples/awm/lora/anima/artist_multi_reward.yaml
```

The production YAML currently contains `num_processes: 1` and
`config_file: null`. The explicit `accelerate launch` arguments above are
therefore required for the validated two-GPU route. Running the YAML through a
single-process launcher is not equivalent.

Before an unattended formal run, set a finite `train.max_epochs`. The current
artist configuration omits it and therefore runs until interrupted. Also review
its evaluation cost: 16 evaluation prompts at 1024 resolution, 28 inference
steps, and `eval_freq: 1`.

## Formal Run Checklist

- Use a run-specific `log.run_name` and output directory.
- Set a finite `train.max_epochs`.
- Confirm the base Anima checkpoint and every auxiliary model path.
- Confirm LoRA rank, alpha, target components, and target modules.
- Confirm the dataset names, train/eval splits, and reward routing names.
- Confirm that every training dataset has at least one applicable reward.
- Parse the YAML with the target `WORLD_SIZE` and inspect warnings.
- Start only the required reward services and verify their health endpoints.
- Run the two-GPU smoke configuration with the same backend first.
- Confirm that reward standard deviations and advantages are non-zero.
- Confirm that loss and gradient norm are finite.
- Confirm that at least one optimizer step completes on every epoch.
- Compare consecutive LoRA checkpoints and confirm a finite parameter delta.
- Stop all reward workers after the run and verify that GPU allocations are gone.

## Checkpoint and Resume Rules

Prepared models now use one `ModelBundle` root. Checkpoints contain trainable
components only; frozen-but-shardable members are intentionally skipped.

Resume with the same:

- base model checkpoint
- `target_components`
- LoRA rank, alpha, and target modules
- trainable bundle composition
- distributed backend where full state is restored
- preprocessing and CFG assumptions

`resume_type: state` loads after `accelerator.prepare()` and restores model,
optimizer, and RNG state. This requires a checkpoint written with
`log.save_model_only: false`. The current Anima examples use
`save_model_only: true`, which writes model weights only; resume those as LoRA
weights rather than as full training state.

DeepSpeed ZeRO-3 remains unsupported. Use ZeRO-1, ZeRO-2, or FSDP2.

## DPO Status

Anima DPO passed a real two-GPU DeepSpeed ZeRO-2 smoke test with:

- `group_size: 2`
- `per_device_batch_size: 1`
- `beta: 100.0`
- `guidance_scale: 1.0`
- Aesthetic Shadow remote reward
- one optimizer step per epoch for two epochs

Observed behavior:

- DPO loss moved from `0.6931` to `0.6925`.
- Implicit preference accuracy moved from `0` to `1`.
- Raw gradient norms were large but finite and were clipped by
  `max_grad_norm: 1.0`.
- `280 / 560` LoRA tensors changed between checkpoints.
- All checkpoint values and deltas were finite.

This validates the Anima DPO execution path, not a production DPO recipe. There is
currently no tracked `examples/dpo/lora/anima/default.yaml`. A formal DPO run
should first add a reviewed, finite production example with deliberate beta,
learning rate, CFG, reward, and checkpoint settings.

## Validation Reference

The post-merge test results provide the following baseline:

| Path | Result |
|---|---|
| AWM single GPU | One real optimizer step; finite reward, advantage, and gradient norm |
| AWM two-GPU ZeRO-2 | Two epochs, three remote rewards, two optimizer steps, finite LoRA checkpoint delta |
| AWM two-GPU FSDP2 | Eight generated samples, finite reward and gradient norm, clean exit |
| DPO two-GPU ZeRO-2 | Two epochs, real reference forwards, improving pairwise metrics, finite checkpoint delta |
| Reward services | All commands in `command.txt` loaded, scored, offloaded, and shut down cleanly |

Detailed Anima implementation and historical validation notes are available in
[`anima_support.md`](anima_support.md).

## New Upstream Capabilities

The merged upstream branch also adds or expands:

- DPPO, DGPO, CRD, and DiffusionOPD trainers
- multi-dataset training with source-aware reward routing
- Bagel and LTX2 model support
- Qwen Image Bench and GenEval2 reward/evaluation paths
- model-agnostic latent geometry
- Qwen CFG merging, host-to-device/FSDP prefetch, and communication fusion
- trainable-only checkpoint loading for bundled FSDP2 models

These capabilities do not require changing a working Anima AWM run. Introduce one
algorithm, reward, model, or backend change at a time and repeat the smoke test
after each change.
