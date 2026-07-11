# 上游合并后的 Anima 新训练指南

## 文档目的

本文说明将 `upstream/main@40bc85c` 合并到本地分支后，如何开始一次新的
Anima 训练。内容重点是相对于合并前分支的操作变化、已经验证的启动路径，
以及长时间训练开始前必须完成的检查。

对应英文文档：[`anima_post_merge_training.md`](anima_post_merge_training.md)。

验证基线：

- 上游合并提交：`4507c4e`
- 双卡验证提交：`8fe85eb`
- 验证日期：2026 年 7 月 12 日
- 验证硬件：两张 NVIDIA RTX 4080 SUPER
- 已验证的 Anima 算法：AWM、DPO
- 已验证的分布式后端：DeepSpeed ZeRO-2、FSDP2

当前推荐的正式训练基线是 Anima LoRA + AWM + BF16 + 双卡 DeepSpeed
ZeRO-2 + 主进程远程奖励调度。每次创建新的正式配置后，都应先运行对应的
冒烟测试。

## 合并后的主要变化

| 范围 | 以前的用法 | 当前用法 | 对训练的影响 |
|---|---|---|---|
| 示例目录 | `examples/awm/lora/anima48.yaml` 等扁平路径 | `examples/{algorithm}/{finetune_type}/{model_type}/{variant}.yaml` | 旧示例路径已经失效。 |
| 数据集配置 | 扁平的 `data.dataset_dir` 和顶层 `eval_datasets` | 统一的 `data.datasets`，每个数据源包含 `train`、`eval` 子块 | 应使用稳定的数据集名称，不要混用新旧结构。 |
| 可训练参数精度 | `master_weight_dtype` | `model.trainable_parameters_dtype` | 自定义旧配置必须替换该字段。 |
| 训练参数 | 共用一套较宽泛的参数 | AWM、DPO、GRPO、DPPO、DGPO、NFT、CRD、OPD 使用各自的参数类 | 某个算法有效的字段不一定适用于其他算法。 |
| 批次几何 | 通常手动计算 | 根据 world size、sampler、group size、batch size 和优化步数对齐 | 不要把旧的梯度累积值直接带入新分布式任务。 |
| 分布式准备 | 分别 prepare 多个组件 | 模型通过单一 `ModelBundle` 根节点和优化器一起 prepare | 完整状态恢复依赖 bundle 和可训练组件组成。 |
| 奖励调度 | 通常每个 rank 各自请求 | 支持 `main_process` 和 `per_rank` | 本机只有一个服务端点时，优先使用 `main_process`。 |
| 数据缓存 | 经常手动删除整套缓存 | 使用指纹化、分布式的 Flow-Factory 缓存 | 优先使用 `force_reprocess: true` 或定向删除。 |
| 启动参数 | YAML 通常代表完整启动状态 | CLI、环境变量和 YAML 会在运行时协调 | 显式的 `accelerate launch` 参数可以覆盖 YAML 中的进程设置。 |

当前规范的 Anima 示例包括：

- `examples/awm/lora/anima/default.yaml`
- `examples/awm/lora/anima/artist_multi_reward.yaml`
- `examples/awm/lora/anima/dual_reward.yaml`
- `examples/awm/lora/anima/single_gpu_smoke.yaml`
- `examples/awm/lora/anima/multi_gpu_deepspeed_zero2_smoke.yaml`
- `examples/awm/lora/anima/multi_gpu_fsdp2_smoke.yaml`
- `examples/grpo/lora/anima/default.yaml`
- `examples/nft/lora/anima/default.yaml`

## Anima 必需配置

启动前检查每一个路径：

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

Anima 特有的注意事项：

- 外部 `sd-scripts` 运行时仍然是必需依赖。
- 默认 LoRA 只作用于 transformer blocks，不包含 `llm_adapter`。如需训练
  `llm_adapter`，必须显式指定目标模块。
- LoRA rank、alpha、目标模块和基础模型共同决定 checkpoint 是否兼容。
  rank 8 的冒烟测试 checkpoint 不能作为 rank 64 正式训练的恢复点。
- `guidance_scale > 1.0` 会启用 CFG，并要求缓存中包含 negative prompt 编码。
  rollout、训练和评估使用的 guidance 必须经过明确设计。
- `mixed_precision`、`model.trainable_parameters_dtype` 和
  `train.latent_storage_dtype` 应统一使用 BF16。
- 不要手动转换已经 prepare 的模型参数精度。

单卡和 FSDP2 冒烟配置使用 `anima-base-v1.0.safetensors`，artist ZeRO-2
路径使用 `animaBase1_ex05.safetensors`。不要把不同冒烟配置生成的 checkpoint
当作可以互换的训练起点。

## 数据配置迁移

所有新配置都应使用统一的数据集结构：

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

每个数据集的 `name` 是路由标识符，不只是显示名称。限制到特定数据源的
奖励必须准确引用这些名称：

```yaml
rewards:
  - name: source_specific_reward
    applicable_datasets: ["anime_custom_artist"]
```

解析器目前仍能迁移顶层 `eval_datasets`，但该结构已经弃用。新配置应将
评估设置放入对应数据源的 `eval` 子块。

预处理缓存指纹由数据集、模型、预处理函数和相关参数共同决定。如果修改了
预处理代码，但指纹输入没有发生变化，应设置 `force_reprocess: true`。
修改 Anima guidance scale、基础模型或 prompt 编码方式时，应按新的预处理
状态处理。不要默认删除全部 Hugging Face 和 Flow-Factory 缓存；
`command.txt` 中的破坏性删除命令已经有意注释。

## 算法与 Scheduler 兼容性

Flow-Factory 当前区分三类训练范式：

| 范式 | 算法 | 允许的 dynamics |
|---|---|---|
| 耦合训练 | GRPO、GRPO-Guard、DPPO | 只能使用 SDE：`Flow-SDE`、`Dance-SDE` 或 `CPS` |
| 解耦训练 | AWM、DPO、NFT、DGPO、CRD | 可以使用 ODE 或 SDE |
| 蒸馏 | DiffusionOPD | 可以使用 ODE 或 SDE |

已经验证的 Anima AWM 和 DPO 路径使用：

```yaml
scheduler:
  dynamics_type: "ODE"
```

不能只把 AWM 配置中的 `trainer_type` 改成 GRPO 就启动训练。GRPO 还必须
切换到 SDE scheduler，并使用 GRPO 对应的算法参数。

DPO 会在优化开始时构造 chosen/rejected 样本对。它要求 `group_size >= 2`、
参考模型，以及组内非退化的奖励结果。

## 分布式批次几何

定义：

- `M = unique_sample_num_per_epoch`
- `K = group_size`
- `W = distributed world size`
- `B = per_device_batch_size`
- `G = gradient_step_per_epoch`

自动计算梯度累积时，重复采样总数必须与 `W * B * G` 对齐。
`group_contiguous` 还要求 `M` 可以被 `W` 整除。解析器可能会增大 `M`，
以满足这些限制。

不要假定 YAML 中填写的值就是最终运行值。启动前应使用目标 world size
解析配置：

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

当前已经验证的双卡解析结果：

| 配置 | M | K | B | 每个 epoch 的 batch 数 | 梯度累积步数 |
|---|---:|---:|---:|---:|---:|
| ZeRO-2 冒烟测试 | 2 | 2 | 1 | 2 | 2 |
| FSDP2 冒烟测试 | 4 | 2 | 1 | 4 | 4 |
| Artist 多奖励正式配置 | 48 | 16 | 8 | 48 | 192 |

Artist 配置每个 epoch 会在 512 分辨率下生成 `48 * 16 = 768` 张 rollout
图片，并使用 4 个训练 timestep。它是正式训练负载，不是冒烟测试。

## 奖励服务

只启动当前训练 YAML 实际引用的服务。完整且已验证的服务命令位于
[`command.txt`](../command.txt)。

当前示例的服务对应关系：

| 端口 | 服务 | 使用场景 |
|---:|---|---|
| 18081 | Aesthetic Shadow | 单卡、FSDP2 和 DPO 冒烟路径 |
| 18082 | WD prompt-hash similarity | Artist 多奖励路径 |
| 18084 | Wavelet prompt-hash similarity | Artist 多奖励路径 |
| 18085 | WD ConvNeXt perceptual reward | Artist 多奖励路径 |

启动训练前检查健康状态：

```bash
curl -fsS http://127.0.0.1:18082/health
curl -fsS http://127.0.0.1:18084/health
curl -fsS http://127.0.0.1:18085/health
```

多个训练 rank 共用一个本地端点时，使用：

```yaml
remote_dispatch_mode: "main_process"
remote_offload_after_compute: true
```

`main_process` 会汇总适用的样本，发送一次逻辑奖励任务，再把结果分发回各个
rank。`remote_offload_after_compute: true` 会在奖励计算和策略优化之间释放
奖励模型占用的 GPU 显存。模型加载期间仍会产生瞬时显存压力，因此正式训练
必须预留显存余量。

## 已验证的启动顺序

### 1. 单卡预检

先启动端口 `18081` 上的 Aesthetic Shadow 服务，再运行：

```bash
ff-train examples/awm/lora/anima/single_gpu_smoke.yaml
```

### 2. 双卡 ZeRO-2 冒烟测试

先启动端口 `18082`、`18084` 和 `18085`，再运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file config/deepspeed/deepspeed_zero2.yaml \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    -m flow_factory.train \
    examples/awm/lora/anima/multi_gpu_deepspeed_zero2_smoke.yaml
```

### 3. Artist 多奖励正式训练

确认冒烟测试 checkpoint 正常后，启动正式配置：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file config/deepspeed/deepspeed_zero2.yaml \
    --num_processes 2 \
    --num_machines 1 \
    --mixed_precision bf16 \
    -m flow_factory.train \
    examples/awm/lora/anima/artist_multi_reward.yaml
```

正式 YAML 当前仍包含 `num_processes: 1` 和 `config_file: null`。因此，已验证
的双卡路径必须使用上面显式提供的 `accelerate launch` 参数。通过单进程
launcher 直接运行该 YAML 并不等价。

无人值守的正式训练必须先设置有限的 `train.max_epochs`。当前 artist 配置
没有该字段，因此会一直运行到人工中断。同时需要确认评估开销是否符合预期：
16 个评估 prompt、1024 分辨率、28 个 inference step，并且
`eval_freq: 1`。

## 正式训练检查清单

- 使用本次训练独立的 `log.run_name` 和输出目录。
- 设置有限的 `train.max_epochs`。
- 检查 Anima 基础模型和全部辅助模型路径。
- 检查 LoRA rank、alpha、目标组件和目标模块。
- 检查数据集名称、训练/评估 split 和奖励路由名称。
- 确认每个训练数据源至少有一个适用的奖励。
- 使用目标 `WORLD_SIZE` 解析 YAML，并阅读所有 warning。
- 只启动实际需要的奖励服务，并检查健康端点。
- 先使用相同分布式后端运行双卡冒烟配置。
- 确认 reward standard deviation 和 advantage 非零。
- 确认 loss 和 gradient norm 为有限值。
- 确认每个 epoch 至少完成一次 optimizer step。
- 比较连续两个 LoRA checkpoint，确认存在有限的参数变化。
- 训练结束后停止全部奖励 worker，并确认 GPU 占用已经释放。

## Checkpoint 与恢复训练

prepare 后的模型现在使用单一 `ModelBundle` 根节点。checkpoint 只包含
可训练组件；冻结但需要参与分片的组件会被有意跳过。

恢复训练时必须保持以下内容一致：

- 基础模型 checkpoint
- `target_components`
- LoRA rank、alpha 和目标模块
- 可训练 bundle 的组件组成
- 恢复完整状态时使用的分布式后端
- 预处理和 CFG 假设

`resume_type: state` 会在 `accelerator.prepare()` 后加载，并恢复模型、
优化器和 RNG 状态。它要求 checkpoint 由 `log.save_model_only: false`
生成。当前 Anima 示例均使用 `save_model_only: true`，只保存模型权重；
这些 checkpoint 应作为 LoRA 权重加载，而不是完整训练状态。

DeepSpeed ZeRO-3 仍不受支持。只能使用 ZeRO-1、ZeRO-2 或 FSDP2。

## DPO 当前状态

Anima DPO 已通过真实的双卡 DeepSpeed ZeRO-2 冒烟测试，测试配置包括：

- `group_size: 2`
- `per_device_batch_size: 1`
- `beta: 100.0`
- `guidance_scale: 1.0`
- Aesthetic Shadow 远程奖励
- 两个 epoch，每个 epoch 完成一次 optimizer step

观察结果：

- DPO loss 从 `0.6931` 下降到 `0.6925`。
- 隐式偏好准确率从 `0` 上升到 `1`。
- 原始 gradient norm 较大但保持有限，并由 `max_grad_norm: 1.0` 裁剪。
- 两个 checkpoint 之间有 `280 / 560` 个 LoRA tensor 发生变化。
- checkpoint 的所有数值和参数差值均为有限值。

这些结果验证了 Anima DPO 的执行路径，但不能作为正式 DPO 超参数方案。
当前还没有受版本控制的 `examples/dpo/lora/anima/default.yaml`。正式 DPO
训练开始前，应先增加一个经过评审且 epoch 有限的正式示例，并明确设计
beta、learning rate、CFG、奖励和 checkpoint 设置。

## 验证结果参考

合并后的测试结果提供了以下基线：

| 路径 | 结果 |
|---|---|
| AWM 单卡 | 完成一次真实 optimizer step；reward、advantage 和 gradient norm 均有限 |
| AWM 双卡 ZeRO-2 | 两个 epoch、三个远程奖励、两次 optimizer step，LoRA checkpoint 差值有限 |
| AWM 双卡 FSDP2 | 生成 8 个样本，reward 和 gradient norm 有限，进程正常退出 |
| DPO 双卡 ZeRO-2 | 两个 epoch，执行真实 reference forward，偏好指标改善，checkpoint 差值有限 |
| 奖励服务 | `command.txt` 中所有命令均完成加载、评分、卸载和干净退出验证 |

Anima 的实现细节和历史验证记录见
[`anima_support.md`](anima_support.md)。

## 上游新增能力

本次合并还新增或扩展了以下能力：

- DPPO、DGPO、CRD 和 DiffusionOPD trainer
- 多数据集训练和按数据源路由奖励
- Bagel 和 LTX2 模型支持
- Qwen Image Bench 和 GenEval2 奖励/评估路径
- 模型无关的 latent geometry
- Qwen CFG 合并、H2D/FSDP 预取和通信融合
- 面向 bundle FSDP2 模型的仅可训练组件 checkpoint 加载

这些能力不会要求已经正常工作的 Anima AWM 任务同步改动。一次只引入一种
算法、奖励、模型或后端变化，并在每次变化后重新执行冒烟测试。
