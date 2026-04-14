# Anima LoRA GRPO 执行计划

## 0. 当前执行状态（2026-04-12）

当前这份计划已经不是纯规划，而是包含了实际执行结果的状态文档。

已完成：

1. `scripts/prepare_anime_custom_dataset.py` 已落地，可在本地生成训练 / 测试 JSONL 拆分。
2. 数据转换规则已按最终要求落实：
   - 保留 Markdown 标题、段落与换行结构
   - 仅移除 `Artist` 段
3. 生成后的数据集文件保持本地使用，不提交到仓库。
4. `shadow` pointwise reward 服务已跑通：
   - 服务端口：`http://127.0.0.1:18081`
5. `UnifiedReward-Flex` 的 vLLM + bridge 已跑通：
   - vLLM 端口：`http://127.0.0.1:18082/v1`
   - bridge 端口：`http://127.0.0.1:18083`
6. 训练配置已落地：
   - 正式配置：`examples/grpo/lora/anima_anime_custom.yaml`
   - dual reward smoke：`examples/grpo/lora/anima_anime_custom_dual_smoke.yaml`
   - 2-GPU fallback short：`examples/grpo/lora/anima_anime_custom_short_2gpu.yaml`
7. dual reward smoke test 已完成，并成功跑完 `sample -> reward -> optimize` 闭环。
8. 2-GPU fallback short run 已完成，并保存了 LoRA checkpoint：
   - `saves/anima_anime_custom_short_2gpu/checkpoints/checkpoint-0`

关键结果：

1. smoke：
   - `train/reward_aesthetic_shadow_mean=0.5170`
   - `train/reward_aesthetic_shadow_std=0.2906`
   - `train/reward_unifiedreward_flex_mean=0.5000`
   - `train/reward_unifiedreward_flex_std=0.4330`
   - `train/grad_norm=38.9422`
2. short fallback：
   - `train/reward_aesthetic_shadow_mean=0.5189`
   - `train/reward_aesthetic_shadow_std=0.1982`
   - `train/reward_unifiedreward_flex_mean=0.5000`
   - `train/reward_unifiedreward_flex_std=0.3560`
   - `train/grad_norm=0.7264`

当前未完成项：

1. formal run 尚未执行。
2. 本轮 short fallback 为了压缩 wall-clock，使用了 `eval_freq: 0`，因此没有沉淀可直接人工对比的评估样图。

2026-04-12 补充更新：

1. `UnifiedReward-Flex` 当前推荐不再使用旧的 vLLM 环境，而是改用仓库内 `.venv` 的独立环境启动。
2. 当前推荐的更快 serving 拓扑为：
   - vLLM：`http://127.0.0.1:8080/v1`
   - bridge：`http://127.0.0.1:18083`
3. `bridge` 已支持 `--max-parallel-pairs`，建议先从 `2` 开始，而不是盲目拉高并发。
4. 已实测：
   - `http://127.0.0.1:8080/health` 返回 `200`
   - `http://127.0.0.1:18083/health` 返回 `{"status": "ok"}`
5. 注意：`.venv` 中的 vLLM 版本会提示 `VLLM_DISABLE_FLASHINFER_GDN_PREFILL` 是未知环境变量；服务仍可正常启动，但该环境变量是否生效不能假定。

2026-04-12 当前决策变更：

1. 当前活动训练方案已经去掉 `UnifiedReward-Flex`。
2. 当前唯一训练 reward 为 `aesthetic_shadow`。
3. `UnifiedReward-Flex` 相关服务、bridge 和 dual-reward 结果只保留为历史验证记录，不再属于当前推荐执行路径。

## 1. 目标

目标不是再次“接入 Anima”，而是在当前仓库已有 Anima 支持的基础上，完成一次可复现的 `anima + lora + grpo` 训练闭环，满足以下交付：

1. 将 `dataset/anime_custom/anime10k.json` 转成 Flow-Factory 可直接读取的数据集，并拆分训练集与测试集。
2. 接入训练 reward：
   - 唯一 reward：`/root/reward_models/aesthetic-shadow-v2-backup`
3. 基于现有 `examples/grpo/lora/anima.yaml` 产出一份专用训练配置并跑通完整训练。
4. 记录实施步骤、配置、日志、样图、reward 变化和最终结论，沉淀为文档。

## 2. 先纠偏：当前仓库现状

以下判断已经明确，后续计划必须基于这些事实执行：

1. `anima` 已经在当前仓库中接入完成，不应重复做模型适配。
   - 代码位置：`src/flow_factory/models/anima/anima.py`
   - 注册位置：`src/flow_factory/models/registry.py`
   - 样例配置：`examples/grpo/lora/anima.yaml`
   - 现状记录：`guidance/anima_support.md`
2. `dataset/anime_custom/anime10k.json` 当前是一个字典：
   - key 为图片文件名
   - value 为 `{caption, image_size}`
   - 当前目录下没有配套图片文件，说明本次应按“纯 prompt RL 数据集”处理，而不是监督微调数据集。
3. Flow-Factory 原生支持从 `train.txt` / `test.txt` 或 `train.jsonl` / `test.jsonl` 读取数据。
4. reward model 默认会在训练进程内被加载；对于 `shadow` 和 `UnifiedReward-Flex` 这类大模型，不适合直接让每个训练 rank 各自加载。
5. `UnifiedReward-Flex` 当前仓库提供的是 pairwise rank 玩法，不是现成 pointwise 分数模型。
   - 这意味着它更适合作为 `GroupwiseRewardModel` 使用。
6. 若引入 pairwise reward，`group_size=16` 的成本过高，因为组内比较是平方级增长；推荐将正式训练的 `group_size` 降到 `4`。

## 3. 推荐总体路线

推荐路线如下：

1. 不改 Anima 主适配器，只做数据、reward、配置、训练和文档四块。
2. 数据集走“纯 prompt 数据集”：
   - 保留原始 caption 的 Markdown 结构
   - 仅移除 `Artist` 段
   - 生成 `train.jsonl` 和 `test.jsonl`
3. reward 走“外置服务 + Flow-Factory 远程 reward 包装器”路线，而不是把 reward 模型直接塞进训练进程。
   - 直接复用现有：
     - `flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel`
     - `flow_factory.rewards.my_reward_remote.RemoteGroupwiseRewardModel`
4. `shadow` 作为唯一 reward，做 pointwise 打分。
5. `UnifiedReward-Flex` 不再进入当前训练链路，只保留历史验证记录。
6. 训练配置从 `examples/grpo/lora/anima.yaml` 复制修改，不从空白 YAML 开始写。

## 4. 资源与拓扑建议

如果本机是 8 卡，推荐采用以下拓扑，而不是让训练直接占满全部卡：

1. 训练使用 4 卡。
2. `UnifiedReward-Flex` 的 vLLM 服务使用 2 卡。
3. `shadow` 服务使用 1 卡。
4. 预留 1 卡给调试、评估或波动缓冲。

如果 GPU 不足，则按以下优先级退化：

1. 先保留训练 + `shadow` 主 reward。
2. 优先保障训练与 `shadow` 服务稳定。
3. `UnifiedReward-Flex` 不再作为当前训练目标的一部分。

关键结论：

1. 不建议在启用两个 reward 的同时继续使用 `num_processes: 8`。
2. 不建议在启用 `UnifiedReward-Flex` 时保留 `group_size: 16`。

## 5. 分阶段执行

### 阶段 0：环境预检

目标：确认训练、reward 和外部依赖路径都可用。

任务：

1. 确认以下路径存在且可读：
   - `/models`
   - `/tokenizer`
   - `/root/sd-scripts`
   - `/root/UnifiedReward`
   - `/root/reward_models/UnifiedReward-Flex-qwen35-4b`
   - `/root/reward_models/aesthetic-shadow-v2-backup`
2. 若需要联网拉依赖或 trust-remote-code，先设置代理：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export ALL_PROXY=socks5h://127.0.0.1:7891
```

3. 确认训练环境具备：
   - `accelerate`
   - `transformers`
   - `datasets`
   - `openai`
   - `vllm`（仅 `UnifiedReward-Flex` 服务需要）
4. 确认 GPU 支持 BF16；Anima 现有样例已按 BF16 路线配置。

验收标准：

1. Anima 样例配置可以被正常加载。
2. reward 模型路径可读。
3. 训练与 reward 服务的 GPU 分配方案确定。

### 阶段 1：数据集转换与拆分

目标：把 `anime10k.json` 变成可直接训练的 Markdown prompt 数据集。

建议新增脚本：

1. `scripts/prepare_anime_custom_dataset.py`

转换规则：

1. 输入：`dataset/anime_custom/anime10k.json`
2. 读取每个样本的 `caption`
3. 保留原始 Markdown 标题、段落和换行格式
4. 仅删除 `## Artist` 段及其对应内容
5. 不删除 `Texts`、`Background`、`Image effects`、`Atmosphere` 等其他段落
6. 不做单行化，不重写段落结构
7. 固定随机种子 `42` 做拆分

拆分建议：

1. 测试集固定抽取 `256` 条 prompt
2. 其余全部进入训练集
3. 额外输出一份映射清单，便于追溯原始样本来源

建议输出：

1. `dataset/anime_custom/train.jsonl`
2. `dataset/anime_custom/test.jsonl`
3. `dataset/anime_custom/split_manifest.jsonl`

`jsonl` 而不是 `txt` 的原因：

1. `txt` 是单行 prompt 语义，不适合保留多行 Markdown。
2. `jsonl` 可以在 `prompt` 字段中稳定保留换行。

验收标准：

1. `train.jsonl` / `test.jsonl` 可被 `GeneralDataset` 正常读取。
2. 没有空 prompt。
3. Markdown 标题与换行结构被保留。
4. `Artist` 段已移除，其余段落保留。
5. 随机种子固定后可重复生成相同拆分结果。

### 阶段 2：主 reward 接入（Aesthetic Shadow）

目标：让 `shadow` 成为主导 pointwise reward。

推荐实现路线：

1. 不把 `shadow` 模型直接加载到每个训练进程里。
2. 启动独立 reward 服务，只保留轻量客户端在训练进程中。

建议新增：

1. `scripts/reward_servers/shadow_server.py`

服务职责：

1. 加载 `/root/reward_models/aesthetic-shadow-v2-backup`
2. 使用 `ViTImageProcessor` + `ViTForImageClassification`
3. 接收单张或批量图片
4. 返回 pointwise score

首版分数定义建议：

1. 默认返回 `P(hq)`，即高质量类别概率
2. 若后续发现 reward 差异过小，再切换为 `logits[hq] - logits[lq]`

Flow-Factory 侧配置建议：

1. 直接使用远程包装器：
   - `reward_model: "flow_factory.rewards.my_reward_remote.RemotePointwiseRewardModel"`
2. 通过 `server_url` 指向本地服务，例如：
   - `http://127.0.0.1:18081`

初始权重建议：

1. `shadow.weight = 0.7`

验收标准：

1. 单张图返回稳定数值。
2. 一批图可正常批处理。
3. 对明显质量差异的样图能给出可区分的分数。

### 阶段 3：辅 reward 接入（UnifiedReward-Flex，历史验证路径）

目标：保留已验证过的历史路径说明；当前活动训练方案不再使用该模型。

关键判断：

1. `UnifiedReward-Flex` 当前代码以 pairwise rank 为主。
2. 它更适合比较同一 prompt 下的一组候选图像。
3. 对 GRPO 来说，这正好可以转成组内 win-rate reward。

推荐实现路线：

1. 该路径仅用于历史验证复现，不再用于当前训练。
2. 如后续重新启用，仍应采用 vLLM + bridge 的外置服务方式。

建议新增：

1. `scripts/reward_servers/unifiedreward_flex_bridge.py`

服务职责：

1. 调用本地 vLLM OpenAI-compatible 接口
2. 对同组样本做两两比较
3. 从 pairwise 输出中解析 winner
4. 将每张图的胜场数转成 `[0, 1]` 区间的 groupwise reward

vLLM 启动建议：

```bash
source .venv/bin/activate
export VLLM_DISABLE_FLASHINFER_GDN_PREFILL=1
export TOKENIZERS_PARALLELISM=false

vllm serve /root/reward_models/UnifiedReward-Flex-qwen35-4b \
  --host 0.0.0.0 \
  --port 8080 \
  --trust-remote-code \
  --served-model-name UnifiedReward \
  --gpu-memory-utilization 0.7 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

说明：

1. 上游脚本里 `tensor-parallel-size=8` 是通用范式，不适合和训练同时跑。
2. 本次目标是完成一次训练闭环，应优先保留训练资源。
3. 当前经验表明，使用独立 `.venv` 的这套 vLLM 启动参数，比旧环境中的默认启动更快。
4. 该版本的 vLLM 启动预热时间较长，首次可用前大约需要数十秒到两分钟，不应把冷启动时间误判为服务故障。

bridge 启动建议：

```bash
python scripts/reward_servers/unifiedreward_flex_bridge.py \
  --host 127.0.0.1 \
  --port 18083 \
  --api-base-url http://127.0.0.1:8080/v1 \
  --model UnifiedReward \
  --request-timeout 180 \
  --max-parallel-pairs 2
```

说明：

1. 训练侧仍然统一访问 `18083`，只替换 bridge 背后的 vLLM 实现即可。
2. `--max-parallel-pairs=2` 是当前更稳妥的默认值；单卡下继续提高不一定更快，可能只是把排队从 bridge 转移到 vLLM。

说明：

1. 本节只保留为历史验证记录。
2. 当前主训练配置与 short 配置已经去掉该 reward。

### 阶段 4：训练配置落地

目标：基于已有 Anima 样例产出专用训练 YAML。

建议新增：

1. `examples/grpo/lora/anima_anime_custom.yaml`

直接从以下文件复制：

1. `examples/grpo/lora/anima.yaml`

核心修改项：

1. `data.dataset_dir: "dataset/anime_custom"`
2. `log.logging_backend: "tensorboard"` 或已配置好的 `wandb`
3. `train.group_size: 4`
4. `train.unique_sample_num_per_epoch: 64`
5. `train.gradient_step_per_epoch: 2`
6. `train.max_epochs`
   - smoke：`2`
   - short run：`10`
   - formal run：`50`
7. `train.resolution: 512`
8. `train.num_inference_steps: 8` 或 `10`
9. `eval.eval_freq`
   - smoke：`1`
   - short/formal：`5` 或 `10`
10. `log.save_freq`
    - smoke：`1`
    - short/formal：`5` 或 `10`
11. 训练 reward 切到远程服务：
    - `shadow` 使用 `RemotePointwiseRewardModel`

建议保留的 Anima 配置：

1. `mixed_precision: "bf16"`
2. `model.master_weight_dtype: "bf16"`
3. `train.latent_storage_dtype: "bf16"`
4. `model.sd_scripts_root: "~/sd-scripts"`

验收标准：

1. YAML 能被 `Arguments.load_from_yaml(...)` 正常解析。
2. reward 客户端参数可被正确传入。
3. sampler 几何约束满足。

### 阶段 5：分三级执行训练

目标：避免一上来直接长跑，先把 reward 和参数更新验证清楚。

#### 5.1 Smoke Test

建议配置：

1. `num_processes: 1`
2. `max_dataset_size: 16`
3. `resolution: 256` 或 `384`
4. `num_inference_steps: 2`
5. `group_size: 4`
6. `unique_sample_num_per_epoch: 4` 或 `8`
7. `max_epochs: 2`

要验证的内容：

1. 能正常完成 `sample -> prepare_feedback -> optimize`
2. `shadow` 返回非空分数
3. advantage 有波动，不是全零
4. LoRA 参数发生变化

#### 5.2 Short Run

建议配置：

1. `num_processes: 4`
2. `max_dataset_size: 256` 或 `512`
3. `resolution: 512`
4. `num_inference_steps: 6`
5. `group_size: 4`
6. `unique_sample_num_per_epoch: 32`
7. `max_epochs: 10`

要验证的内容：

1. reward 曲线没有明显塌缩
2. 保存和评估流程正常
3. 样图质量至少不比初始模型更差
4. 没有持续性 OOM、死锁、HTTP 超时风暴

本次实际执行结果（2-GPU fallback）：

1. 由于当前实际拓扑是“1 张卡训练 + 1 张卡承载 reward 服务”，本轮没有采用原计划中的 `num_processes: 4`。
2. 实际完成的是 `examples/grpo/lora/anima_anime_custom_short_2gpu.yaml`：
   - `num_processes: 1`
   - `max_dataset_size: 32`
   - `resolution: 256`
   - `num_inference_steps: 4`
   - `group_size: 4`
   - `unique_sample_num_per_epoch: 8`
   - `max_epochs: 1`
   - `eval_freq: 0`
3. 该 fallback short run 已成功跑完，并完成 reward 计算、参数更新与 checkpoint 保存。

#### 5.3 Formal Run

建议配置：

1. 使用完整训练集
2. `num_processes: 4`
3. `resolution: 512`
4. `num_inference_steps: 8` 或 `10`
5. `group_size: 4`
6. `unique_sample_num_per_epoch: 64`
7. `max_epochs: 50`

可选扩展：

1. 若 50 epoch 后 reward 和样图仍在持续改善，再追加到 `100`
2. 当前 formal run 仅依赖 `aesthetic_shadow`，不再依赖 `UnifiedReward-Flex`

当前状态：

1. formal run 的 YAML 已准备好，但本轮尚未执行。
2. 若继续推进，建议优先补一轮带评估图导出的中等规模 run，再决定是否直接进入 `50 epoch` 正式训练。

## 6. 记录与产出物

目标：满足“记录实施流程和结果，保存到文档”。

建议新增：

1. `guidance/anima_custom_training_report.md`

文档至少记录：

1. 使用的 commit id
2. 训练机器和 GPU 拓扑
3. 代理与环境变量
4. 最终使用的 YAML
5. 数据集转换规则
6. reward 服务启动方式和端口
7. smoke / short / formal 三轮结果
8. 样图路径
9. checkpoint 路径
10. 失败尝试与修正
11. 最终结论和后续建议

建议同时保留：

1. `saves/...` 下的 checkpoint
2. 训练日志
3. TensorBoard 或 WandB 曲线截图
4. 关键 epoch 的评估图片

## 7. 风险与回退方案

### 风险 1：reward 模型占满显存

原因：

1. reward 默认会在训练进程内各自加载

处理：

1. 坚持采用外置服务
2. 不让训练进程直接加载 `shadow`

### 风险 2：历史 `UnifiedReward-Flex` 路径推理过慢

原因：

1. 它是 pairwise 比较，复杂度随 `group_size` 上升很快
2. 即使切换到更快的 vLLM 环境，组内两两比较仍然是主要 wall-clock 开销

处理：

1. 当前活动训练方案已经移除该路径
2. 如后续重新启用，只建议放到评估阶段或离线对比阶段

### 风险 3：`shadow` 输出过于饱和

表现：

1. 大部分图片分数非常接近

处理：

1. 将分数从 `P(hq)` 改为 `hq-lq logit margin`
2. 必要时重新校准 `shadow` 的 reward 定义或 batch 设置

### 风险 4：原始 caption 不适合作为训练 prompt

表现：

1. prompt 过长
2. 带结构化标题
3. 包含无关作者或水印信息

处理：

1. 首轮仅移除 `Artist` 段，不改动其余 Markdown 结构
2. 若后续发现 `Texts` 或其他段落明显干扰训练，再单独评估是否追加裁剪规则
3. 保留原始映射文件，保证可回退

## 8. 最终验收清单

完成标准：

1. [x] 数据转换脚本已完成，本地可生成 `train.jsonl` / `test.jsonl`，且生成物不入库。
2. [x] `shadow` reward 服务可正常打分。
3. [x] `examples/grpo/lora/anima_anime_custom.yaml` 可直接用于 `ff-train`。
4. [x] smoke test 中出现非零参数更新。
5. [x] short run 已完成，reward 没有塌成全零；但本轮未输出评估样图。
6. [ ] formal run 成功保存 checkpoint、日志和评估结果。
7. [x] `guidance/anima_custom_training_report.md` 已记录已完成流程、失败修复与后续建议。

## 9. 推荐执行顺序

严格按下面顺序推进：

1. 预检环境与 GPU 拓扑
2. 仅移除 `Artist` 段并拆分 `anime10k.json`
3. 单独跑通 `shadow` 服务
4. 生成专用 YAML
5. 跑 smoke test
6. 跑 short run
7. 跑 formal run
8. 汇总结果并写报告

这个顺序的原因很简单：

1. 当前最大不确定性已经收敛到 `shadow` reward 的稳定性和训练规模本身。
2. 只保留单 reward 后，Anima 训练主链路更简单，也更容易扩展到正式训练。
3. 先缩小问题面，再放大训练规模，风险最低。
