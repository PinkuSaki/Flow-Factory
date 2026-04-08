# Anima 模型接入与训练适配计划

## 1. 背景

目标是在 Flow-Factory 中新增 `anima` 模型类型，并完成训练链路的全面适配，使其能够在当前框架下支持：

- `grpo` / `nft` / `awm` 三类训练器
- `lora` 与 `full` 两种微调方式
- 训练、评估、断点恢复、保存导出
- 现有 reward / sample / distributed preprocessing 工作流

本计划以本机 `~/sda1/sd-scripts` 中的 Anima 相关实现为参考，重点参考以下文件：

- `anima_train.py`
- `anima_train_network.py`
- `library/anima_models.py`
- `library/anima_utils.py`
- `library/anima_train_utils.py`

## 2. 现状判断

### 2.1 Flow-Factory 当前的扩展方式

当前仓库的新模型接入入口已经比较清晰：

- 模型注册与加载：
  - `src/flow_factory/models/registry.py`
  - `src/flow_factory/models/loader.py`
- 统一适配器基类：
  - `src/flow_factory/models/abc.py`
- 训练器主流程：
  - `src/flow_factory/trainers/abc.py`
  - `src/flow_factory/trainers/grpo.py`
  - `src/flow_factory/trainers/nft.py`
  - `src/flow_factory/trainers/awm.py`
- 数据预处理与缓存：
  - `src/flow_factory/data_utils/dataset.py`
  - `src/flow_factory/data_utils/loader.py`

因此，Anima 接入不需要重写训练框架，重点是补齐一个符合 `BaseAdapter` 规范的自定义模型运行时。

### 2.2 Anima 与现有模型的关键差异

Anima 不能简单复用现有 diffusers pipeline 适配逻辑，原因包括：

- 没有现成的 diffusers `Pipeline.from_pretrained(...)` 入口
- 文本条件链路不是常规的单路 text encoder 输出，而是：
  - Qwen3 tokenizer / text encoder
  - T5 tokenizer 输入
  - 通过 Anima DiT 内部 `_preprocess_text_embeds(...)` 构造 cross-attention 条件
- DiT 前向输入是 5D latent：`[B, C, T, H, W]`
- 图像任务虽然是单帧 T2I，但运行时仍带有 frame 维
- 参考实现存在 Anima 专属训练能力：
  - 独立 param group 学习率
  - 自定义 gradient checkpointing / block swap / offload
  - 自定义 full checkpoint 加载逻辑

结论：Anima 更接近“自定义运行时模型”，不是“套一层现成 diffusers pipeline”即可完成。

## 3. 总体设计原则

### 3.1 接入策略

- 不把 `sd-scripts` 作为 Flow-Factory 的运行时依赖
- 仅参考其实现，在 Flow-Factory 内部落地最小可维护版本
- 优先打通 Flow-Factory 的 RL 训练闭环，再补 sd-scripts 的高级省显存特性

### 3.2 复用策略

优先复用的部分：

- Flow-Factory 的 trainer 主流程
- `BaseAdapter` 的 LoRA、EMA、checkpoint 管理框架
- `GeneralDataset` 的 preprocess + cache 机制
- 现有 reward / sample / distributed gather 逻辑

需要单独实现或重写的部分：

- Anima 运行时 pipeline
- Anima 文本条件编码
- Anima 的 `inference()` / `forward()`
- Anima 专属 optimizer param groups
- Anima 的 full checkpoint 存取

## 4. 实施范围

### 4.1 必做范围

1. 新增 `model_type: anima`
2. 新增 `AnimaAdapter`
3. 打通 `grpo + lora`
4. 打通 `grpo + full`
5. 支持基础 eval / save / resume
6. 补示例 YAML 与接入文档

### 4.2 “全面适配训练”的建议范围

在必做范围之外，建议一并完成：

1. `nft` 训练 smoke test
2. `awm` 训练 smoke test
3. 独立 param groups 学习率
4. Anima 专属配置项显式化

### 4.3 可延期范围

以下能力建议放在第二阶段或第三阶段：

- `cpu_offload_checkpointing`
- `unsloth_offload_checkpointing`
- `blocks_to_swap`
- 与 sd-scripts 完全等价的 text encoder / latent cache 细节
- markdown section dropout 等 Anima 专属数据增强

## 5. 文件级改动计划

### 5.1 模型与运行时

建议新增目录：

- `src/flow_factory/models/anima/`

建议新增文件：

- `src/flow_factory/models/anima/anima.py`
  - `AnimaAdapter`
  - `AnimaSample`
- `src/flow_factory/models/anima/pipeline_anima.py`
  - 轻量级 `AnimaPipeline`
  - 持有 `tokenizer`、`text_encoder`、`vae`、`transformer`、`scheduler`
- `src/flow_factory/models/anima/anima_utils.py`
  - Anima 模型加载
  - Qwen3 tokenizer / text encoder 加载
  - 权重 key 处理

如需进一步拆分，可补：

- `src/flow_factory/models/anima/anima_train_utils.py`
  - param groups
  - loss weighting
- `src/flow_factory/models/anima/anima_modeling.py`
  - 若直接内嵌参考实现中的 DiT 结构

### 5.2 配置与注册

需要修改：

- `src/flow_factory/hparams/model_args.py`
  - 增加 `model_type: "anima"`
  - 增加 Anima 相关字段或先通过 `extra_kwargs` 承接
- `src/flow_factory/models/registry.py`
  - 注册 `anima -> flow_factory.models.anima.anima.AnimaAdapter`

### 5.3 训练器与优化器

建议修改：

- `src/flow_factory/trainers/abc.py`

原因：

- 当前 `_init_optimizer()` 只接受 `self.adapter.get_trainable_parameters()`
- Anima 需要按模块族划分参数组：
  - `base`
  - `self_attn`
  - `cross_attn`
  - `mlp`
  - `mod`
  - `llm_adapter`

建议抽象：

- 在 `BaseAdapter` 新增可覆盖接口，例如：
  - `get_optimizer_param_groups()`
- 默认返回普通参数列表
- Anima 覆盖该接口返回 param groups
- Trainer 根据返回值统一构建 `AdamW`

### 5.4 示例与文档

建议新增：

- `examples/grpo/lora/anima.yaml`
- `examples/grpo/full/anima.yaml`
- `examples/nft/lora/anima.yaml`
- `examples/awm/lora/anima.yaml`

如果首轮只做最小闭环，至少补：

- `examples/grpo/lora/anima.yaml`
- `examples/grpo/full/anima.yaml`

## 6. 分阶段实施计划

### 阶段一：最小运行时打通

目标：完成 `anima + grpo + lora` 的最小闭环。

任务：

1. 新增 `anima` 模型注册
2. 新建 `AnimaPipeline`
3. 新建 `AnimaAdapter`
4. 实现以下核心接口：
   - `load_pipeline()`
   - `encode_prompt()`
   - `decode_latents()`
   - `inference()`
   - `forward()`
5. 补最小示例配置 `examples/grpo/lora/anima.yaml`

阶段验收标准：

- 能启动 `ff-train`
- 能完成采样
- reward 能消费生成图像
- 能进行反向传播
- 能保存 LoRA checkpoint

### 阶段二：补齐 full finetune 与恢复能力

目标：完成 `grpo + full`，并保证 checkpoint 可用。

任务：

1. 实现 Anima full checkpoint 保存
2. 实现 Anima full checkpoint 加载
3. 对齐 `resume_path` / `resume_type`
4. 补 `examples/grpo/full/anima.yaml`

阶段验收标准：

- full 训练可启动
- full checkpoint 可保存
- 从保存点恢复后继续训练不报错

### 阶段三：优化器参数组与配置显式化

目标：补齐 Anima 的训练可控性。

任务：

1. 在 `hparams` 中补以下配置：
   - `qwen3`
   - `vae`
   - `t5_tokenizer_path`
   - `attn_mode`
   - `split_attn`
   - `vae_chunk_size`
   - `vae_disable_cache`
   - `self_attn_lr`
   - `cross_attn_lr`
   - `mlp_lr`
   - `mod_lr`
   - `llm_adapter_lr`
2. 在 adapter 中实现 param group 划分
3. 在 trainer 中支持 param groups 初始化 optimizer

阶段验收标准：

- 不同参数组可以使用不同学习率
- 配置通过 YAML 显式传递
- 冻结某组参数时训练行为符合预期

### 阶段四：扩展到 NFT / AWM

目标：验证 Anima 对 Flow-Factory 现有训练算法的兼容性。

任务：

1. 用 `nft` 配置做 smoke test
2. 用 `awm` 配置做 smoke test
3. 校验 `forward()` 输出满足三类 trainer 的需要：
   - `noise_pred`
   - `log_prob`
   - `next_latents`
   - `next_latents_mean`
   - `dt`

阶段验收标准：

- `nft` 可跑最小训练
- `awm` 可跑最小训练
- 不需要针对 trainer 写 Anima 特殊分支

### 阶段五：高级省显存与数据特性补齐

目标：向 sd-scripts 的训练体验靠拢。

任务：

1. 补 `blocks_to_swap`
2. 补 `cpu_offload_checkpointing`
3. 补 `unsloth_offload_checkpointing`
4. 评估是否需要接入：
   - text encoder outputs cache
   - latent cache
   - markdown section dropout

说明：

- 这一阶段不是打通训练闭环的前置条件
- 应在主流程稳定后逐项补充

## 7. Adapter 设计细节

### 7.1 `AnimaPipeline` 设计

建议让 `AnimaPipeline` 至少暴露以下属性，保持与 `BaseAdapter` 的组件发现逻辑兼容：

- `tokenizer`
- `text_encoder`
- `vae`
- `transformer`
- `scheduler`

同时提供最小配置对象或属性，保证以下能力可工作：

- tokenizer 解码 prompt ids 给 reward 使用
- VAE 编解码
- transformer 前向
- scheduler 替换与训练模式切换

### 7.2 `encode_prompt()` 输出建议

Anima 的文本条件不是简单返回 `prompt_embeds` 即可，建议缓存并输出：

- `prompt_ids`
- `prompt_embeds`
- `source_attention_mask`
- `target_input_ids`
- `target_attention_mask`

原因：

- `prompt_ids` 用于 reward 侧解码
- `forward()` 需要完整条件才能调用 Anima DiT
- 离线 preprocess 缓存需要这些字段全部可序列化为 CPU tensor

### 7.3 `inference()` 设计

建议参考现有：

- `src/flow_factory/models/qwen_image/qwen_image.py`
- `src/flow_factory/models/wan/wan2_t2v.py`

实现时重点处理：

1. latent 初始化形状为 5D：`[B, C, 1, H, W]`
2. 文本条件需先走 `_preprocess_text_embeds(...)`
3. CFG 需要正负条件两路前向
4. 采样过程中要兼容 Flow-Factory 的 trajectory collector
5. 输出样本建议使用 `T2ISample`

说明：

- 从任务语义上看，Anima 在 Flow-Factory 中应按 T2I 模型接入
- 虽然内部 latent 带 frame 维，但首期不需要把它当 T2V 模型处理

### 7.4 `forward()` 设计

`forward()` 需要保证与 trainer 的接口契约一致。

最低要求：

- 输入：
  - `t`
  - `t_next`
  - `latents`
  - `next_latents`
  - `guidance_scale`
  - `compute_log_prob`
  - 已缓存的 prompt / attention 条件
- 输出：
  - `noise_pred`
  - `next_latents`
  - `next_latents_mean`
  - `log_prob`
  - `dt`

建议实现方式：

1. 调用 Anima transformer 获得 `noise_pred`
2. 调用 Flow-Factory 的 scheduler `step(...)`
3. 统一封装为 `SDESchedulerOutput`

## 8. 优化器与参数组计划

参考 `sd-scripts/library/anima_train_utils.py`，Anima 建议支持六组参数：

- `base`
- `self_attn`
- `cross_attn`
- `mlp`
- `mod`
- `llm_adapter`

第一版建议：

- `lora` 模式：沿用现有 LoRA 训练逻辑，不启用分组学习率
- `full` 模式：启用 param groups

第二版可进一步支持：

- `lora` 模式下按组件族区分 target modules
- 更细粒度的学习率与冻结控制

## 9. 配置字段建议

建议在 `model` 段增加如下字段：

```yaml
model:
  model_type: "anima"
  model_name_or_path: "/path/to/anima_dit.safetensors"
  qwen3: "/path/to/qwen3"
  vae: "/path/to/anima_vae"
  t5_tokenizer_path: "/path/to/t5_tokenizer"
  attn_mode: "torch"
  split_attn: true
  vae_chunk_size: null
  vae_disable_cache: false
```

建议在 `train` 段增加如下字段：

```yaml
train:
  self_attn_lr: null
  cross_attn_lr: null
  mlp_lr: null
  mod_lr: null
  llm_adapter_lr: null
```

说明：

- 若短期内不想修改 dataclass，也可以先让这些字段通过 `extra_kwargs` 透传
- 但从可维护性出发，最终应补为显式字段

## 10. 风险点

### 10.1 最大风险

Anima 不是标准 diffusers 模型，以下能力不能默认认为可复用：

- `from_pretrained()`
- `save_pretrained()`
- 通用 full checkpoint 保存
- 通用 LoRA target module 推断

### 10.2 次级风险

- Anima 文本条件构造错误会导致训练可跑但效果失真
- 5D latent 与现有 4D T2I 模型混用时，`forward()` / `decode_latents()` 容易出现维度错误
- param groups 如果直接塞进现有 trainer，需要注意与 `accelerator.prepare(...)` 的交互

### 10.3 控制策略

- 每个阶段都做 smoke test，不一次性堆完所有特性
- 先跑 `grpo + lora`，再补 full 和高级特性
- checkpoint 逻辑单独测试，不与训练主流程耦合排查

## 11. 验收清单

### 11.1 代码层

- [ ] `model_type: anima` 可正常加载
- [ ] `AnimaAdapter` 能完成 preprocess / inference / forward
- [ ] LoRA 保存与恢复可用
- [ ] full checkpoint 保存与恢复可用
- [ ] trainer 无需为 Anima 写硬编码特殊分支，或特殊分支最小化

### 11.2 运行层

- [ ] `python -m compileall src`
- [ ] `ff-train examples/grpo/lora/anima.yaml` 最小闭环通过
- [ ] `ff-train examples/grpo/full/anima.yaml` 最小闭环通过
- [ ] `ff-train examples/nft/lora/anima.yaml` smoke test 通过
- [ ] `ff-train examples/awm/lora/anima.yaml` smoke test 通过

### 11.3 功能层

- [ ] eval 可正常出图
- [ ] reward 可消费生成结果
- [ ] 断点恢复后继续训练不报错
- [ ] 多卡下 preprocess cache 与训练主循环正常

## 12. 推荐落地顺序

推荐按以下顺序推进：

1. `anima + grpo + lora`
2. `anima + grpo + full`
3. optimizer param groups
4. `nft` / `awm` smoke test
5. block swap / offload / cache 等高级特性

这样做的原因是：

- 先拿到最短可运行路径
- 把复杂度集中在 adapter 层，不提前扩散到 trainer 与 data 层
- 避免一开始就把 sd-scripts 的全部训练特性一并搬入，导致排错面过大

## 13. 结论

Anima 接入 Flow-Factory 的核心工作不是“新增一个模型名”，而是新增一套符合 `BaseAdapter` 契约的自定义运行时。最合理的实现路径是：

- 以 `QwenImageAdapter` 作为文本条件与 T2I 主模板
- 以 `Wan` 系列适配器作为 5D latent / scheduler step 参考
- 以 `sd-scripts` 的 Anima 代码作为运行时与参数组设计参考

只要先把 `AnimaAdapter`、optimizer param groups、checkpoint 三个点打稳，训练器主流程基本可以保持不变，Anima 就能进入 Flow-Factory 的统一训练框架。
