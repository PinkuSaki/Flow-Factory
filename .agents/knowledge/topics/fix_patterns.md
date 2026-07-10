# Fix Patterns

**Read when**: After completing a bug fix.

---

This document defines the recording template and archival rules for fix experiences.

## Fix Entry Template

Each fix record uses the following format:

```markdown
### [Short Title]
- **Date**: YYYY-MM-DD
- **Symptom**: What the user observed (error message / abnormal behavior)
- **Root Cause**: Root cause analysis (one sentence)
- **Fix**: What was changed (files involved and key modifications)
- **Lesson**: Implications for future development (why this happened, how to prevent it)
- **Related Constraint**: If a new hard constraint was created, reference the constraint number (N/A if none)
```

## Archival Location Decision Table

Based on the fix type, write the fix entry to the appropriate document:

| Fix Type | Archival Location | Example |
|----------|------------------|---------|
| Violated an existing constraint | `constraints.md` — add "common violation case" under the relevant entry | Forgot to update registry path |
| Discovered a new hard constraint | `constraints.md` — new entry | Found ZeRO-2 + EMA incompatibility |
| Architecture / data-flow misunderstanding | `architecture.md` — relevant module section | Misunderstood preprocess_func call timing |
| Subsystem-specific pitfall | `topics/<topic>.md` — corresponding topic | Sampler boundary condition |
| Does not fit any of the above | This document's "Recorded Fix Patterns" section below | Append as a new record |

**Decision flow**: Check whether the fix matches the first four rows; if none match, fall back to this document.

## Recorded Fix Patterns

<!-- This section accumulates over time. Append new records at the end using the template above. -->

### [Anima BF16 Timestep Dtype Mismatch]
- **Date**: 2026-04-11
- **Symptom**: Anima BF16 inference failed inside `t_embedder.linear_1` with `expected mat1 and mat2 to have the same dtype, but got: c10::Half != c10::BFloat16`.
- **Root Cause**: The Anima adapter derived timestep dtype from stored trajectory latents, which could be downcast to FP16 by `latent_storage_dtype`, while the transformer weights stayed in BF16.
- **Fix**: Updated `src/flow_factory/models/anima/anima.py` so transformer-facing tensors use the transformer's device and dtype explicitly, and updated Anima example configs to store trajectory latents in BF16.
- **Lesson**: For rollout-to-train pipelines, model-input dtype must be derived from the active module weights, not from intermediate storage tensors that may use a cheaper archival precision.
- **Related Constraint**: N/A

### [Cached Components Are Not Always Accelerator-Prepared]
- **Date**: 2026-04-11
- **Symptom**: `adapter.on_load(cuda)` did not move LoRA-wrapped transformers to GPU after `load_model(...)`, causing direct inference paths to leave the trainable component on CPU.
- **Root Cause**: `BaseAdapter` used `_components` both as a cache for replaced modules and as the marker for accelerator-managed modules, so any cached LoRA component was incorrectly skipped by device-loading logic.
- **Fix**: Updated `src/flow_factory/models/abc.py` and `src/flow_factory/trainers/abc.py` to track accelerator-prepared components separately via `_prepared_components`, and only skip manual device management for components that were actually passed through `accelerator.prepare(...)`.
- **Lesson**: Module replacement and distributed wrapping are different states; caching a component must not imply that Accelerate owns its device placement.
- **Related Constraint**: N/A

### [Anima Evaluation Can Legitimately Omit Trajectory Latents]
- **Date**: 2026-04-11
- **Symptom**: An Anima GRPO short run crashed during evaluation before training with a `NoneType` failure while stacking `all_latents`.
- **Root Cause**: `AnimaAdapter.inference()` assumed the latent collector always produced trajectory latents, but evaluation paths may pass `trajectory_indices=None` and intentionally disable latent collection.
- **Fix**: Updated `src/flow_factory/models/anima/anima.py` so `all_latents` stays `None` when the collector is disabled instead of being unconditionally stacked.
- **Lesson**: Adapter outputs must preserve optional trajectory fields across both rollout and evaluation paths; assuming rollout-only data in shared code breaks train-eval parity.
- **Related Constraint**: N/A

### [Loopback Reward Services Must Bypass Environment Proxies]
- **Date**: 2026-04-11
- **Symptom**: Local reward service calls were at risk of being routed through global proxy environment variables, causing connection failures or unnecessary indirection.
- **Root Cause**: The training-side `requests` client and the bridge-side `httpx` / OpenAI client inherited environment proxy settings even for `127.0.0.1` endpoints.
- **Fix**: Updated `src/flow_factory/rewards/my_reward_remote.py` and `scripts/reward_servers/unifiedreward_flex_bridge.py` to disable environment proxies for loopback targets.
- **Lesson**: Local control-plane traffic should never depend on external proxy configuration; loopback endpoints need explicit proxy bypass in all HTTP clients.
- **Related Constraint**: N/A

### [Pairwise Judge Parsing Must Tolerate Verbose or Truncated Winners]
- **Date**: 2026-04-11
- **Symptom**: The UnifiedReward-Flex bridge returned `500` errors on long Markdown prompts because the judge sometimes emitted verbose string winners or truncated JSON.
- **Root Cause**: The bridge expected a narrow winner schema and forwarded unbounded prompt text, which increased the chance of echoed prompts and clipped JSON payloads.
- **Fix**: Updated `scripts/reward_servers/unifiedreward_flex_bridge.py` to normalize and truncate prompts, increase the completion budget, and parse integer, string, and partially truncated `winner` fields.
- **Lesson**: LLM-based control outputs need bounded prompts and schema-tolerant parsing; production bridges should handle minor format drift without collapsing the training job.
- **Related Constraint**: N/A

### [Distributed Eval Logging Must Gather Samples]
- **Date**: 2026-04-28
- **Symptom**: TensorBoard logged only the main-rank subset of evaluation samples during multi-process training.
- **Root Cause**: Trainer evaluation loops gathered reward tensors for global statistics but passed rank-local `all_samples` to the logger.
- **Fix**: Added a lightweight eval sample gather helper in `src/flow_factory/trainers/abc.py` and used it in GRPO, DPO, NFT, and AWM evaluation before main-process logging.
- **Lesson**: Distributed eval logging needs separate sample gathering; metric gathering alone does not make media logs global.
- **Related Constraint**: N/A

### Multi-modal batch homogeneity (R6)
- **Date**: 2026-04
- **Symptom**: HF `Dataset.map` produced inconsistent per-sample modality types and batch-length mismatches when a sample had no media item.
- **Root Cause**: `_preprocess_batch` mixed `None`, tensors, and tensor lists in the same Arrow column.
- **Fix**: Standardized each modality column as `List[List[Media]]`, using an empty list for missing media, and aligned adapter preprocessing and audio aliases with that representation.
- **Lesson**: Variable-cardinality Arrow columns need one homogeneous representation for empty, single, and multiple media values.
- **Related Constraint**: N/A

### Non-abstract encoder defaults (R7)
- **Date**: 2026-04
- **Symptom**: Adding audio support initially required no-op encoder stubs in every adapter that did not consume audio.
- **Root Cause**: Optional modality encoders were incorrectly treated as mandatory abstract methods.
- **Fix**: Made prompt, image, video, and audio encoders optional no-op defaults while keeping universal adapter operations abstract.
- **Lesson**: Partial-coverage extension points should use opt-in defaults; reserve abstract methods for contracts every subclass must implement.
- **Related Constraint**: #12

### [Remote Reward Routing Must Filter Before Serialization]
- **Date**: 2026-07-11
- **Symptom**: After merging multi-source reward routing, remote pointwise rewards could serialize and score samples from datasets outside `applicable_datasets`.
- **Root Cause**: The remote `per_rank` and `main_process` dispatch paths bypassed the source-aware gate used by local pointwise rewards.
- **Fix**: Updated `src/flow_factory/rewards/reward_processor.py` to mark applicability on local samples, serialize only applicable sub-batches, scatter finite scores back into their original positions, and keep non-applicable positions as `NaN` in both dispatch modes.
- **Lesson**: Alternative execution backends must share the same routing semantics as the canonical local path; filtering must happen before expensive serialization or model calls.
- **Related Constraint**: #13

### [Anima CFG Preprocessing Must Follow Guidance Scale]
- **Date**: 2026-07-11
- **Symptom**: Anima preprocessing always encoded negative prompts even when CFG was disabled, and a direct `forward()` with incomplete negative conditioning silently skipped requested CFG.
- **Root Cause**: The adapter retained an older `do_classifier_free_guidance` parameter instead of deriving CFG from the user-facing `guidance_scale` used by the unified adapter contract.
- **Fix**: Updated `src/flow_factory/models/anima/anima.py` so `encode_prompt()` derives CFG from `guidance_scale > 1.0`, inference passes the scale directly, and `forward()` warns before falling back when negative conditioning is incomplete.
- **Lesson**: Prompt preprocessing and denoising must derive CFG from the same public field and threshold so cached conditioning matches runtime behavior.
- **Related Constraint**: #12

## Cross-refs

- `constraints.md` (archival target for constraint violations)
- `architecture.md` (archival target for data-flow misunderstandings)
- `ff-debug/SKILL.md` Phase 5 (knowledge capture workflow)
