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

## Cross-refs

- `constraints.md` (archival target for constraint violations)
- `architecture.md` (archival target for data-flow misunderstandings)
- `ff-debug/SKILL.md` Phase 5 (knowledge capture workflow)
