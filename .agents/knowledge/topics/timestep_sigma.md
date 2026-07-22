# Timestep & Sigma Convention

**Read when**: Touching `TimeSampler`, `adapter.forward(t=...)`, `scheduler.step(timestep=...)`, `flow_match_sigma()`, or `timestep_range` config fields.

---

Throughout the codebase, two related but distinct scales are used for time:

| Name | Variable | Scale | Meaning |
|------|----------|-------|---------|
| **Timestep** | `t`, `timestep` | `[0, 1000]` | Scheduler-scale time. All public interfaces (`TimeSampler` outputs, `adapter.forward(t=...)`, `scheduler.step(timestep=...)`) use this scale. |
| **Sigma** | `σ`, `sigma` | `[0, 1]` | Flow-matching noise level. Used for latent interpolation `x_t = (1-σ) x_0 + σ ε` and loss weighting. Obtained via `flow_match_sigma(t) = t / 1000`. |

## Rules

1. `TimeSampler` always returns `t` in `[0, 1000]`. Trainers pass it directly to `adapter.forward(t=...)` without scaling.
2. When interpolating latents or computing noise-level-dependent weights, convert explicitly: `sigma = flow_match_sigma(t)`.
3. Each model adapter internally converts `t` to whatever its underlying transformer expects (e.g., Flux divides by 1000, SD3.5 passes as-is). This conversion is encapsulated inside the adapter's `forward()` method.
4. `timestep_range=(frac_lo, frac_hi)` is a fraction along the denoising axis from 1000 (noisy) toward 0 (clean), mapped via `t = 1000 * (1 - frac)`. So `(0, 0.99)` corresponds to `t ∈ [10, 1000]`.

## Gotchas

1. **Don't divide `t` by 1000 before passing to `adapter.forward()`** — the adapter handles internal conversion.
2. **`timestep_range` uses denoising-axis fractions, not raw timesteps** — `(0, 0.5)` means the noisier half `t ∈ [500, 1000]`, not the cleaner half.
3. **`flow_match_sigma()` is the only sanctioned conversion** — do not use `t / 1000` directly; use the function for traceability.
4. **Anima inference requires an explicit raw sigma grid** — `sd-scripts` constructs
   `linspace(1, 0, num_inference_steps + 1)` and then applies `discrete_flow_shift`. Calling
   Diffusers `set_timesteps(num_inference_steps=...)` without explicit sigmas uses a different
   grid. Pass the terminal-excluded raw sigmas to `set_timesteps(sigmas=...)`; the scheduler
   appends the terminal zero and applies its configured shift.

## Recorded Fix Patterns

### [Anima Inference Must Supply the sd-scripts Raw Sigma Grid]
- **Date**: 2026-07-23
- **Symptom**: Setting `model.discrete_flow_shift: 3.0` did not reproduce the timestep and sigma sequence from `sd-scripts --flow_shift 3` inference.
- **Root Cause**: The Anima adapter delegated raw sigma construction to Diffusers, whose default `set_timesteps(num_inference_steps=...)` grid differs from the linear terminal-inclusive grid used by `sd-scripts`.
- **Fix**: Updated `src/flow_factory/models/anima/anima.py` to pass terminal-excluded raw linear sigmas explicitly before the scheduler applies `discrete_flow_shift`, and documented the inference mapping in `guidance/anima_support.md`.
- **Lesson**: Matching a scheduler shift parameter is insufficient for inference parity; the pre-shift sigma grid and terminal convention must also match the reference pipeline.
- **Related Constraint**: N/A

## Cross-refs

- `constraints.md` #7 (coupled/decoupled paradigm — affects which timestep sampling is valid)
- `topics/train_inference_consistency.md` (same `t` must produce same output in rollout vs training)
- `topics/adapter_conventions.md` (adapter encapsulates timestep-to-model conversion)
