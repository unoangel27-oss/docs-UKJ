# UKJ + Cabrita — Specification v1

> Source: 120-question questionnaire (round 1), 75 questions (round 2) and the contradictions resolved on 2026-11-22.
> Status: **v1 for user approval and review before publishing.**

---

## 0. Identity

- **UKJ** = Universal Kernelized Joint-weights. New model format, extension `.ukj`, independent from GGUF/safetensors, with bidirectional conversion.
- **Cabrita** = runtime in Rust that executes UKJ. Local, offline, no telemetry, OpenAI-compatible HTTP API, MIT license.
- **Phase 1: MoE only.** It is a phase restriction, not the format's: the format accepts other architectures in the future (Mamba, diffusion, etc.).

## 1. Files

- **`.ukj`** — the single distributable file: **one file = one model**. Container (zip-like, custom extension) with: weights, parameters, **subparameters (dormant)**, routes/worlds, operators, density declarations and metadata.
- **`.subs`** — working copy of the subparameters, **one per model**, in Cabrita's system folder (e.g. `~/.cabrita/`). Materialized on install/convert of the `.ukj`. If missing or out of sync: **Cabrita repairs** it by regenerating it from the `.ukj`.

- **Incomplete mode:** a `.ukj` without a materialized `.subs` runs with parameters alone: possible, but incomplete (empty of route knowledge). Cabrita warns. It is the fallback of "world not found" (degraded option, not default).
- **Updates:** only UKJ sections are rewritten (metadata, routes, operators, densities); weights do not change. **Progressive checkpoints**; mid-failure → rollback to last checkpoint.

## 2. Hierarchy and routes

- **Two levels:** parameter → subparameters. Each parameter carries **chained ids** that invoke its subparameters (in `.subs`).
- Two nodes with the same name → **merge** into a single node.

- **Routes/worlds:** separate layer (not part of the hierarchy), **they live in the `.ukj`**. Taxonomy **manually labeled** when translating each MoE. A **tiny router** (slightly trained) picks the world for each request.
- **Loading:** only the **activated** routes by the router — one if enough, several if the request needs them (with priorities), plus **provisional worlds** ("just in case"). **Never** the whole model.

## 3. Operators

The two operators are **independent** (they do not form a pair; the reversibility "stop and continue later" is a property of the mechanism, not of the relationship between operators).

- **`stop-subparameter`:** restricts subparameter deployment; subparameters deploy only when the **tics** level reaches the anomalous threshold. Allows **hot offload** and **total jump** of a complete substructure (parameter + all its subparameters).
- **`continue parameter`:** definition 46 — distribute irregular parameters (without perfect quantization) into independent parameters **without subparameters**.

- **Who decides:** dynamically, **Cabrita** decides in hot (not a static rule of the file).
- **Tics:** continuous activation that, when repeated until the threshold, forces releasing subparameters. Defaults in spec: **2/s normal, 20/s anomalous**; Cabrita can calibrate per hardware.
- **Conditional operators (phase 1):** each operator accepts a condition: memory threshold or route match.
- **Names:** official as-is (`stop-subparameter`, `continue parameter`); the guion/space difference is frozen in the spec.
- **More operators:** the header reserves an **extensible table**; `skip`, `resume`, `override`, `reset` remain reserved, not implemented in phase 1.
- **Audit:** session log (timestamp, operator, target, tics level) visible with `cabrita inspect --log`.
- **Validity:** a stop **does not** leave the model unusable — subparameters are never deleted from `.subs`, only not deployed, and deployment is always reversible. Validator: **the file declares schema version and Cabrita validates on load**; invalid file → rejection with message, never loads halfway.
- **Break vs GGUF:** structural = **subparameters + routes**; control mechanism = **operators** (they make them usable in hot).

## 4. `p_p` Quantization

- The official term is **quantization `p_p`** (that is what the user calls it; "density" remains a synonym). **It measures subparameters per parameter, not bits.** `p5_p` = 5 subparameters per parameter.
- `p_p` = official alias of `p1_p`. **`p0_p` does not exist** (zero subparameters = dense model without UKJ). Phase 1: `p1_p`–`p7_p`; top `p8_p`.
- **Scope:** per tensor (parameter node); inherits the model default if not declared. **Mix allowed** (`p5_p` in one part, `p2_p` in another of the same file). A route can override the density of its subtree (`route/capa3/p_p`).
- **`q*` = bits per weight with inherited GGUF semantics** (q4_0 = 4, q4_k = 4, q5_* = 5, q8_0 = 8…). **`q*` is orthogonal to `p_p`.** GGUF blocks (group size, scales, zero points) only for `q*`; `p_p` does not use blocks.
- **Requantize in flight: no** in phase 1 (offline conversion only; reserved for v2).
- **Quality/performance table by code:** generated by the tooling (Python) and published in the documentation; benchmarks in team CI.
- **Metadata in the header** (before tensors): magic, version, flags, model info, routes table, operators table, density declarations, tensor index (offsets) and checksums. **Mandatory minimum header** (magic, version, checksum, ids, tables); tensor body may vary by model.
- **The file name includes** the quantization.
- **Degradation above `p8_p`:** measured in F1.2 on OLMoE q2_K (`p1_p`…`p12_p`, 512-token ctx): perplexity 20.870 and short QA 80.0% in all twelve variants, because `p_p` splits subparameters without changing a single weight. No quality threshold triggers by `p12_p`; the cost grows with every step (195 subparameters more per variant, 2,340 in `p12_p`) with no gain, so **`p8_p` remains the official ceiling** and `p9_p`–`p12_p` stay experimental by cost, not by quality.

### 4.1 Official definition of `p1_p` (D54) — approved by the user (2026-11-23)

> `p1_p` is the density declaration that assigns one subparameter to each parameter of the scope it applies to.

- **Unit:** count ratio *subparameters ÷ parameters*, **not bits**. `p1_p` = 1:1, `p5_p` = 5:1.
- **Scope:** declared per tensor (parameter node); if it is not declared, it inherits the model default; a route can override the density of its subtree (`route/capa3/p_p`).
- **Effective reach:** only the tensors under the declared scope; the rest of the file keeps its own density (free mixing).
- **Alias:** `p_p` means exactly `p1_p`, with no additional nuance.
- **Lower bound:** `p0_p` does not exist: 0 subparameters = model without the UKJ extension (pure dense), which is another state of the file, not a density on the scale.
- **Orthogonality:** `p1_p` does not fix bits per weight; the bits remain `q*` (q4_0, q4_k, q8_0…) with inherited GGUF semantics.
- **File registration:** the declaration lives in the header (density declarations table), before the tensors.

## 5. Cabrita Runtime — phase 1

- **Context:** traditional context window. `toji` + `.mem` = phase 2.
- **Inference:** prefill + decode + streaming. Backends: **CPU and Vulkan**. Platforms: Windows (`.exe`) and Linux (AppImage); macOS later.
- **Loading:** streaming without loading all into RAM, per route; `.subs` dormant until activation. A 200 GB model does not consume 200 GB: only the activated piece.
- **Metrics:** tokens/s, RAM, latency; public benchmark in F1.2.

## 6. Decisions taken (the 21 blanks + pending)

| # | Question | Decision |
|---|---|---|
| D35 | Hot offload | **Yes** |
| D36 | Total jump | **Yes** |
| D37 | Names | **As-is** |
| D38 | More operators | Extensible table reserved; `skip`/`resume`/`override`/`reset` not in phase 1 |
| D39 | Conditional operators | **Yes, minimal** |
| D40 | Audit | Session log + `cabrita inspect --log` |
| D41 | Invalid state | **Impossible** |
| D42 | Validator | **Both** |
| D43 | Real break | Structure = subparameters + routes; control = operators |
| D44 | `p_p` without number | Alias of `p1_p` |
| D45 | `p0_p` | **Does not exist** |
| D46 | Scope | Per tensor (node), inherits model default |
| D47 | Mix | **Yes**, free mix in the same file |
| D48 | Bits | `q*` with inherited GGUF semantics; `p_p` orthogonal |
| D49 | Requantize | **No** in phase 1 (offline); reserved for v2 |
| D50 | Table | **Yes**, generated by Python tooling in the docs, benchmarks in CI |
| D51 | Blocks | GGUF inherited only for `q*` |
| D52 | Metadata | `.ukj` header (see §4) |
| D53 | `p_p` in routes | **Yes**, overrides subtree |
| D54 | Spec | Drafted `p1_p`, **user approves**, review before publishing |
| D55 | Above `p8_p` | Experimental by cost; no quality loss up to `p12_p` (F1.2) |
| M4 | Schema | **Yes**: mandatory minimum header |
| M5 | Owner | Spec → published docs; implementation separate; layout neutral |
| M6 | Tic threshold | Defaults in spec (2/s, 20/s), calibratable by Cabrita |
| — | Same-name nodes | **Merge** (round-1 answer retained) |

## 7. Pending items requiring the user

Resolved:

1. ~~**Approval of the official definition of `p1_p`** (D54) before publishing the specification.~~ — approved 2026-11-23 (§4.1).
2. ~~**Confirmation of the reading of `.subs`**: content inside the `.ukj` + working copy extracted in Cabrita's folder (§1).~~ — confirmed 2026-11-23 (§1).

Open:

3. ~~**Degradation threshold above `p8_p`** (D55).~~ — set by F1.2 on 2026-11-25: `p1_p`…`p12_p` give identical perplexity and QA, so `p9_p`–`p12_p` stay experimental by cost and `p8_p` remains the ceiling (§4).
