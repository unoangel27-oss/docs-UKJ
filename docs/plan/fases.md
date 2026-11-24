# UKJ + Cabrita — Phase plan

> Breakdown by real maturity: functional prototype → iterations → mature product.

## Phase 0 — Functional prototype (MVP)

- **Objective:** test the idea in motion — convert a real MoE to UKJ and run it in Cabrita with routes, tics and operators working.
- **Closure criteria (checklist):**
  - [ ] OLMoE converted to a valid `.ukj` (header + checksum OK).
  - [ ] Router activates the right routes; only activated routes are loaded.
  - [ ] RAM in small models absurdly low (measured, per route).
  - [ ] Parameter→subparameter connection works (ids resolve).
  - [ ] Tics work (thresholds 2/s and 20/s) and `stop-subparameter` does hot offload.
  - [ ] `continue parameter` works (definition 46).
  - [ ] System prompt injectable; ≥3 densities (`p1_p`, `p3_p`, `p5_p`) convertible.
  - [ ] Metrics visible (tokens/s, RAM, latency).

## Phase 1 — Iterations (validation, errors, stabilization, key modules)

- **I1 — Error handling:** corruption/truncation detection, repair with checkpoints, backup of worlds + restore + light retraining, missing-world messages, incomplete-mode warning.
- **I2 — Stabilization and benchmarks:** public benchmark (tokens/s, RAM, latency), quality table per `q*` code, `>p8_p` degradation experiment, audit via `cabrita inspect --log`.
- **I3 — Key modules:** OpenAI-compatible HTTP server + Cabrita-embedded GUI, distribution `.exe`/AppImage, world editor (open world → deploy weights).
- **Dependencies:** Phase 0 closed.
- **Closure criteria:** I1–I3 with tests; no regressions on the Phase 0 checklist.

## Phase 2 — Mature product (scalability and delivery)

- **T1 — toji + `.mem`:** hybrid context window (traditional for the present + persistent memory `.mem`), light memory-writing training, neurons/satellites/sectors.
- **T2 — Scalability:** sharding v2 (reserved "parts" field in header), macOS, batching and several models in parallel.
- **T3 — Delivery:** public spec + audit (@Auditor with `$audit-context-building` and `$variant-analysis`), MkDocs published (@Documentador), single `cabrita` repo with spec in `docs/` (@Gestionador), full-stack web app **TypeScript + HTML + CSS** on Vercel (@Web Developer, when there is a solid prototype), video/TikTok (@Scriptwriter & Editor + @Gestionador).

## Work split

| Stage | Owner | Medium |
|---|---|---|
| Spec and plan | @Ideador & Planificador | this folder |
| MkDocs documentation | @Documentador | docs/ of the repo |
| Converter, Cabrita, modules | @Developer | Linear issues |
| Spec and product audit | @Auditor | review before publishing |
| GitHub repo | @Manager | upload after audit |
| Web app + Vercel | @Web Developer | after solid prototype |
| Video and TikTok | @Scriptwriter & Editor + @Manager | project summary |
| Light tasks | @General helper | on demand |

## Project rules

- The whole project lives in `/home/angel/Documentos/ukj-cabrita/`, never in work spaces.
- Warn the full flow at the start of each stage.
- Flow: User ↔ Ideador & Planificador → Documentador → Developer → Auditor → Manager → Scriptwriter & Editor → Manager (TikTok).
