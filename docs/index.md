# Cabrita

**Cabrita** is the Rust runtime that executes the **UKJ** model format —
*Universal Kernelized Joint-weights*, extension **`.ukj`** — and its ecosystem of
subparameters, routes/worlds, operators and `p_p` quantization.

- **UKJ** is the format: a `.ukj` file = one model. It contains parameters,
  subparameters, routes/worlds, operators, densities and metadata.
- **Cabrita** is the runtime: local, offline, no telemetry, OpenAI-compatible HTTP
  API, MIT license.

!!! danger "Security: `--api-key` protects only `/v1/*`"

    With `cabrita serve`, `--api-key` gates **only** `/v1/*`.
    `/api/status`, `/api/worlds/deploy` and `/api/worlds/retrain` answer
    **without a key**, and **every** response sends
    `Access-Control-Allow-Origin: *`. Any web page open in your browser can
    call them while the server runs. The default bind address is `127.0.0.1`;
    keep it local and do not publish the port.

## Sections

- [UKJ + Cabrita Specification](spec-index.md): identity, files, hierarchy and routes, operators, `p_p` quantization, runtime, decisions and pending.
- [UKJ v1 Format (implementation)](formato-ukj-v1.md): container bytes, checksums, `.subs`, CLI and real results.

## Status

!!! success "Spec v1 approved by the user (2026-11-23)"

    - [x] Approval of the official `p1_p` definition (D54) — §4.1 closed.
    - [x] Confirmation of the `.subs` reading (§1).
    - [x] Degradation threshold above `p8_p` (D55) — set by F1.2: same perplexity and QA up to `p12_p`, so `p9_p`–`p12_p` stay experimental by cost.

    The review is still pending before publishing.

!!! info "Phase 0 prototype (2026-11-23)"

    - [x] GGUF→UKJ conversion, `.subs`, `inspect` and `verify`.
    - [x] Native inference without llama.cpp: prefill, decode, streaming, system prompt and metrics (tokens/s, RAM, latency, TTFT).
    - [x] Router of 8 worlds (95.8% held-out) and loading only activated routes: `chat` routes by default (`Code 88% · 16 experts per layer`, RSS 819 MB); `--world` forces a world.
    - [x] Densities `p1_p`, `p3_p` and `p5_p` converted, loaded and executed.
    - [x] Tics (2/s normal, 20/s anomalous) and `stop-subparameter` with hot offload and audit visible in `cabrita inspect --log`.
    - [x] `continue parameter` (definition 46): irregular parameters split into independent parameters and reassembled on load.

    The review is still pending before publishing.

!!! info "Phase 1 (2026-11-25)"

    - [x] ISS-14 — integrity: corruption and truncation detection, atomic checkpoints with rollback, `world backup|restore|retrain` with a SHA-256 manifest.
    - [x] ISS-15 — public benchmarks: `eval perplexity|qa`, the `p1_p`…`p12_p` degradation experiment (identical quality; D55 sets `p8_p` as the ceiling) and the full conversions of `p1_p`…`p8_p`, which share the payload byte for byte. Resolution of the published figures: perplexity over **160 tokens** (corpus of **798 bytes**) and QA over **10 questions** (80.0% = **8/10**; one answer = **10 pp**).
    - [x] ISS-16 — OpenAI-compatible server with embedded GUI and world editor, HTTPS and API key; Linux AppImage and Windows `.exe` builds. `dist/cabrita.exe` was **never executed**: it is validated only as a PE32+ binary, because the machine has no Windows and no Wine.

!!! warning "Router percentage ≠ model quality"

    The `89%` published in the videos is the **router's confidence for the prompt `rust error`** (world `codigo`), not a quality metric. It is reproducible with `cabrita route --text "rust error"` → `codigo 89.0%`. The model-level figure is the held-out accuracy of 95.8%.

## Next steps

1. Review of the v1 spec.
2. Upload of the single `cabrita` repo (after review).
3. Publish the MkDocs site.
