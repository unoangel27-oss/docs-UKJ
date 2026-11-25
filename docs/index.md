# Cabrita

**Cabrita** is the Rust runtime that executes the **UKJ** model format —
*Universal Kernelized Joint-weights*, extension **`.ukj`** — and its ecosystem of
subparameters, routes/worlds, operators and `p_p` quantization.

- **UKJ** is the format: a `.ukj` file = one model. It contains parameters,
  subparameters, routes/worlds, operators, densities and metadata.
- **Cabrita** is the runtime: local, offline, no telemetry, OpenAI-compatible HTTP
  API, MIT license.

## Sections

- [UKJ + Cabrita Specification](spec-index.md): identity, files, hierarchy and routes, operators, `p_p` quantization, runtime, decisions and pending.
- [UKJ v1 Format (implementation)](formato-ukj-v1.md): container bytes, checksums, `.subs`, CLI and real results.

## Status

!!! success "Spec v1 approved by the user (2026-11-23)"

    - [x] Approval of the official `p1_p` definition (D54) — §4.1 closed.
    - [x] Confirmation of the `.subs` reading (§1).
    - [ ] Degradation threshold above `p8_p` (D55) — set after the F1.2 experiment.

    The review is still pending before publishing.

!!! info "Phase 0 prototype (2026-11-23)"

    - [x] GGUF→UKJ conversion, `.subs`, `inspect` and `verify`.
    - [x] Native inference without llama.cpp: prefill, decode, streaming, system prompt and metrics (tokens/s, RAM, latency, TTFT).
    - [x] Router of 8 worlds (95,8% held-out) and loading only activated routes: `chat` routes by default (`Code 88% · 16 experts per layer`, RSS 819 MB); `--world` forces a world.
    - [x] Densities `p1_p`, `p3_p` and `p5_p` converted, loaded and executed.
    - [x] Tics (2/s normal, 20/s anomalous) and `stop-subparameter` with hot offload and audit visible in `cabrita inspect --log`.
    - [x] `continue parameter` (definition 46): irregular parameters split into independent parameters and reassembled on load.

    The review is still pending before publishing.

## Next steps

1. Review of the v1 spec.
2. Upload of the single `cabrita` repo (after review).
3. Publish the MkDocs site.
