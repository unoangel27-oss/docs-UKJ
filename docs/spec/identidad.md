# Identity — §0

- **UKJ** = Universal Kernelized Joint-weights. New model format, extension **`.ukj`**, independent from GGUF/safetensors, with bidirectional conversion.
- **Cabrita** = the Rust runtime that executes UKJ. Local, offline, no telemetry, OpenAI-compatible HTTP API, MIT license.
- **Phase 1: MoE only.** Phase restriction, not format restriction: the format admits other architectures in the future (Mamba, diffusion, etc.).

## Relation with GGUF

UKJ is an **independent** format: it coexists and converts with GGUF and safetensors. The break vs GGUF is not of weights, but of **structure**: subparameters and routes.
