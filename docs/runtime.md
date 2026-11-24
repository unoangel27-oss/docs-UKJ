# Cabrita Runtime — phase 1

- **Context:** traditional context window. `toji` + `.mem` = phase 2.
- **Inference:** prefill + decode + streaming. Backends: **CPU and Vulkan**. Platforms: Windows (`.exe`) and Linux (AppImage); macOS later.
- **Loading:** streaming without loading all into RAM, per route; `.subs` dormant until activation. A 200 GB model does not consume 200 GB: only the activated piece.
- **Metrics:** tokens/s, RAM, latency; public benchmark in F1.2.
- **System prompt:** injectable in the `.ukj`.
- **GUI:** served by Cabrita itself (embedded HTTPS server) — F1.3.
- **Languages:** Rust (core and inference), C (kernels), Python (conversion and density), Julia (extension tooling — optional; the binary layout is specified **language-neutral**, any runtime can write it).
- **Integrity:** corruption and truncation are detected by Cabrita (F1.1, with checkpoints).
