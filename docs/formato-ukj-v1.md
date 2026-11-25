# UKJ v1 Format — implementation

> Technical implementation document of Phase 0 (ISS-6 and ISS-7).
> The functional specification lives in [`spec/ukj-spec-v1.md`](ukj-spec-v1.md).
> This document describes the bytes, not the intentions.

## 1. `.ukj` Container

Everything in the file is little endian.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 4 | magic `UKJ1` |
| 4 | 4 | schema version (`u32`, value 1) |
| 8 | 8 | `header_len` (`u64`) |
| 16 | 32 | SHA-256 of the JSON header |
| 48 | 8 | `payload_len` (`u64`) |
| 56 | 32 | SHA-256 of the payload |
| 88 | `header_len` | JSON header UTF-8 |
| 88 + `header_len` | `payload_len` | payload |
| final | 32 | SHA-256 of all previous (preamble + header + payload) |

The preamble is packed with `struct.Struct("<4sIQ32sQ32s")` (88 bytes).

### Payload

- Tensors are copied in the physical order of the GGUF.
- Each tensor starts at an offset that is a multiple of **64** relative to the payload.
- The gaps between tensors and the end of the payload are zeros.
- The bytes of each tensor are **exactly** those of the GGUF: no requantization, no reordering and no touching of the `q*` blocks.
- `payload_len` includes the final padding up to a multiple of 64.

### Header

The header is compact JSON, UTF-8, with keys ordered by the Python converter. Rust does not depend on that order: the checksum is computed over the bytes as saved.

```json
{
  "format": "UKJ",
  "version": 1,
  "created_at": "2026-11-22T20:00:00+00:00",
  "generator": "ukj-convert 0.1.0",
  "layout": {
    "header": "json-utf8",
    "tensor_offset_base": "payload",
    "sub_offset_base": "tensor",
    "endianness": "little",
    "alignment": 64
  },
  "model": {
    "id": "olmoe-q2_k-p3_p",
    "name": "Olmoe Instruct Hf",
    "architecture": "olmoe",
    "quantization": "q2_k",
    "tensor_count": 195,
    "parameter_count": 6919161856,
    "metadata_sha256": "…",
    "source": {"format": "GGUF", "version": 3, "file": "…", "size": 2562763232, "data_start": 1782240},
    "metadata": {"general.architecture": "olmoe", "tokenizer.ggml.tokens": ["…"]}
  },
  "payload": {"length": 2560980992, "alignment": 64, "sha256": "…"},
  "routes": [
    {"id": "global", "label": "Global", "kind": "global", "auto": true, "tensors": ["token_embd.weight", "output_norm.weight"]},
    {"id": "capa.0", "label": "Layer 0", "kind": "layer", "auto": true, "tensors": ["blk.0.attn_norm.weight"]}
  ],
  "operators": [
    {"name": "stop-subparameter", "enabled": true, "tics": {"normal": 2, "anomalous": 20}},
    {"name": "continue parameter", "enabled": true, "definition": 46}
  ],
  "densities": {"default": "p3_p", "allowed": ["p1_p", "…", "p8_p"]},
  "tensors": [
    {
      "name": "blk.0.attn_norm.weight",
      "dims": [2048],
      "ggml_type": 0,
      "type": "F32",
      "offset": 0,
      "nbytes": 8192,
      "rows": null,
      "route": "capa.0",
      "density": "p3_p",
      "sha256": "…",
      "subs": [
        {"index": 0, "row_start": 0, "row_count": 0, "byte_offset": 0, "nbytes": 2732},
        {"index": 1, "row_start": 0, "row_count": 0, "byte_offset": 2732, "nbytes": 2732},
        {"index": 2, "row_start": 0, "row_count": 0, "byte_offset": 5464, "nbytes": 2728}
      ]
    }
  ]
}
```

Tensor fields:

| Field | Meaning |
|---|---|
| `name` | original GGUF name |
| `dims` | GGUF dimensions (dims[0] is the fast dimension) |
| `ggml_type` / `type` | numeric GGML type and name (`Q2_K`, `F32`, …) |
| `offset` | offset relative to the payload, multiple of 64 |
| `nbytes` | exact size of the tensor in bytes |
| `rows` | complete contiguous rows (`dims[1] * …`), or `null` in 1D |
| `route` | id of the route containing the tensor |
| `density` | effective density of that tensor |
| `sha256` | SHA-256 of the tensor, informational |
| `subs` | contiguous subparameters that sum `nbytes` |
| `continued_from` | name of the source tensor if the portion comes from `continue parameter` |
| `independent` | `true` in continued portions: they do not share subparameters with the source |
| `source_offset` | byte offset of the portion in the original tensor |

`routes`, `operators` and `densities` are extensible tables. In Phase 0 the routes are automatic and provisional: `global` for what is not `blk.N.*`, and `capa.N` per block. They are marked with `auto: true` so that manual labeling replaces them.

## 2. `p_p` Quantization in the file

`p_p` is **structural**: it splits each tensor into `p` contiguous subparameters. It does not change bits, blocks or values.

1. If the tensor has complete rows (`rows >= p`) and each row is a multiple of a GGUF block, it is cut by rows: `p` subparameters of complete rows.
2. If not, it is cut by bytes, respecting complete GGUF blocks (`block_bytes` of the type: 84 for `Q2_K`, 110 for `Q3_K`, 4 for `F32`, …).
3. If the tensor is too small for valid `p` cuts, the effective number is reduced and the header declares the real density of that tensor (`p2_p`, `p1_p`, …). The model default remains the requested one.

Concatenating the `subs` in order reproduces the exact tensor. The subparameter offsets are relative to the start of the tensor.

## 3. Operators and tics

The header declares the operators in `operators`; the runtime applies them and leaves each change in an auditable JSONL log.

```json
[
  {"name": "stop-subparameter", "enabled": true, "scope": "hot",
   "tics": {"normal": 2, "anomalous": 20},
   "condition": {"type": "memory_threshold", "value_bytes": null}},
  {"name": "continue parameter", "enabled": true, "definition": 46,
   "condition": {"type": "route_match", "value": null}}
]
```

### `continue parameter` (definition 46)

It splits an irregular parameter into independent parameters, each with a single subparameter of its own:

1. If the tensor has complete rows, cut by rows; if not, by complete GGUF blocks. The rest of the bytes go to the last portion.
2. The portions are written contiguously in the payload as `<base>.c0`, `<base>.c1`, …; the original name is not duplicated.
3. Each portion declares `continued_from`, `independent: true`, `source_offset` (source byte) and a `subs` with a single entry.
4. `model.continued_tensors` counts the source tensors that were split.
5. On load, the runtime validates that the portions are contiguous and that the total is a multiple of `row_bytes`; then it reassembles them under the original name in a separate `logical` map. The portions remain addressable for offload.

Real example: `out/tiny-olmoe-p8-continue.ukj` splits 5 norms of 16 bytes into 20 parameters of 4 bytes; `cabrita chat` loads and generates with the reassembled tensor.

### Convention: tensor without subparameters

A tensor without subparameters is serialized with a single sub that is the full slice. It is an implementation convention derived from definition 46, not a spec change: when there are no subparameters to distribute, the single declared sub is the complete portion of the tensor.

### `stop-subparameter` and tics

A **tic** is an expert activation on a layer. The engine keeps a 1 second sliding window per layer and computes `tics/s`.

| Observed rate | State | Action |
|---|---|---|
| `>=` anomalous (default 20/s) | `deployed` | `MADV_WILLNEED` on the expert pages |
| `<=` normal (default 2/s) | `stopped` | `MADV_DONTNEED` (hot offload) |
| between both | `holding` | keeps the previous state (hysteresis) |

- `stop-subparameter` only restricts deployment and prefetch: the weights stay readable via mmap and **the `.subs` is never deleted**; a stop does not disable the model.
- `cabrita chat --tics --tics-normal 2 --tics-anomalous 20` activates the tic engine; `forward_token` calls `tick_tics` before each token, so an inactive layer for more than the window goes to `stopped` without waiting for the end.
- `cabrita tics --demo` simulates normal → anomalous → offload without loading the model.
- Thresholds are calibratable; 2/s and 20/s are the default, not a format constant.

### Audit log

Each transition writes a JSON line to `<model>.ops.jsonl` (by default, next to the `.ukj`; `--log-file` changes it):

```json
{"ts_ms":1763956940200,"operator":"stop-subparameter","target":"blk.10.ffn_*_exps",
 "tics_per_s":8.0,"state":"deployed","action":"deploy","rss_bytes":306708480}
```

`cabrita inspect <model> --log` first looks for `<model>.ops.jsonl` and, if it does not exist, the latest `~/.cabrita/logs/session-*.jsonl`. `cabrita tics --json` emits the same event to stdout.

### Vulkan backend

`--backend vulkan` replaces the CPU matvec with a WGSL compute shader over `wgpu` 0.19 (Vulkan backend):

- One workgroup of 256 threads per output row; each thread accumulates a part of the product and reduces in shared memory.
- Weights are uploaded once per tensor and cached in GPU; expert portions are addressed by byte offset.
- `--vulkan-cache-mb` limits the cache (default 4096 MB). If the tensor does not fit, it falls back to CPU for that tensor.
- `cabrita vulkan --probe` lists adapter, version and limits; `cabrita vulkan --selftest` compares F32/F16/Q2_K/Q3_K/Q6_K against CPU.
- CPU remains the default backend; Vulkan is explicit.

## 4. `.subs` Container

Same 88-byte preamble, magic `SUBS`, version 1. The JSON header declares:

```json
{
  "format": "SUBS",
  "version": 1,
  "model_id": "olmoe-q2_k-p3_p",
  "source": {
    "file": "olmoe-q2_k-p3_p.ukj",
    "size": 2562763232,
    "header_sha256": "…",
    "payload_sha256": "…",
    "payload_len": 2560980992
  },
  "entries": [
    {"tensor": "token_embd.weight", "index": 0, "offset": 0, "nbytes": 11268096, "sha256": "…"}
  ]
}
```

- `entries[].offset` is relative to the `.subs` payload and a multiple of 64.
- The payload contains real copies of the subparameters, with zero padding.
- `source.header_sha256` and `source.payload_sha256` identify the exact `.ukj` it was extracted from. If they do not match, the `.subs` is **out of sync**.
- Default path: `$CABRITA_HOME/models/<model_id>/subs.subs`, or `~/.cabrita/models/<model_id>/subs.subs` if there is no `CABRITA_HOME`.
- The `.subs` is regenerated from the `.ukj`; it does not need the original GGUF.

## 5. Integrity and rejection

Three layers are validated:

1. **Header**: magic, version, declared sizes, SHA-256 of the header.
2. **Payload**: SHA-256 of the payload in the preamble.
3. **File**: final SHA-256 over preamble + header + payload.

In addition, the header is validated structurally: unique tensors, offsets within the payload and non-overlapping, contiguous subparameters that sum `nbytes`, coherent density, routes that cover all tensors exactly once, operators with valid name and default density.

A truncated file, with wrong magic, with altered header or payload is rejected with a concrete message; it never loads halfway. `cabrita inspect` validates header and trailer by default; `--deep` adds the payload. `cabrita chat` verifies payload and trailer by default; `--no-verify-payload` skips it with an explicit warning.

Small updates are written through a checkpoint: the previous file is copied to `<file>/checkpoints`, the new content goes to a temporary file, and the checkpoint is restored if validation fails. This applies to `.subs` repair, world restore and router retraining.

## 6. CLI

### Python (`ukj-convert`)

```bash
# convert and materialize .subs
./bin/ukj-convert convert model/olmoe-q2_k.gguf -p p3_p -o out/olmoe-q2_k-p3_p.ukj

# inspect routes, tensors and checksums
./bin/ukj-convert inspect out/olmoe-q2_k-p3_p.ukj --tensors
./bin/ukj-convert inspect out/olmoe-q2_k-p3_p.ukj --json

# validate payload and .subs
./bin/ukj-convert verify out/olmoe-q2_k-p3_p.ukj --subs

# extract, verify or repair .subs
./bin/ukj-convert subs extract out/olmoe-q2_k-p3_p.ukj
./bin/ukj-convert subs verify out/olmoe-q2_k-p3_p.ukj
./bin/ukj-convert subs repair out/olmoe-q2_k-p3_p.ukj
```

### Rust (`cabrita`)

```bash
cd rust
cargo build --release
./target/release/cabrita inspect ../out/olmoe-q2_k-p3_p.ukj --tensors
./target/release/cabrita inspect ../out/olmoe-q2_k-p3_p.ukj --json --subs
./target/release/cabrita verify ../out/olmoe-q2_k-p3_p.ukj

# chat native on CPU, streaming and metrics
./target/release/cabrita chat ../out/olmoe-q2_k-p1_p.ukj \
  --prompt "Write a Python function that sums two numbers" --system "Answer in Spanish, briefly." --max-tokens 64 --ctx 512

# deterministic bank (no sampler)
./target/release/cabrita bench ../out/olmoe-q2_k-p1_p.ukj --tokens 32 --json

# world router + route plan + measured RAM
./target/release/cabrita route ../out/olmoe-q2_k-p1_p.ukj \
  --worlds ../worlds/olmoe.json --eval
./target/release/cabrita route ../out/olmoe-q2_k-p1_p.ukj \
  --worlds ../worlds/olmoe.json \
  --text "write a Python function" --load --unload

# chat: the router picks the world from the prompt (no --world)
./target/release/cabrita chat ../out/olmoe-q2_k-p1_p.ukj \
  --prompt "Write a Python function that sums two numbers" --max-tokens 32

# inference restricted to the experts measured in a world (forced)
./target/release/cabrita chat ../out/olmoe-q2_k-p1_p.ukj \
  --prompt "Write a Python function that sums two numbers" --world codigo --no-stream

# tics + stop-subparameter + audit
./target/release/cabrita tics --demo --json
./target/release/cabrita chat ../out/olmoe-q2_k-p1_p.ukj --tics \
  --tics-normal 2 --tics-anomalous 20 --log
./target/release/cabrita inspect ../out/olmoe-q2_k-p1_p.ukj --log

# world taxonomy: backup, restore with rollback and light retraining
./target/release/cabrita world backup --worlds ../worlds/olmoe.json
./target/release/cabrita world restore --worlds ../worlds/olmoe.json \
  --backup ../worlds/backups/olmoe-YYYYMMDDTHHMMSS.mmmZ.json
./target/release/cabrita world retrain --worlds ../worlds/olmoe.json --eval --json

# reproducible evaluation and OpenAI-compatible server
./target/release/cabrita eval perplexity ../out/olmoe-q2_k-p1_p.ukj \
  --text ../benchmarks/corpus.txt --ctx 512 --json
./target/release/cabrita eval qa ../out/olmoe-q2_k-p1_p.ukj \
  --data ../benchmarks/qa.jsonl --ctx 512 --json
./target/release/cabrita serve ../out/olmoe-q2_k-p1_p.ukj \
  --addr 127.0.0.1:8080 --ctx 512 --worlds ../worlds/olmoe.json

# HTTPS with a local certificate
./target/release/cabrita serve ../out/olmoe-q2_k-p1_p.ukj \
  --addr 127.0.0.1:8443 --tls-cert cert.pem --tls-key key.pem

# distribution
bash scripts/package_linux.sh          # dist/Cabrita-<version>-x86_64.AppImage
bash scripts/package_windows_cross.sh  # dist/cabrita.exe cross-built on Linux (mingw-w64)
powershell -File scripts/package_windows.ps1  # dist/cabrita.exe native build on Windows

# degradation experiment p1_p…p12_p (perplexity + QA)
python3 scripts/run_degradation.py --base ../out/olmoe-q2_k-p1_p.ukj --ctx 512

# full conversions p1_p…p8_p and payload comparison
python3 scripts/check_full_conversions.py --gguf ../olmoe-q2_k.gguf \
  --reference ../out/olmoe-q2_k-p1_p.ukj

# Vulkan backend (CPU is still the default)
./target/release/cabrita vulkan --probe
./target/release/cabrita vulkan --selftest
./target/release/cabrita bench ../out/olmoe-q2_k-p1_p.ukj --tokens 16 \
  --backend vulkan --vulkan-cache-mb 4096
```

`--subs-dir` changes the model root (by default `~/.cabrita/models`). `cabrita verify --quick` omits the payload SHA-256. `cabrita chat` uses mmap and dequantizes per block: there is no full model copy in RAM. `chat` trains the router with `--worlds` (by default it looks for `worlds/olmoe.json` or `../worlds/olmoe.json`) and masks the experts of the chosen world; if the text does not exceed the threshold, it warns and uses the base model. `--world` forces a world; `--mask-all` and `--no-renorm` are diagnostics. `--tics` activates the tic engine and `--log` writes `<model>.ops.jsonl`; `inspect --log` shows it. `--backend vulkan` is explicit and `--vulkan-cache-mb` limits the GPU weight cache. `cabrita world backup` copies the taxonomy with a SHA-256 manifest; `world restore` validates the backup and rolls back to `<worlds>/checkpoints` on failure; `world retrain` trains the linear router and writes `<worlds>.router.json`. `cabrita serve` exposes `/v1/models`, `/v1/chat/completions` and `/v1/completions` plus an embedded GUI at `/`; the GUI lists worlds, opens a world and deploys its expert ranges. HTTPS is enabled with `--tls-cert`/`--tls-key`, and `--api-key` protects `/v1/*`.

## 7. Real results (OLMoE q2_K)

Test model: `allenai/OLMoE-1B-7B-0924-Instruct-GGUF`, `olmoe-1b-7b-0924-instruct-q2_k.gguf`, 2.562.763.232 bytes.

| Check | Result |
|---|---|
| GGUF read | v3, 195 tensors, 38 metadata |
| Parameters | 6.919.161.856 |
| UKJ payload | 2.560.980.992 bytes (2.39 GiB) |
| SHA-256 payload | `25e12a6e…e0acee02` (identical in `p1_p` and `p3_p`) |
| Roundtrip | 195/195 tensors identical to the GGUF, byte to byte |
| `p1_p` | 195 subparameters |
| `p3_p` | 585 subparameters |
| `p1_p` conversion without `.subs` | ~15.5 s |
| `p3_p` conversion with `.subs` | ~35 s (includes extraction and verification) |
| `cabrita inspect --subs` | ~11 s on release |
| Repair of deleted `.subs` | ~35 s; action `created` |
| Repair of corrupted `.subs` | ~40 s; action `repaired` |

The payload matches the GGUF in this model because its offsets were already multiples of 64. If a GGUF brings offsets at 32, the converter inserts padding up to 64; the tensors are still identical, but the full payload no longer is a contiguous copy of the GGUF.

### Native inference (without llama.cpp)

Own engine in `rust/src/model.rs`, `quant.rs`, `tokenizer.rs` and `chat.rs`: mmap over the payload, Q2_K/Q3_K/Q6_K dequantization per block, parallel matvec, RoPE, KV cache, MoE top-8 and sampler with temperature/top-k/top-p.

| Check | Result |
|---|---|
| Dequantization vs numpy reference | Q2_K/Q3_K/Q6_K identical (diff 0.0) |
| Logits vs numpy OLMoE reference | `max_abs_diff=2.7e-5`, `mean_abs_diff=4e-6` |
| Tokenizer vs Hugging Face | 80/80 strings |
| `p1_p` chat (25 token prompt, ctx 128) | prefill ~11.5 s · decode 2.3 tok/s · RSS 1.39 GB |
| `p3_p` chat | prefill ~8.9 s · decode 2.5 tok/s · RSS ~1.4 GB |
| `p5_p` chat | prefill ~8.5 s · decode 3.1 tok/s · RSS ~1.4 GB |
| KV cache | 32 MB at ctx 128 (128 MB at ctx 512) |

RSS measured with `/usr/bin/time -v`; the old 1.88 GB figure does not reproduce and was corrected after independent verification.

`p1_p`, `p3_p` and `p5_p` share the payload byte for byte; `pN_p` only changes the subparameter splitting, not the weights.

### Densities `p1_p`…`p12_p`, degradation threshold (F1.2)

Experiment of 2026-11-25 with `scripts/run_degradation.py`: every density is produced from the same `olmoe-q2_k-p1_p.ukj` (header-only rewrite, `converter/tools/resplit_density.py`), then evaluated with `cabrita eval perplexity` on `benchmarks/corpus.txt` (160 tokens) and `cabrita eval qa` on `benchmarks/qa.jsonl` (10 items), ctx 512, `p1_p` as baseline.

!!! info "Resolution of these numbers"

    Perplexity is computed over **160 tokens**; the corpus file is **798
    bytes**. QA is computed over **10 questions**, so 80.0% means **8/10** and
    **one answer moves QA by 10 percentage points** — exactly the width of the
    ≥10% / −10 pp rule. Differences smaller than a step cannot be measured with
    this suite.

| Density | Subparameters | Perplexity | Δ vs `p1_p` | QA | Δ QA |
|---|---:|---:|---:|---:|---:|
| `p1_p` | 195 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p2_p` | 390 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p3_p` | 585 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p4_p` | 780 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p5_p` | 975 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p6_p` | 1170 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p7_p` | 1365 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p8_p` | 1560 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p9_p` | 1755 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p10_p` | 1950 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p11_p` | 2145 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| `p12_p` | 2340 | 20.870 | +0.0% | 80.0% | +0.0 pp |

The twelve variants carry the same payload and the same weights: `p_p` only changes how each tensor is cut into subparameters. The measured consequence is that the quality cliff §4 was looking for does not exist up to `p12_p`; what grows is the cost — 195 subparameters per step, 2,340 in `p12_p`, and a bigger subparameter index in the header — with no gain in perplexity or QA. Hence the decision recorded in D55: `p8_p` stays the official ceiling and `p9_p`–`p12_p` remain experimental by cost, not by quality. The rule set in advance (a variant counts as degraded at ≥10% perplexity or ≤−10 QA points) never triggers, so there is no threshold density to publish.

### Full conversions from the GGUF (`p1_p`…`p8_p`)

`scripts/check_full_conversions.py` converts the real GGUF once per density (not a header rewrite), verifies each container and compares its payload SHA-256 with `p1_p`:

| Density | File bytes | Payload bytes | Tensors | Payload SHA-256 | `cabrita verify` | Seconds |
|---|---:|---:|---:|---|---:|---:|
| `p1_p` | 2,562,220,570 | 2,560,980,992 | 195 | same | yes | 52 |
| `p2_p` | 2,562,236,261 | 2,560,980,992 | 195 | same | yes | 51 |
| `p3_p` | 2,562,251,808 | 2,560,980,992 | 195 | same | yes | 33 |
| `p4_p` | 2,562,267,542 | 2,560,980,992 | 195 | same | yes | 33 |
| `p5_p` | 2,562,283,024 | 2,560,980,992 | 195 | same | yes | 29 |
| `p6_p` | 2,562,298,476 | 2,560,980,992 | 195 | same | yes | 30 |
| `p7_p` | 2,562,313,731 | 2,560,980,992 | 195 | same | yes | 29 |
| `p8_p` | 2,562,329,323 | 2,560,980,992 | 195 | same | yes | 30 |

The eight containers are valid and share the payload `25e12a6e…e0acee02`; the file only grows with the subparameter index (+108,753 bytes from `p1_p` to `p8_p`). The source GGUF (2,562,763,232 bytes) lives outside the repository and is never committed.

!!! note "Regenerated artifacts (M5): why `p1_p` has two byte counts"

    This table reports `p1_p` = 2,562,220,570 bytes, while the `q*` table below
    reports 2,562,220,548. Both are correct: `conversions` **regenerated** the
    `.ukj` files from the GGUF with a header listing `p1_p`…`p12_p`, which is
    **22 bytes longer** than the Phase 0 header (eight allowed densities). The
    payload — and its SHA-256 — is identical in both. The original output
    folders were deleted, so neither artifact is reproducible in place.
    Quality: perplexity over **160 tokens**, QA over **10 questions** (80.0% =
    8/10; one answer = 10 pp).

`--continue-irregular` on the full model: the flag converts without error and reports `continued_tensors: 0` for OLMoE q2_K, because all 195 tensors always split into at least 12 valid subparameters. A `p12_p` file built this way loads and answers (`chat --prompt "Say hi" --max-tokens 2` → `"Hello"`, RSS 1.49 GB). The irregular path itself (tensors that cannot be split) is covered by the `tiny-olmoe` fixture: 5 norms of 16 B become 20 independent parameters of 4 B and the model chats with them.

### Quality/performance table `q*` × `p_p`

`converter/tools/bench_public.py` runs three fixed prompts (16 tokens, ctx 512) through `cabrita bench` and adds the same perplexity and QA suite; results in `benchmarks/results/q_star.{json,md}`.

| Model | `q*` | `p_p` | tok/s | TTFT ms | Latency ms | RSS MB | Perplexity | QA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `olmoe-q2_k-p1_p.ukj` | q2_k | `p1_p` | 2.72 | 8301 | 14207 | 2124.3 | 20.870 | 80.0% |
| `olmoe-q2_k-p3_p.ukj` | q2_k | `p3_p` | 2.67 | 9900 | 15918 | 2124.9 | 20.870 | 80.0% |
| `olmoe-q2_k-p5_p.ukj` | q2_k | `p5_p` | 2.93 | 7704 | 13185 | 2123.8 | 20.870 | 80.0% |
| `olmoe-q2_k-p8_p.ukj` | q2_k | `p8_p` | 2.61 | 8238 | 14362 | 2124.4 | 20.870 | 80.0% |

!!! note "Resolution and regenerated artifacts"

    Perplexity: **160 tokens**; QA: **10 questions**, so 80.0% = **8/10** and
    one answer = **10 pp**. The `p1_p` measured here is the Phase 0 file
    (2,562,220,548 bytes), 22 bytes smaller than the regenerated `p1_p` in the
    conversions table above, whose header lists `p1_p`…`p12_p`; the payload
    SHA-256 is the same.

The spread in tokens/s (2.61–2.93) and TTFT (7.7–9.9 s) is run-to-run noise on the same CPU: all four models do the same arithmetic over the same payload. RSS stays at 2.12 GB because the whole 2.39 GB payload is mapped and touched during the three prompts. `q*` and `p_p` are independent: `q*` fixes the bits per weight, `p_p` fixes how many subparameters each tensor is cut into.

### Router and routes

Taxonomy of 8 worlds in `worlds/olmoe.json` (code, math, cooking, travel, health, law, science, languages). The tiny router is trained with labeled examples; the novelty gate uses centroids and a noise class.

| Check | Result |
|---|---|
| Held-out accuracy (24 reserved examples) | 23/24 = 95.8% |
| Query "write a Python function…" | world `codigo`, p=88.2% |
| Query "rust error" | world `codigo`, p=89.0% |
| Nonsensical query | incomplete mode, similarity 0.05 |
| World plan (16 experts/layer) | 774.3 MB of 2.39 GB = 31.7% |
| RAM after loading the plan | 28.2 MB → 840.2 MB |
| Plan download | 840.2 MB → 63.5 MB |
| Chat restricted to `codigo` | RSS 817 MB (vs 1.39 GB full) |
| Chat restricted to `codigo`+`matematicas` | 19 experts/layer · RSS 885.2 MB |
| `chat` without `--world` (automatic router) | `Code 88% · 16 experts/layer` · RSS 819 MB |
| `chat --mask-all` (diagnostic) | 64 experts/layer · RSS 1.83 GB |

The experts per layer were measured with `examples/experts.rs`, which runs the real OLMoE router on examples from each world. The worlds share many experts: OLMoE has no clean separation by topic, so per-layer masks are the honest label, not an arbitrary partition.

!!! info "What a router percentage measures (the published 89%)"

    The figures `p=88.2%`, `Code 88%` and **`89%`** are the linear router's
    **confidence for one concrete prompt**, not a model quality metric. They
    are reproducible and prompt-dependent:
    `cabrita route ../out/olmoe-q2_k-p1_p.ukj --worlds ../worlds/olmoe.json --text "rust error"`
    prints `mundo: codigo … p=89.0%` with the candidate list
    (`codigo 89% · cocina 3% · idiomas 2% · derecho 2%`). The 89% shown in the
    TikTok videos is exactly this value for the prompt `rust error`, and the
    same line appears in the real `cabrita chat` capture. The held-out router
    accuracy above (95.8%) is the separate, model-level quality figure.

### OpenAI-compatible API, GUI and distribution

| Check | Result |
|---|---|
| Endpoints | `/v1/models`, `/v1/chat/completions` (JSON and SSE), `/v1/completions` (JSON and SSE), `/health`, `/api/status`, `/api/worlds`, `/api/worlds/deploy`, `/api/worlds/retrain`, embedded GUI at `/` |
| Integration test | `cli_serve_openai_and_gui`: models, chat, streaming, completions, 401 without key, GUI HTML, world list, deploy and retrain |
| HTTPS | `--tls-cert`/`--tls-key` with rustls 0.23; `curl -k https://127.0.0.1:18443/api/status` returns JSON |
| `/api/status` on the fixture | `{"ctx":64,"density":"p8_p","experts":2,"heads":1,"hidden":4,"layers":1,"model":"tiny-olmoe-p8_p","payload_bytes":9344,"rss_mb":17.19,"top_k":1,"uptime_s":31}` |
| AppImage | `dist/Cabrita-0.1.0-x86_64.AppImage`, 5.1 MB; `--version` → `cabrita 0.1.0` |
| Windows `.exe` | `dist/cabrita.exe`, 21 MB, PE32+; imports only `kernel32`, `msvcrt`, `user32`, `d3dcompiler_47`… (no mingw runtime DLLs) |

The server is single-threaded over a plain `TcpListener`: each request is served
in order, which keeps the KV cache and the loaded world consistent.

!!! warning "`--api-key` protects only `/v1/*`; `/api/*` stays open and CORS is `*`"

    `--api-key` gates **only** `/v1/*`. `GET /api/status`, `POST
    /api/worlds/deploy` and `POST /api/worlds/retrain` answer **without any
    key**, and **every** response carries
    `Access-Control-Allow-Origin: *`. While `cabrita serve` runs, any web page
    open in the same browser can read the server state and deploy or retrain
    worlds. The default bind address is `127.0.0.1`, so the exposure is local
    unless the port is published; do not expose it to a network you do not
    trust.

!!! note "The Windows `.exe` was never executed"

    `dist/cabrita.exe` is validated only as a binary: PE32+ header,
    architecture and imported DLLs (`kernel32`, `msvcrt`, `user32`,
    `d3dcompiler_47`…). The build machine has neither Windows nor Wine, so no
    one has run it or its GUI/API on Windows. The AppImage, in contrast, was
    run: `--version` and `serve --help`.

### Tics, `continue parameter` and Vulkan

| Check | Result |
|---|---|
| Real tics (16 layers, thresholds 1/5) | 16 deploys of 145,764,352 bytes/layer at 8 tics/s |
| Offload after silence (thresholds 2/20) | 16 `offload` + 32 `deploy` in 3 turns |
| Audit log | `<model>.ops.jsonl`, readable with `inspect --log` |
| `continue parameter` (fixture) | 5 norms of 16 B → 20 parameters of 4 B; loads and generates |
| Vulkan selftest F32/F16/Q2_K/Q3_K/Q6_K | max relative difference 5.352e-6 |
| CPU vs Vulkan (16 tokens) | same text; CPU 2.53 tok/s · Vulkan 2.64 tok/s |
| Prefill CPU vs Vulkan | 9562 ms · 10924 ms |
| RSS CPU vs Vulkan | 2.12 GB · 2.47 GB |

Vulkan does not accelerate this OLMoE on Renoir: the bottleneck is the bandwidth and the initial weight upload. The phase value is that the GPU path exists, gives the same result and is ready for hardware with more memory.

## 8. Interoperability and tests

- `converter/tests/` (Python, `unittest`): 18 tests of densities, `p1_p`, `p3_p`, `p5_p`, roundtrip, header, `.subs`, rejection of invalid files, `--continue-irregular` and density re-splitting.
- `rust/tests/ukj.rs`: 16 tests of parser, `inspect`, extraction, verification, repair, desync, truncation, world backup/restore/retrain, checkpoints, evaluation, OpenAI server/GUI and automatic routing of `chat` (world chosen, missing world and incomplete mode).
- `rust/src/*` (26 tests): dequantization, matvec, tokenizer, RMSNorm, RoPE, sampler, router, route planes, tics/transitions, operator log, checkpoints, evaluation, embedded GUI and `continue parameter`.
- Independent reference: `ref_olmoe.py` (numpy, without llama.cpp) compares the Rust engine logits tensor to tensor.
- The `.subs` generated by Python is verified with Rust and vice versa; tested with the fixture and with real OLMoE.
- Small fixtures: `rust/tests/fixtures/tiny.gguf`, `tiny.ukj`, `tiny.subs` and `out/tiny-olmoe-p8-continue.ukj` for reassembly.

## 9. Limits of this phase

- The OpenAI-compatible HTTP API and the persistent memory `.mem` are left for later phases.
- Vulkan uses a single device, no batching or multi-GPU. The weight cache has a cap (`--vulkan-cache-mb`, 4096 MB default); if a tensor does not fit, that tensor falls back to CPU.
- No requantization: `p_p` only splits; the bits are those of the GGUF.
- The automatic routes of the `.ukj` remain per layer; the semantic taxonomy lives in `worlds/olmoe.json` and uses expert masks.
- Tics use a 1 s window and default thresholds 2/s and 20/s; per-hardware calibration is still manual (`--tics-normal`, `--tics-anomalous`).
- The full payload is only a contiguous copy of the GGUF if its offsets were already multiples of 64; what is guaranteed is tensor to tensor.
- Restricted inference to a world is coherent but not identical to the full model: by limiting the MoE the available experts per token change.
