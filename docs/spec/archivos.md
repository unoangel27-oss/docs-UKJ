# Files — §1

- **`.ukj`** — the single distributable file: **one file = one model**. Container (zip-like, custom extension) with: weights, parameters, **subparameters (dormant)**, routes/worlds, operators, density declarations and metadata.
- **`.subs`** — working copy of the subparameters, **one per model**, in Cabrita's system folder (e.g. `~/.cabrita/`). Materialized on install/convert of the `.ukj`. If missing or out of sync: **Cabrita repairs** it by regenerating it from the `.ukj`.
- **`.mem`** — conversation memory (toji state), in Cabrita's system folder. **Phase 2 only** (toji); phase 1 does not use it.

- **Incomplete mode:** a `.ukj` without a materialized `.subs` runs with parameters alone: possible, but incomplete (empty of route knowledge). Cabrita warns. It is the "world not found" fallback (degraded option, not default).
- **Updates:** only UKJ sections are rewritten (metadata, routes, operators, densities); weights do not change. **Progressive checkpoints**; mid-failure → rollback to last checkpoint.

## Table

| File | Content | Phase |
|---|---|---|
| `.ukj` | Full model: parameters, subparameters, routes/worlds, operators, density | Yes |
| `.subs` | Working copy of subparameters | Phase 1 |
| `.mem` | Conversation memory (toji) | Phase 2 |
