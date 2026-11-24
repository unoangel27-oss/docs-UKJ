# Hierarchy and routes — §2

- **Two levels:** parameter → subparameters. Each parameter carries **chained ids** that invoke its subparameters (in `.subs`).
- Two nodes with the same name → **merge** into a single node.

## Routes / worlds

- Separate layer (not part of the hierarchy), **they live in the `.ukj`**.
- Taxonomy **manually labeled** when translating each MoE.
- A **tiny router** (slightly trained) picks the world for each request.

## Loading

- Only the **activated** routes by the router — one if enough, several if the request needs them (with priorities), plus **provisional worlds** ("just in case").
- **Never** the whole model.

## No world match

Message + retry option + option to load the incomplete mode (without subparameters).

## Delete / recover a world

- Delete a world: light retraining to adapt the model without it.
- Recover: restore the backup and retrain to the original state.

## World editor (phase 1.3)

Press a world → "open" → deploy all its weights.

## Route diagram

``` mermaid
flowchart LR
    Req["Request"] --> Router{"Tiny router"}
    Router -->|"activated route"| Load["Load per route"]
    Load --> Subs["Subparameters"]
```
