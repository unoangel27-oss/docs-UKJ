# Operators — §3

The two operators are **independent** (they do not form a pair; the reversibility "stop and continue later" is a property of the mechanism, not of the relationship between operators).

- **`stop-subparameter`:** restricts subparameter deployment; subparameters deploy only when the **tics** level reaches the anomalous threshold. Allows **hot offload** and **total jump** of a whole substructure (parameter + all its subparameters).
- **`continue parameter`:** definition 46 — distribute irregular parameters (without perfect quantization) into independent parameters **without subparameters**.

## Tics

- Continuous activation that, when repeated until the threshold, forces releasing subparameters.
- Defaults in spec: **2/s normal, 20/s anomalous**; Cabrita can calibrate per hardware.

## Who decides

Dynamic: **Cabrita** decides in hot (not a static rule of the file).

## Conditional operators (phase 1)

Each operator accepts a condition: memory threshold or route match.

## Nomenclature and reserves

- Official names as-is: `stop-subparameter`, `continue parameter`; the guion/space difference is frozen in the spec.
- The header reserves an **extensible table**; `skip`, `resume`, `override`, `reset` remain reserved, not implemented in phase 1.

## Audit and validity

- Session log (timestamp, operator, target, tics level) visible with `cabrita inspect --log`.
- A stop **does not** leave the model unusable: subparameters are never deleted from `.subs`, only not deployed; deployment is always reversible.
- Validator: the file declares schema version and Cabrita validates on load; invalid file → rejection with message, never loads halfway.

## Break vs GGUF

- Structural = **subparameters + routes**.
- Control mechanism = **operators** (they make them usable in hot).
