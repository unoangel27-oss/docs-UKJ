# docs-UKJ

Documentation for the **UKJ v1** model format: container bytes and checksums,
subparameters, routes/worlds, operators, `p_p` quantization and the Cabrita
runtime.

This repository holds documentation only. The converter and the native Rust
runtime live in [Cabrita](https://github.com/unoangel27-oss/Cabrita).

## Layout

All source pages are flat files in `docs/` (no nested page folders):

- `index.md`: entry point and links to the rest of the site.
- `formato-ukj-v1.md`: binary layout, checksums, `.subs`, CLI and results.
- `spec-index.md`: index of the functional specification pages.
- `ukj-spec-v1.md`, `identidad.md`, `archivos.md`, `jerarquia.md`,
  `operadores.md`, `cuantizacion.md`, `runtime.md`, `decisiones.md`,
  `pendientes.md`: the UKJ v1 specification.
- `assets/`: rounded logo and favicon used by the site header.

## Build

```bash
pip install -r requirements.txt
mkdocs build --strict
```

Use `mkdocs serve` for a local preview. The generated site is written to
`site/`, which is ignored by git.
