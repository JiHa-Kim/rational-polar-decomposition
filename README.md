# Minimal DWH2 vs PE5 benchmark

This repository is a deliberately small head-to-head comparison between two fast approximate polar-factor methods on tall matrices relevant to Muon-like optimizers:

- `dwh2`: 2-step dynamically weighted Halley.
- `pe5`: 5-step degree-5 Polar Express.

The repo is intentionally focused on matrix-only inputs and realistic low-precision GPU behavior.

## Current headline

Fresh current-`HEAD` benchmark on this machine with shared $\ell_0 = 10^{-3}$,
the default `spectral_bound` normalizer, and the default DWH2
`smallside_bounded` mode:

| Method | Median runtime | Median `q_fro_error` | Median `ortho_fro` |
| --- | ---: | ---: | ---: |
| `dwh2` | **381.06 ms** | **0.03537** | **0.07029** |
| `pe5` | 674.85 ms | 0.08877 | 0.18633 |

`dwh2` is 1.77x faster by median runtime and lower on `q_fro_error` in 10/11
default cases.

## Quick start

Default run:

```bash
uv run bench --device cuda --tf32
```

Rectangular DWH2 reference:

```bash
uv run bench --device cuda --tf32 --dwh2-mode rectangular
```

Frobenius baseline:

```bash
uv run bench --device cuda --tf32 --normalizer fro
```

Write JSONL output:

```bash
uv run bench --device cuda --tf32 --output runs/run1/results.jsonl --quiet
```

Run the normalization sweep:

```bash
uv run norm-sweep --device cuda --tf32 --quiet --output runs/norm_sweep/results.jsonl
```

Run the low-memory projected-objective audit:

```bash
uv run bench --device cuda --tf32 --reference fp32 --audit --audit-device same --audit-chunk-rows 512
```

## Docs

The README stays intentionally lean. Detailed notes live under [docs/](docs/README.md):

- [docs/implementation.md](docs/implementation.md): algorithm, normalization, and SPD inverse details.
- [docs/benchmarking.md](docs/benchmarking.md): cases, metrics, methodology, and detailed benchmark tables.
- [docs/dwh2-smallside-diagnosis.md](docs/dwh2-smallside-diagnosis.md): ongoing bounded small-side DWH2 diagnosis log.

## Source layout

- `src/polar_decomposition/algorithms`: DWH2 and PE5 implementations.
- `src/polar_decomposition/utils`: normalization and SPD inverse helpers.
- `src/polar_decomposition/kernels`: Triton/CUDA-specific fused kernels.
- `src/polar_decomposition/cli`: benchmark, sweep, and profiling entrypoints.
