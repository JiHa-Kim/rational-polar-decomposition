# Minimal DWH2 vs GNS benchmark

This repository is a deliberately small head-to-head comparison between two fast approximate polar-factor methods on tall matrices relevant to Muon-like optimizers:

- `dwh2`: 2-step dynamically weighted Halley.
- `gns`: Gram-Newton-Schulz method.

The repo is intentionally focused on matrix-only inputs and realistic low-precision GPU behavior.

## Current headline

Fresh current-`HEAD` benchmark on this machine with shared $\ell_0 = 10^{-3}$,
the default `spectral_additive` normalization, and the bounded small-side DWH2
kernel:

| Method | Median runtime | Median `ortho_proj` | Median `p2_gram_rel_fro` |
| --- | ---: | ---: | ---: |
| `dwh2` | **196.04 ms** | **9.52e-02** | **6.13e-02** |
| `gns` | 284.32 ms | 2.00e-01 | 1.68e-01 |

`dwh2` is ~1.45x faster by median runtime and more accurate in terms of orthogonality and gram reconstruction in this case.


## Quick start

Default run (TF32 enabled by default):

```bash
uv run python scripts/bench_profile.py --device cuda
```

To disable TF32:

```bash
uv run python scripts/bench_profile.py --device cuda --no-tf32
```

Write JSONL output:

```bash
uv run python scripts/bench_profile.py --device cuda --output results/results.jsonl --quiet
```

Run full instability profiles:

```bash
uv run python scripts/run_all_profiles.py
```

## Source layout

- `dwh2.py`: Main DWH2 implementation with stable rank and SPD inverse logic.
- `scripts/bench_profile.py`: Performance benchmarking and metric collection.
- `scripts/profile_instability.py`: Detailed stability analysis for specific matrix cases.
- `scripts/run_all_profiles.py`: Orchestrator for full stability suite.
