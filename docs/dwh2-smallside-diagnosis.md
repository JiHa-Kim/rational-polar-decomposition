# DWH2 Small-Side Diagnosis

This note tracks the bounded small-side DWH2 kernel and the main conclusions
from the stability/speed investigation.

## Current Status

The current `smallside_bounded` kernel in
[dwh2.py](../src/polar_decomposition/algorithms/dwh2.py) is now the default
DWH2 mode.

On the `16384 x 4096`, `seed=0`, `spectral_additive` benchmark path:

- rectangular reference: [runs/dwh2_rectangular_additive_20260402.jsonl](../runs/dwh2_rectangular_additive_20260402.jsonl)
- current bounded run: [runs/dwh2_smallside_bounded_invdelta_20260402.jsonl](../runs/dwh2_smallside_bounded_invdelta_20260402.jsonl)

Integrated DWH2-only result:

- `rectangular` median runtime: `390.26 ms`
- `smallside_bounded` median runtime: `362.62 ms`
- `smallside_bounded` median `q_fro_error`: `0.02964` vs rectangular `0.03062`
- faster on `11/11`
- better `q_fro_error` on `9/11`
- remaining worse rows: `lowrank_noise`, `rank_1_heavy`

## Current Kernel

The bounded kernel uses:

- balanced split-K tree reduction for the initial Gram
- first inverse
  $H_0 = (I + c_0 G_0)^{-1}$
  from a Cholesky factor
- direct second-stage inverse input
  $A_1 = I + (c_1 / c_0) M_0 \Delta_0 M_0$
  with $\Delta_0 = c_0 G_0$
- direct second-stage inverse
  $H_1 = A_1^{-1}$
  on the small side
- one final large right-multiply

The important structural change is that the kernel no longer forms the old
dense-RHS second solve

$$
H_1 = K^{-1} H_0.
$$

Instead it builds the second inverse input from the initial bounded state:

$$
\Delta_0 = c_0 G_0,\qquad
H_0 = (I + \Delta_0)^{-1},\qquad
M_0 = \alpha_0 I + \beta_0 H_0,
$$

$$
A_1 = I + \frac{c_1}{c_0} M_0 \Delta_0 M_0.
$$

Using

$$
\Delta_0 H_0 = I - H_0,
$$

the implementation evaluates

$$
A_1
=
I + \frac{c_1}{c_0}
\Bigl(
\alpha_0^2 \Delta_0
+
2 \alpha_0 \beta_0 (I - H_0)
+
\beta_0^2 H_0 (I - H_0)
\Bigr).
$$

The bounded $H_0 @ (I - H_0)$ site is unit-diagonal scaled before the TF32 GEMM.

## What Actually Mattered

The investigation narrowed to a few real issues.

### 1. The old instability was structural.

The explicit propagated Gram-like state was the original failure mode. The
bounded recurrence removed the SPD-loss pathology without needing fallback
branches in the good path.

### 2. The small-side Gram was not the main problem.

On the hard rows, the TF32 split-K Gram was already very close to the FP32 one.
The dominant gap came from how the bounded state was propagated and applied.

### 3. The old dense-RHS second solve was a real bottleneck.

The current improvement came from removing that solve entirely, not from trying
to tune it forever. Once the second stage was rewritten as a direct inverse of a
small SPD block, both speed and quality improved materially.

### 4. Identity-heavy terms should stay out of TF32 matmuls.

Rewrites that keep the dominant identity contribution in scalar arithmetic or in
explicit diagonal updates consistently behaved better than raw TF32 matmul
chains on the same algebra.

## What Worked

- bounded small-side state instead of explicit propagated congruence state
- balanced split-K tree reduction
- direct second-stage inverse input built from $\Delta_0$ and $H_0$
- triangular inverse backend for the second small-side inverse
- unit-diagonal scaling of the bounded $H_0 @ (I - H_0)$ matmul
- explicit symmetrization of the small-side SPD states

## What Did Not Work

- more Cholesky jitter
- blaming the problem on the initial TF32 Gram
- focusing on the second solve backend after the direct-inverse rewrite
- full right-side second-solve reformulations
- factor-layout changes or column equilibration
- polynomial rewrites that lost SPD in TF32

## Remaining Gap

The remaining losses are narrow:

- `lowrank_noise`: slightly worse than rectangular on `q_fro_error`
- `rank_1_heavy`: still trails rectangular, though much less than before

Those rows are now small residual gaps inside an otherwise better kernel. The
main future target is no longer “make small-side DWH work at all”; it is
“tighten the last low-rank rows without giving back the current speed win.”
