# DWH2 Small-Side Diagnosis

This note tracks the bounded small-side DWH2 kernel and the main conclusions
from the stability/speed investigation.

## Current Status

The current `smallside_bounded` kernel in
[dwh2.py](../src/polar_decomposition/algorithms/dwh2.py) is now the default
DWH2 mode.

On the `16384 x 4096`, `seed=0`, `spectral_additive` benchmark path:

- current bounded run: [runs/dwh2_smallside_bounded_finalsolve_q_20260403.jsonl](../runs/dwh2_smallside_bounded_finalsolve_q_20260403.jsonl)

Integrated DWH2-only result:

- `smallside_bounded` median runtime: `345.64 ms`
- `smallside_bounded` median `q_fro_error`: `0.02963`
- bounded DWH2 now beats PE5 on `q_fro_error` in `11/11` default cases
- the remaining internal hard rows are still `lowrank_noise` and `rank_1_heavy`

## Current Kernel

The bounded kernel uses:

- balanced split-K tree reduction for the initial Gram
- first inverse
  $H_0 = (I + c_0 G_0)^{-1}$
  from a Cholesky factor
- direct second-stage inverse input
  $A_1 = I + (c_1 / c_0) M_0 \Delta_0 M_0$
  with $\Delta_0 = c_0 G_0$
- one final large right-multiply
- direct final apply
  $K = \alpha_1 M_0 + \beta_1 (M_0 A_1^{-1})$
  through two right-side triangular solves

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

The bounded $H_0 @ (I - H_0)$ site is unit-diagonal scaled before the TF32
GEMM. The newest change also removes the last explicit small-side inverse/GEMM
pair from the bounded path by applying $A_1^{-1}$ directly to $M_0$ through the
second Cholesky factor.

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
- unit-diagonal scaling of the bounded $H_0 @ (I - H_0)$ matmul
- direct final right-side solve for $M_0 A_1^{-1}$
- explicit symmetrization of the small-side SPD states

## What Did Not Work

- more Cholesky jitter
- blaming the problem on the initial TF32 Gram
- focusing on the second solve backend after the direct-inverse rewrite
- full right-side second-solve reformulations
- factor-layout changes or column equilibration
- polynomial rewrites that lost SPD in TF32

## Remaining Gap

The remaining hard rows are:

- `lowrank_noise`
- `rank_1_heavy`

Against PE5 the current bounded kernel now wins `q_fro_error` on all `11/11`
default cases while keeping the large runtime advantage. The future target is
no longer “make small-side DWH work at all”; it is “tighten those two hard rows
without giving back the current speed win.”
