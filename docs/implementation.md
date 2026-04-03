# Implementation Notes

## Repository layout

- `src/polar_decomposition/algorithms`: core DWH2 and PE5 implementations.
- `src/polar_decomposition/utils`: normalization and SPD inverse helpers.
- `src/polar_decomposition/kernels`: Triton/CUDA-specific fused kernels.
- `src/polar_decomposition/cli`: benchmark, sweep, and profiling entrypoints.

## DWH2

We use the DWH scalar schedule from Nakatsukasa, Bai, and Gygi,

$$
a_k = h(\ell_k),\qquad
b_k = \frac{(a_k - 1)^2}{4},\qquad
c_k = a_k + b_k - 1,
$$

with

$$
\ell_{k+1} = \ell_k \frac{a_k + b_k \ell_k^2}{1 + c_k \ell_k^2}.
$$

The matrix update is

$$
X_{k+1} = X_k M_k, \quad M_k = \alpha_k I + \beta_k (I + c_k G_k)^{-1},
$$

where $G_k = X_k^\top X_k$ and $\alpha_k = b_k/c_k$, $\beta_k = a_k - \alpha_k$.
This is algebraically the same as the paper's

$$
X_{k+1} = X_k (a_k I + b_k G_k)(I + c_k G_k)^{-1},
$$

just rewritten to avoid explicitly forming the $a_k I + b_k G_k$ factor.

### Bounded small-side mode

The DWH2 implementation is the bounded small-side mode.

It keeps the first step in the initial small-side state,

$$
\Delta_0 = c_0 G_0,\qquad
H_0 = (I + \Delta_0)^{-1},\qquad
M_0 = \alpha_0 I + \beta_0 H_0,
$$

and forms the second inverse input directly as

$$
A_1 = I + \frac{c_1}{c_0} M_0 \Delta_0 M_0.
$$

Using $\Delta_0 H_0 = I - H_0$, the implementation evaluates this as

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
\Bigr),
$$

then factors $A_1$ directly on the small side and finishes with one large
right-multiply.

This removes the old dense-RHS second solve `K^{-1} H_0` from the critical
path. The bounded `H_0 @ (I - H_0)` site is unit-diagonal scaled before the
TF32 GEMM so the tensor-core multiply sees a correlation-like matrix instead of
the raw SPD block.

The final update is now applied directly through the Cholesky factor of $A_1$:

$$
K = \alpha_1 M_0 + \beta_1 (M_0 A_1^{-1}).
$$

Instead of explicitly forming $H_1 = A_1^{-1}$ and then multiplying by it, the
implementation applies $A_1^{-1}$ to $M_0$ with two right-side triangular
solves. That removes the last small-side GEMM from the bounded path and proved
both faster and slightly more accurate overall on the benchmark suite.

The ongoing bounded-mode diagnosis is documented in
[dwh2-smallside-diagnosis.md](dwh2-smallside-diagnosis.md).

## PE5

We use the Polar Express offline degree-5 coefficient generator with:

- the paper's cushioning,
- recentering around 1,
- the `1.01` safety factor on all but the final polynomial.

The online method is the fast small-side formulation. For a polynomial

$$
p(x) = x(a + b x^2 + c x^4),
$$

we apply it through the small-side Gram $Y = X^\top X$, maintaining a small-side factor $Q$ so that the large matrix is only multiplied at the end of each restart block.

We use restart interval 3. The paper suggests adding $10^{-3} I$ to the first Gram for more conservative low-precision stability; this implementation leaves that off by default because it noticeably hurts easy-case accuracy and is not usually needed in float32/TF32.

## Normalization

The methods do not inspect singular values. Inputs are normalized by a one-sided
additive spectral bound.
For a tall view $X$ and the computed small-side Gram $\widehat G = X^\top X$, we use

$$
t_1(\widehat G) = \operatorname{tr}(\widehat G),\qquad
t_2(\widehat G) = \|\widehat G\|_F^2,
$$

and first bound the computed Gram itself with the PSD 2-moment formula

$$
\widehat\lambda_{\mathrm{ub}}
=
\frac{
t_1(\widehat G) + \sqrt{(n - 1)(n t_2(\widehat G) - t_1(\widehat G)^2)}
}{n}.
$$

We then add the finite-precision Gram envelope directly at the eigenvalue level:

$$
\lambda_{\max}(G)
\le
\widehat\lambda_{\mathrm{ub}} + \eta \|X\|_F^2,
$$

so the default scale is

$$
\alpha_{\mathrm{ub}}
=
\sqrt{\widehat\lambda_{\mathrm{ub}} + \eta \|X\|_F^2}.
$$

This is less conservative in practice than inflating $t_2$ first.

The normalized input is then

$$
X_0 = \frac{A}{\alpha_{\mathrm{ub}} + 10^{-3}}.
$$

On CUDA `float32` with TF32 tensor-core matmuls enabled, $\eta$ is the simple dot-product envelope $2^{-10} + \gamma_m^{(32)}$, where $m$ is the Gram inner-product length. So the default scale is conservative for a written-down finite-precision model, not an empirical ridge knob.

The shared CLI design lower bound is

$$
\ell_0 = 10^{-3}
$$

by default. This is a design parameter, not an oracle estimate.

## SPD preconditioning

Every DWH solve uses the same small-side SPD inverse stack:

1. unit-diagonal scaling via `rsqrt(diag)`,
2. explicit symmetrization to fix float32 rounding from sequential scaling,
3. conditional jitter in the scaled space when needed,
4. `cholesky`,
5. inverse formation via Cholesky solves or triangular solves, depending on the call site.

The scale-and-symmetrize step also has a Triton fast path for large contiguous CUDA `float32` matrices.
