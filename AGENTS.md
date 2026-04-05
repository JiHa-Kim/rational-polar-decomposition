# Agent Engineering Standards

Agents MUST adhere to the following workflow to ensure technical integrity and performance stability.

## 1. Mandatory A/B Verification
Every change—whether it's a refactor, optimization, or bug fix—**MUST** be verified using the A/B benchmarking suite before committing. Do not rely on local "one-off" tests for performance-critical code.

## 2. Two-Pass Validation Workflow
To balance iteration speed with final verification, use the two-pass approach provided by `bench_profile.py`:

- **Phase 1: Fast Eager Pass** (`--no-compile`)
  - **Goal**: Catch logic and accuracy regressions immediately.
  - **Action**: Run `uv run compare-hard --no-compile`. 
  - **Exit Condition**: If `Ortho Drift` or `P-Err Drift` exceeds 1%, abort and fix logic.

- **Phase 2: Full Compiled Pass** (`--compile`)
  - **Goal**: Verify real-world performance gains and ensure compatibility with `torch.compile` (Inductor).
  - **Action**: Run `uv run compare-hard --compile`.
  - **Requirement**: Performance must meet or exceed the target baseline without erratic "compilation outliers."

## 3. Robust Baseline Selection
Always compare against a meaningful point in history using git revisions:
```bash
uv run bench_profile.py --baseline <commit_hash> --hard --compile
```
- Compare against the "last known best" performance (e.g., commit `42a1aaf` for 185ms).
- Ensure benchmarking scripts are robust enough to handle interface changes (e.g., config objects vs global constants) when checking out older revisions.

## 4. Atomic Commits & Verification
- commit **only** after both validation phases pass.
- One logical change per commit (e.g., "refactor: workspace reuse", "perf: use cholesky_inverse").
- Include summary of A/B results in the commit message if performance was the primary goal.

## 5. Scripting & Tooling
- Prefer extending `bench_profile.py` for general benchmarking needs.
- Use `torch._dynamo.explain()` or `check_compile.py` to diagnose graph breaks before committing "compiled optimizations."
