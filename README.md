# boosting-grf

## Overview
- Implements Algorithm 1 from *causalgrf.pdf*: use boosted trees to learn only the splitting structure (Stage 1), then reuse those structures as adaptive kernels for generalized random forest inference (Stage 2).
- Stage 1 operates on a structure-learning sample `D1`, repeatedly subsampling observations (`ξ₁`) and previous trees (`q`) to grow depth-limited trees keyed to the ρ-score.
- Stage 2 projects a GRF sample `D2` through the cached partitions, yielding β-weights that support downstream CATE estimation and inference.

## Implementation Highlights
- **Leaf-level caching:** All leaf assignments and α-weight sufficient statistics are cached once per round, so later queries only look up precomputed aggregates.
- **Dropout signature aggregation:** Unique combinations of previous-tree leaves are collapsed; each signature is solved once, dramatically cutting the weighted least-squares workload.
- **Vectorized ρ updates:** The split residuals use closed-form 2×2 solves with ridge regularization, avoiding per-observation Python loops.
- **Stage 2 acceleration:** Leaf membership dictionaries reduce β-weight updates from `O(|D2|·B)` scans to `O(L)` lookups, where `L` is the size of the active leaf.
- **Reusable datasets:** `boosting_grf.datasets.generate_causal_data` matches the `aw1/aw2/aw3` benchmarks used in the GRF literature for fast experimentation.

## Usage Example
```python
import numpy as np
from boosting_grf import fit_alg1_grf

# Synthetic data (match dictionary format expected by the API)
X = np.random.normal(size=(500, 5))
A = np.random.binomial(1, 0.5, size=500).astype(float)
Y = (1 + X[:, 0] - 0.5 * X[:, 1]) * A + np.random.normal(size=500)
obs = [{"Y": float(y), "A": float(a)} for y, a in zip(Y, A)]

split = 300
X1, O1 = X[:split], obs[:split]
X2, O2 = X[split:], obs[split:]

model = fit_alg1_grf(X1, O1, X2, O2, B=32, q=0.6, xi1=0.6, max_depth=3, min_leaf=20, seed=3)
pred = model.predict_theta(X[:5])
print(pred["theta"])  # Estimated CATEs at the first five locations
```

## Tests and Benchmarks
- Unit smoke tests live under `test/test_basic.py`. Run them with `pytest test/test_basic.py`.
- The `test/plot_rmse_panel.py` script sweeps six hyperparameters (sample size, boosting rounds, tree depth, dropout rate `q`, subsample rate `ξ₁`, ridge ε) and emits a 2×3 RMSE panel using `matplotlib.fill_between` for one-standard-deviation bands. Results are saved under `plots/` by default. Use `--jobs` to parallelise Monte Carlo trials:
  ```bash
  python test/plot_rmse_panel.py --trials 8 --jobs 8
  ```
- `test/benchmark_table1.py` reproduces the simulations behind Table 1 of Athey et al. (2019). It reports the original WA-1/WA-2 baselines alongside the estimator implemented here (`GRFBoost` and `C.GRFBoost`). Pass `--jobs` to parallelise across scenarios and Monte Carlo trials:
  ```bash
  python test/benchmark_table1.py --reps 20 --trees 400 --jobs 8 --output table1_results.json
  ```
  The JSON artefact records both the reference values (WA-1/WA-2/GRF/C.GRF) and the estimates obtained with this repository.
- Visualise the resulting JSON with `test/visualize_table1.py` to see the WA baselines and GRFBoost comparisons side-by-side:
  ```bash
  python test/visualize_table1.py --input table1_results.json --output plots/table1_comparison.png
  ```

## Time Complexity (after optimizations)
- Let `m ≈ ξ₁·|D1|` be the subsample size, `s` the expected dropout load `|S_b|`, `p` the feature count, and `L_b ≤ 2^{max_depth}` the number of leaves at depth `b`.
- **Stage 1 (per boosting round):**
  - Building caches: `O(s · m · depth)` for leaf ID evaluation plus `O(U · s)` to aggregate across `U ≤ m` unique signatures.
  - Split search: `O(p · m log m)` across the frontier.
  - ρ update: `O(m)` thanks to vectorized matrix solves.
  - Overall: `O(B · (s·m + p·m log m))`, linear in the number of stored signatures and quasi-linear in `m`.
- **Stage 2 preparation:** `O(B · |D2|)` to tabulate leaf membership dictionaries.
- **Prediction (`β` weights):** `O((B + Σ_b|S_b|) · ℓ)` per query, where `ℓ` is the size of the active leaf (typically `min_leaf`). This replaces the earlier `O((B + Σ_b|S_b|) · |D2|)` scans.

## Remaining Bottlenecks & Possible Speedups
1. **Stage 2 predictions** still loop through every tree per query. For large batch prediction, vectorizing across queries or caching β-weights for repeated trees would amortize the per-tree cost.
2. **Signature aggregation** runs `np.unique` on an `(m × s)` matrix each round. When `s` is large, consider hashing tuples or incremental updates as trees are added instead of recomputing from scratch.
3. **Parallelism:** Tree rounds are independent once previous caches are built; `joblib` or `multiprocessing` can parallelize split evaluation, and predictions can be chunked across cores.
4. **Numerical kernels:** Heavy-use functions (`compute_beta_weights`, `aggregate_signature_stats`) are prime candidates for Numba/JAX compilation if more speed is required.

## Practical Tips
- Keep `ξ₁` and `min_leaf` aligned: raising `min_leaf` reduces the number of active leaves, which tightens both Stage 1 and Stage 2 runtime.
- Tune dropout `q`: higher dropout boosts statistical robustness but increases `s`; use profiling on your target workload to pick a sweet spot.
- When predicting a large test set, batch the matrix solves by stacking `x_new` rows and reusing the same β-weights whenever two points fall into identical tree leaves.

## Installation

Install the package in editable mode for development:

```bash
python -m pip install -e .
```

After installation the `boosting_grf` package is importable from anywhere, and CLI utilities in `test/` will pick it up without modifying `PYTHONPATH`.
