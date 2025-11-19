"""Generate RMSE panels for Algorithm 1 under varying hyperparameters."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from boosting_grf import GeneralizedBoostedKernels
from boosting_grf.datasets import generate_causal_data

plt.style.use('matplotlibrc')

def build_observations(y: np.ndarray, w: np.ndarray) -> list[Dict[str, float]]:
    """Convert outcome/treatment arrays into the expected observation format."""
    return [{"Y": float(y_i), "A": float(w_i)} for y_i, w_i in zip(y, w)]


def split_indices(n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Split indices into two equal halves for cross-fitting."""
    idx = rng.permutation(n)
    mid = n // 2
    return idx[:mid], idx[mid:]


def evaluate_rmse(
    *,
    n: int,
    p: int,
    dgp: str,
    B: int,
    max_depth: int,
    q: float,
    xi1: float,
    ridge_eps: float,
    min_leaf: int,
    test_size: int,
    seed: int,
) -> float:
    """Evaluate the RMSE of Algorithm 1 for a single configuration."""
    rng = np.random.default_rng(seed)
    data = generate_causal_data(n=n, p=p, dgp=dgp, sigma_tau=1.0, seed=rng.integers(1_000_000))
    test = generate_causal_data(n=test_size, p=p, dgp=dgp, sigma_tau=1.0, seed=rng.integers(1_000_000))

    idx1, idx2 = split_indices(n, rng)
    model = GeneralizedBoostedKernels.fit(
        data.X[idx1],
        build_observations(data.Y[idx1], data.W[idx1]),
        data.X[idx2],
        build_observations(data.Y[idx2], data.W[idx2]),
        B=B,
        q=q,
        xi1=xi1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        ridge_eps=ridge_eps,
        seed=int(rng.integers(1_000_000)),
    )
    tau_hat = model.predict_theta(test.X)["theta"]
    return float(np.sqrt(np.mean((tau_hat - test.tau) ** 2)))


def sweep_parameter(
    values: Iterable[float],
    base_cfg: Dict[str, float],
    trials: int,
    test_size: int,
    seed: int,
    param_name: str,
    workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate RMSEs across a parameter sweep using multiprocessing."""
    value_list = list(values)
    rng = np.random.default_rng(seed)
    means = []
    stds = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for value in tqdm(value_list, desc=f"{param_name} sweep", leave=False):
            seeds = [int(rng.integers(1_000_000)) for _ in range(trials)]
            futures = []
            for seed_val in seeds:
                cfg = base_cfg.copy()
                cfg[param_name] = value
                kwargs = {
                    "n": int(cfg["n"]),
                    "p": int(cfg["p"]),
                    "dgp": str(cfg["dgp"]),
                    "B": int(cfg["B"]),
                    "max_depth": int(cfg["max_depth"]),
                    "q": float(cfg["q"]),
                    "xi1": float(cfg["xi1"]),
                    "ridge_eps": float(cfg["ridge_eps"]),
                    "min_leaf": int(cfg["min_leaf"]),
                    "test_size": test_size,
                    "seed": seed_val,
                }
                futures.append(executor.submit(evaluate_rmse, **kwargs))
            rmses = []
            for fut in tqdm(as_completed(futures), total=len(futures), desc="trials", leave=False):
                rmses.append(fut.result())
            means.append(float(np.mean(rmses)))
            stds.append(float(np.std(rmses)))
    return np.asarray(means), np.asarray(stds)


def main() -> None:
    """CLI entrypoint for generating the RMSE diagnostic panel."""
    parser = argparse.ArgumentParser(description="RMSE diagnostics panel for Algorithm 1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots") / "rmse_panel.png",
        help="Path for the saved panel figure",
    )
    parser.add_argument("--trials", type=int, default=5, help="Number of Monte Carlo trials per configuration")
    parser.add_argument("--test-size", type=int, default=1000, help="Test set size for RMSE evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=None, help="Number of worker processes (default: cores)")
    args = parser.parse_args()

    base_cfg: Dict[str, float] = {
        "n": 2000,
        "p": 10,
        "dgp": "aw3",
        "B": 128,
        "max_depth": 6,
        "q": 1.0,
        "xi1": 1.0,
        "ridge_eps": 1e-8,
        "min_leaf": 8,
    }

    panels = [
        ("Samples", "n", [500, 1000, 2000, 5000, 10000]),
        ("Boosting rounds", "B", [16, 32, 64, 128, 256, 512]),
        ("Tree depth", "max_depth", [2, 4, 6, 8, 10, 12, 16, 24, 32, 40]),
        ("Dropout prob. q", "q", [0.2, 0.4, 0.6, 0.8, 1.0]),
        ("Subsample rate ξ₁", "xi1", [0.2, 0.4, 0.6, 0.8, 1.0]),
        ("Ridge ε", "ridge_eps", [1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes_iter = axes.ravel()

    for idx, (label, pname, vals) in enumerate(tqdm(panels, desc="Hyperparameter sweeps")):
        means, stds = sweep_parameter(
            vals,
            base_cfg,
            trials=args.trials,
            test_size=args.test_size,
            seed=args.seed,
            param_name=pname,
            workers=args.jobs,
        )
        vals_arr = np.asarray(vals, dtype=float)
        ax = axes_iter[idx]
        ax.plot(vals_arr, means, marker="o")
        ax.fill_between(vals_arr, means - 1.96 * stds, means + 1.96 * stds, alpha=0.3)
        if nlabel == 'Samples' or 'Ridge' in nlabel:
            ax.set_xscale("log")
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Algorithm 1 RMSE diagnostics (mean ± 1.96 σ across trials)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved panel to {args.output.resolve()}")


if __name__ == "__main__":
    main()
