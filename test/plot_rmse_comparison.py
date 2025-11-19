"""RMSE diagnostics comparing Algorithm 1 and GRF."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from boosting_grf import GeneralizedBoostedKernels
from boosting_grf import grf
from boosting_grf.datasets import generate_causal_data

plt.style.use("matplotlibrc")


def build_observations(y: np.ndarray, w: np.ndarray):
    return [{"Y": float(y_i), "A": float(w_i)} for y_i, w_i in zip(y, w)]


def crossfit_linear_residuals(
    X: np.ndarray,
    y: np.ndarray,
    folds: int = 2,
    ridge: float = 1e-6,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    fold_ids = rng.integers(folds, size=n)
    preds = np.zeros(n)
    X_aug = np.column_stack([np.ones(n), X])
    for k in range(folds):
        mask = fold_ids == k
        train = ~mask
        if not np.any(mask):
            continue
        X_train = X_aug[train]
        y_train = y[train]
        XtX = X_train.T @ X_train + ridge * np.eye(X_train.shape[1])
        beta = np.linalg.solve(XtX, X_train.T @ y_train)
        preds[mask] = X_aug[mask] @ beta
    return y - preds


def rmse_boosting(cfg: Dict[str, float], seed: int) -> float:
    rng = np.random.default_rng(seed)
    data = generate_causal_data(
        n=int(cfg["n"]),
        p=int(cfg["p"]),
        dgp=str(cfg["dgp"]),
        sigma_tau=1.0,
        seed=int(rng.integers(1_000_000)),
    )
    test = generate_causal_data(
        n=int(cfg["test_size"]),
        p=int(cfg["p"]),
        dgp=str(cfg["dgp"]),
        sigma_tau=1.0,
        seed=int(rng.integers(1_000_000)),
    )
    idx = rng.permutation(int(cfg["n"]))
    mid = int(cfg["n"]) // 2
    d1, d2 = idx[:mid], idx[mid:]
    model = GeneralizedBoostedKernels.fit(
        data.X[d1],
        build_observations(data.Y[d1], data.W[d1]),
        data.X[d2],
        build_observations(data.Y[d2], data.W[d2]),
        B=int(cfg["B"]),
        q=float(cfg["q"]),
        xi1=float(cfg["xi1"]),
        max_depth=int(cfg["max_depth"]),
        min_leaf=int(cfg["min_leaf"]),
        ridge_eps=float(cfg["ridge_eps"]),
        seed=int(rng.integers(1_000_000)),
    )
    tau_hat = model.predict_theta(test.X)["theta"]
    return float(np.sqrt(np.mean((tau_hat - test.tau) ** 2)))


def rmse_grf(cfg: Dict[str, float], seed: int) -> float:
    rng = np.random.default_rng(seed)
    data = generate_causal_data(
        n=int(cfg["n"]),
        p=int(cfg["p"]),
        dgp=str(cfg["dgp"]),
        sigma_tau=1.0,
        seed=int(rng.integers(1_000_000)),
    )
    test = generate_causal_data(
        n=int(cfg["test_size"]),
        p=int(cfg["p"]),
        dgp=str(cfg["dgp"]),
        sigma_tau=1.0,
        seed=int(rng.integers(1_000_000)),
    )

    y_resid = crossfit_linear_residuals(data.X, data.Y, seed=int(rng.integers(1_000_000)))
    w_resid = crossfit_linear_residuals(data.X, data.W.astype(float), seed=int(rng.integers(1_000_000)))

    forest = grf.train_causal_forest(
        data.X,
        y_resid,
        w_resid,
        num_trees=int(cfg["grf_trees"]),
        sample_fraction=float(cfg["grf_sample_fraction"]),
        min_node_size=int(cfg["grf_min_leaf"]),
        honesty=True,
        honesty_fraction=0.5,
        seed=int(rng.integers(1_000_000)),
    )
    tau_hat = grf.predict_causal_forest(forest, test.X)["predictions"].reshape(-1)
    return float(np.sqrt(np.mean((tau_hat - test.tau) ** 2)))


def _evaluate_pair(cfg: Dict[str, float], seed: int) -> Tuple[float, float]:
    """Helper for parallel execution returning (Algorithm1, GRF) RMSEs."""
    return rmse_boosting(cfg, seed), rmse_grf(cfg, seed)


def sweep_parameter(
    values: Iterable[float],
    base_cfg: Dict[str, float],
    trials: int,
    seed: int,
    param_name: str,
    workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    value_list = list(values)
    rng = np.random.default_rng(seed)
    boost_means, boost_stds = [], []
    grf_means, grf_stds = [], []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for value in tqdm(value_list, desc=f"{param_name} sweep", leave=False):
            cfg = base_cfg.copy()
            cfg[param_name] = value
            seeds = [int(rng.integers(1_000_000)) for _ in range(trials)]
            futures = [
                executor.submit(_evaluate_pair, cfg.copy(), trial_seed) for trial_seed in seeds
            ]
            boost_scores = []
            grf_scores = []
            for fut in tqdm(as_completed(futures), total=len(futures), desc="trials", leave=False):
                boost_val, grf_val = fut.result()
                boost_scores.append(boost_val)
                grf_scores.append(grf_val)
            boost_means.append(float(np.mean(boost_scores)))
            boost_stds.append(float(np.std(boost_scores)))
            grf_means.append(float(np.mean(grf_scores)))
            grf_stds.append(float(np.std(grf_scores)))
    return (
        np.asarray(boost_means),
        np.asarray(boost_stds),
        np.asarray(grf_means),
        np.asarray(grf_stds),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RMSE diagnostics comparing Algorithm 1 and GRF.")
    parser.add_argument("--output", type=Path, default=Path("plots") / "rmse_comparison.png")
    parser.add_argument("--trials", type=int, default=10, help="Monte Carlo trials per grid point")
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--jobs", type=int, default=None, help="Worker processes for sweeps")
    args = parser.parse_args()

    base_cfg: Dict[str, float] = {
        "n": 2000,
        "p": 10,
        "dgp": "aw3",
        "B": 256,
        "max_depth": 32,
        "q": 1.0,
        "xi1": 1.0,
        "ridge_eps": 1e-8,
        "min_leaf": 8,
        "test_size": args.test_size,
        "grf_trees": 256,
        "grf_sample_fraction": 0.5,
        "grf_min_leaf": 5,
    }

    panels = [
        ("Samples", "n", [500, 1000, 2000, 5000, 10000]),
        ("Boosting rounds", "B", [16, 32, 64, 128, 256, 512]),
        ("Tree depth", "max_depth", [2, 4, 8, 16, 32, 64]),
        ("Dropout q", "q", [0.2, 0.4, 0.6, 0.8, 1.0]),
        ("Subsample ξ₁", "xi1", [0.2, 0.4, 0.6, 0.8, 1.0]),
        ("GRF sample frac.", "grf_sample_fraction", [0.2, 0.4, 0.6, 0.8, 1.0]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    axes_iter = axes.ravel()

    for ax, (label, pname, vals) in zip(axes_iter, panels):
        boost_mean, boost_std, grf_mean, grf_std = sweep_parameter(
            vals,
            base_cfg,
            trials=args.trials,
            seed=args.seed,
            param_name=pname,
            workers=args.jobs,
        )
        vals_arr = np.asarray(vals, dtype=float)
        ax.plot(vals_arr, boost_mean, marker="o", label="Algorithm 1")
        ax.fill_between(vals_arr, boost_mean - 1.96 * boost_std, boost_mean + 1.96 * boost_std, alpha=0.3)
        ax.plot(vals_arr, grf_mean, marker="s", linestyle="--", label="GRF")
        ax.fill_between(vals_arr, grf_mean - 1.96 * grf_std, grf_mean + 1.96 * grf_std, alpha=0.2)
        if pname in {"n", "ridge_eps"}:
            ax.set_xscale("log")
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Algorithm 1 vs GRF RMSE diagnostics (mean ± 1.96 σ)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved comparison panel to {args.output.resolve()}")


if __name__ == "__main__":
    main()
