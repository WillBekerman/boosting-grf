"""Compare Algorithm 1 against GRF on the Table 1 scenarios."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm.auto import tqdm

from boosting_grf import GeneralizedBoostedKernels
from boosting_grf import grf
from boosting_grf.datasets import generate_causal_data


SCENARIOS: List[Tuple[str, int, int]] = [
    ("aw2", 10, 800),
    ("aw2", 10, 1600),
    ("aw2", 20, 800),
    ("aw2", 20, 1600),
    ("aw1", 10, 800),
    ("aw1", 10, 1600),
    ("aw1", 20, 800),
    ("aw1", 20, 1600),
    ("aw3", 10, 800),
    ("aw3", 10, 1600),
    ("aw3", 20, 800),
    ("aw3", 20, 1600),
]


def build_observations(y: np.ndarray, w: np.ndarray) -> List[Dict[str, float]]:
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


@dataclass
class Scenario:
    dgp: str
    p: int
    n: int


@dataclass
class EvalConfig:
    trees: int
    max_depth: int
    q: float
    xi1: float
    ridge_eps: float
    min_leaf: int
    grf_trees: int
    grf_sample_fraction: float
    grf_min_leaf: int


def run_single_trial(
    scenario: Scenario,
    cfg: EvalConfig,
    test_size: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    data = generate_causal_data(
        n=scenario.n,
        p=scenario.p,
        dgp=scenario.dgp,
        sigma_tau=1.0,
        seed=int(rng.integers(1_000_000)),
    )
    test = generate_causal_data(
        n=test_size,
        p=scenario.p,
        dgp=scenario.dgp,
        sigma_tau=1.0,
        seed=int(rng.integers(1_000_000)),
    )

    idx = rng.permutation(scenario.n)
    mid = scenario.n // 2
    d1, d2 = idx[:mid], idx[mid:]

    model = GeneralizedBoostedKernels.fit(
        data.X[d1],
        build_observations(data.Y[d1], data.W[d1]),
        data.X[d2],
        build_observations(data.Y[d2], data.W[d2]),
        B=cfg.trees,
        q=cfg.q,
        xi1=cfg.xi1,
        max_depth=cfg.max_depth,
        min_leaf=cfg.min_leaf,
        ridge_eps=cfg.ridge_eps,
        seed=int(rng.integers(1_000_000)),
    )
    tau_boost = model.predict_theta(test.X)["theta"]
    mse_boost = float(np.mean((tau_boost - test.tau) ** 2))

    y_resid = crossfit_linear_residuals(data.X, data.Y, seed=int(rng.integers(1_000_000)))
    w_resid = crossfit_linear_residuals(data.X, data.W.astype(float), seed=int(rng.integers(1_000_000)))

    forest = grf.train_causal_forest(
        data.X,
        y_resid,
        w_resid,
        num_trees=cfg.grf_trees,
        sample_fraction=cfg.grf_sample_fraction,
        min_node_size=cfg.grf_min_leaf,
        honesty=True,
        honesty_fraction=0.5,
        seed=int(rng.integers(1_000_000)),
    )
    tau_grf = grf.predict_causal_forest(forest, test.X)["predictions"].reshape(-1)
    mse_grf = float(np.mean((tau_grf - test.tau) ** 2))
    return mse_boost, mse_grf


def evaluate_scenario(
    scenario: Scenario,
    cfg: EvalConfig,
    test_size: int,
    reps: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    boost = []
    grf_scores = []
    for _ in tqdm(range(reps), desc=f"{scenario.dgp}-{scenario.p}-{scenario.n}", leave=False):
        trial_seed = int(rng.integers(1_000_000))
        mse_boost, mse_grf = run_single_trial(scenario, cfg, test_size, trial_seed)
        boost.append(mse_boost)
        grf_scores.append(mse_grf)
    return {
        "boosting_mean": float(np.mean(boost)),
        "boosting_std": float(np.std(boost)),
        "grf_mean": float(np.mean(grf_scores)),
        "grf_std": float(np.std(grf_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Algorithm 1 and GRF on Table 1 scenarios.")
    parser.add_argument("--reps", type=int, default=5, help="Monte Carlo repetitions per scenario")
    parser.add_argument("--test-size", type=int, default=1000, help="Test set size for RMSE estimates")
    parser.add_argument("--seed", type=int, default=17, help="Master random seed")
    parser.add_argument("--trees", type=int, default=128, help="Boosting rounds for Algorithm 1")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--xi1", type=float, default=0.5)
    parser.add_argument("--ridge-eps", type=float, default=1e-8)
    parser.add_argument("--min-leaf", type=int, default=10)
    parser.add_argument("--grf-trees", type=int, default=1500)
    parser.add_argument("--grf-sample-fraction", type=float, default=0.5)
    parser.add_argument("--grf-min-leaf", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("table1_grf_comparison.json"))
    args = parser.parse_args()

    cfg = EvalConfig(
        trees=args.trees,
        max_depth=args.max_depth,
        q=args.q,
        xi1=args.xi1,
        ridge_eps=args.ridge_eps,
        min_leaf=args.min_leaf,
        grf_trees=args.grf_trees,
        grf_sample_fraction=args.grf_sample_fraction,
        grf_min_leaf=args.grf_min_leaf,
    )

    results = []
    for dgp, p, n in tqdm(SCENARIOS, desc="Scenarios"):
        scenario = Scenario(dgp=dgp, p=p, n=n)
        stats = evaluate_scenario(scenario, cfg, args.test_size, args.reps, args.seed)
        entry = {
            "scenario": asdict(scenario),
            "stats": stats,
        }
        results.append(entry)

    payload = {
        "config": asdict(cfg),
        "test_size": args.test_size,
        "reps": args.reps,
        "seed": args.seed,
        "scenarios": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote comparison metrics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
