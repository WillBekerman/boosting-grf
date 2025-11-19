"""Reproduce Table 1 from Athey et al. (2019) for Algorithm 1."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
from tqdm.auto import tqdm

from boosting_grf import GeneralizedBoostedKernels
from boosting_grf.datasets import generate_causal_data


REFERENCE_TABLE1 = {
    ("aw2", 10, 800): {"WA-1": 1.37, "WA-2": 6.48, "GRF": 0.85, "C.GRF": 0.87},
    ("aw2", 10, 1600): {"WA-1": 0.63, "WA-2": 6.23, "GRF": 0.58, "C.GRF": 0.59},
    ("aw2", 20, 800): {"WA-1": 2.05, "WA-2": 8.02, "GRF": 0.92, "C.GRF": 0.93},
    ("aw2", 20, 1600): {"WA-1": 0.71, "WA-2": 7.61, "GRF": 0.52, "C.GRF": 0.52},
    ("aw1", 10, 800): {"WA-1": 0.81, "WA-2": 0.16, "GRF": 1.12, "C.GRF": 0.27},
    ("aw1", 10, 1600): {"WA-1": 0.68, "WA-2": 0.10, "GRF": 0.80, "C.GRF": 0.20},
    ("aw1", 20, 800): {"WA-1": 0.90, "WA-2": 0.13, "GRF": 1.17, "C.GRF": 0.17},
    ("aw1", 20, 1600): {"WA-1": 0.77, "WA-2": 0.09, "GRF": 0.95, "C.GRF": 0.11},
    ("aw3", 10, 800): {"WA-1": 4.51, "WA-2": 7.67, "GRF": 1.92, "C.GRF": 0.91},
    ("aw3", 10, 1600): {"WA-1": 2.45, "WA-2": 7.94, "GRF": 1.51, "C.GRF": 0.62},
    ("aw3", 20, 800): {"WA-1": 5.93, "WA-2": 8.68, "GRF": 1.92, "C.GRF": 0.93},
    ("aw3", 20, 1600): {"WA-1": 3.54, "WA-2": 8.61, "GRF": 1.55, "C.GRF": 0.57},
}


def build_observations(y: np.ndarray, w: np.ndarray) -> list[Dict[str, float]]:
    """Convert arrays into the model's expected observation dictionaries."""
    return [{"Y": float(y_i), "A": float(w_i)} for y_i, w_i in zip(y, w)]


def crossfit_linear_residuals(
    X: np.ndarray,
    y: np.ndarray,
    folds: int = 2,
    ridge: float = 1e-6,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Residualise a signal using cross-fitted ridge regression."""
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


def _run_single_trial(
    scenario: Scenario,
    test_size: int,
    trees: int,
    max_depth: int,
    q: float,
    xi1: float,
    ridge_eps: float,
    min_leaf: int,
    center: bool,
    seed: int,
) -> float:
    """Execute one Monte Carlo replication for a single scenario."""
    rng_local = np.random.default_rng(seed)
    data = generate_causal_data(
        n=scenario.n,
        p=scenario.p,
        dgp=scenario.dgp,
        sigma_tau=1.0,
        seed=int(rng_local.integers(1_000_000)),
    )
    test = generate_causal_data(
        n=test_size,
        p=scenario.p,
        dgp=scenario.dgp,
        sigma_tau=1.0,
        seed=int(rng_local.integers(1_000_000)),
    )

    y_train = data.Y.copy()
    w_train = data.W.astype(float).copy()
    if center:
        y_train = crossfit_linear_residuals(data.X, y_train, seed=int(rng_local.integers(1_000_000)))
        w_train = crossfit_linear_residuals(data.X, w_train, seed=int(rng_local.integers(1_000_000)))

    idx = rng_local.permutation(scenario.n)
    mid = scenario.n // 2
    d1, d2 = idx[:mid], idx[mid:]

    model = GeneralizedBoostedKernels.fit(
        data.X[d1],
        build_observations(y_train[d1], w_train[d1]),
        data.X[d2],
        build_observations(y_train[d2], w_train[d2]),
        B=trees,
        q=q,
        xi1=xi1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        ridge_eps=ridge_eps,
        seed=int(rng_local.integers(1_000_000)),
    )
    tau_hat = model.predict_theta(test.X)["theta"]
    return float(np.mean((tau_hat - test.tau) ** 2))


def evaluate_scenario(
    scenario: Scenario,
    *,
    reps: int,
    test_size: int,
    trees: int,
    max_depth: int,
    q: float,
    xi1: float,
    ridge_eps: float,
    min_leaf: int,
    seed: int,
    center: bool,
) -> float:
    """Estimate the mean squared error for a single scenario."""
    rng = np.random.default_rng(seed)
    seeds = [int(rng.integers(1_000_000)) for _ in range(reps)]

    mses: list[float] = []
    for s in seeds:
        mses.append(
            _run_single_trial(
                scenario,
                test_size,
                trees,
                max_depth,
                q,
                xi1,
                ridge_eps,
                min_leaf,
                center,
                s,
            )
        )
    return float(np.mean(mses)) * 10.0


def _trial_task(
    payload: Tuple[Scenario, int, int, int, float, float, float, int, bool, int]
) -> Tuple[Tuple[str, int, int], bool, float]:
    """Run a single trial for the given scenario/centre combination."""
    (
        scenario,
        test_size,
        trees,
        max_depth,
        q,
        xi1,
        ridge_eps,
        min_leaf,
        center,
        seed,
    ) = payload
    mse = _run_single_trial(
        scenario,
        test_size,
        trees,
        max_depth,
        q,
        xi1,
        ridge_eps,
        min_leaf,
        center,
        seed,
    )
    return (scenario.dgp, scenario.p, scenario.n), center, mse


def main() -> None:
    """CLI entrypoint for reproducing Table 1 results with Algorithm 1."""
    parser = argparse.ArgumentParser(description="Benchmark Algorithm 1 against Table 1")
    parser.add_argument("--reps", type=int, default=5, help="Monte Carlo repetitions per scenario")
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--q", type=float, default=1.)
    parser.add_argument("--xi1", type=float, default=1.)
    parser.add_argument("--ridge-eps", type=float, default=1e-2)
    parser.add_argument("--min-leaf", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output", type=Path, default=Path("table1_results.json"))
    parser.add_argument("--jobs", type=int, default=31, help="Number of worker processes to parallelize trials")
    args = parser.parse_args()

    scenarios = [
        Scenario(dgp, p, n)
        for dgp in ("aw1", "aw2", "aw3")
        for p in (10, 20)
        for n in (800, 1600)
    ]

    results = {}

    if args.jobs and args.jobs > 1:
        tasks: List[Tuple[Scenario, int, int, int, float, float, float, int, bool, int]] = []
        for sc in scenarios:
            key = (sc.dgp, sc.p, sc.n)
            rng = np.random.default_rng(args.seed + abs(hash(key)))
            seeds = [int(rng.integers(1_000_000)) for _ in range(args.reps)]
            for center_flag in (False, True):
                offset = 7919 if center_flag else 0
                for s in seeds:
                    tasks.append(
                        (
                            sc,
                            args.test_size,
                            args.trees,
                            args.max_depth,
                            args.q,
                            args.xi1,
                            args.ridge_eps,
                            args.min_leaf,
                            center_flag,
                            s + offset,
                        )
                    )

        aggregated: Dict[Tuple[str, int, int], Dict[bool, List[float]]] = defaultdict(lambda: {False: [], True: []})
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_trial_task, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Scenario Trials"):
                scenario_key, center_flag, mse = fut.result()
                aggregated[scenario_key][center_flag].append(mse)

        for scenario_key, buckets in aggregated.items():
            grf_values = buckets[False]
            cgrf_values = buckets[True]
            grf_mse = float(np.mean(grf_values) * 10.0) if grf_values else float("nan")
            cgrf_mse = float(np.mean(cgrf_values) * 10.0) if cgrf_values else float("nan")
            reference = REFERENCE_TABLE1.get(scenario_key, {})
            results[str(scenario_key)] = {
                "reference": reference,
                "ours": {
                    "GRFBoost": grf_mse,
                    "C.GRFBoost": cgrf_mse,
                },
            }
            print(
                f"Scenario {scenario_key}: GRFBoost={grf_mse:.3f} (ref {reference.get('GRF')}), "
                f"C.GRFBoost={cgrf_mse:.3f} (ref {reference.get('C.GRF')})"
            )
    else:
        for sc in tqdm(scenarios, desc="Scenarios"):
            key = (sc.dgp, sc.p, sc.n)
            offset = abs(hash(key)) % 10_000
            grf_mse = evaluate_scenario(
                sc,
                reps=args.reps,
                test_size=args.test_size,
                trees=args.trees,
                max_depth=args.max_depth,
                q=args.q,
                xi1=args.xi1,
                ridge_eps=args.ridge_eps,
                min_leaf=args.min_leaf,
                seed=args.seed + offset,
                center=False,
            )
            cgrf_mse = evaluate_scenario(
                sc,
                reps=args.reps,
                test_size=args.test_size,
                trees=args.trees,
                max_depth=args.max_depth,
                q=args.q,
                xi1=args.xi1,
                ridge_eps=args.ridge_eps,
                min_leaf=args.min_leaf,
                seed=args.seed + offset + 1,
                center=True,
            )
            reference = REFERENCE_TABLE1.get(key, {})
            results[str(key)] = {
                "reference": reference,
                "ours": {
                    "GRFBoost": grf_mse,
                    "C.GRFBoost": cgrf_mse,
                },
            }
            print(
                f"Scenario {key}: GRFBoost={grf_mse:.3f} (ref {reference.get('GRF')}), "
                f"C.GRFBoost={cgrf_mse:.3f} (ref {reference.get('C.GRF')})"
            )

    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved detailed results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
