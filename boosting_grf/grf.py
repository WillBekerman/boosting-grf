"""
Thin Pythonic wrappers around the compiled GRF bindings.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from . import _grf  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - compiled extension missing at runtime
    _GRF_AVAILABLE = False
    _grf = None  # type: ignore[assignment]
else:
    _GRF_AVAILABLE = True


RegressionForest = getattr(_grf, "RegressionForest", None)
CausalForest = getattr(_grf, "CausalForest", None)
Algorithm1Model = getattr(_grf, "Algorithm1Model", None)


def _ensure_available() -> None:
    if not _GRF_AVAILABLE:
        raise RuntimeError(
            "boosting_grf._grf extension is unavailable. "
            "Please build the package with the bundled GRF core."
        )


def train_regression_forest(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    mtry: Optional[int] = None,
    num_trees: int = 2000,
    min_node_size: int = 5,
    sample_fraction: float = 0.5,
    honesty: bool = True,
    honesty_fraction: float = 0.5,
    honesty_prune_leaves: bool = True,
    ci_group_size: int = 1,
    alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    compute_oob_predictions: bool = False,
    num_threads: int = 0,
    seed: int = 0,
    legacy_seed: bool = False,
):
    """Train a regression forest via the GRF C++ core."""
    _ensure_available()
    X_arr = np.asarray(X, dtype=np.float64, order="C")
    y_arr = np.asarray(y, dtype=np.float64, order="C").reshape(-1)
    sample_w = None
    if sample_weight is not None:
        sample_w = np.asarray(sample_weight, dtype=np.float64, order="C").reshape(-1)
    return _grf.train_regression_forest(
        X_arr,
        y_arr,
        sample_w,
        mtry,
        num_trees,
        min_node_size,
        float(sample_fraction),
        bool(honesty),
        float(honesty_fraction),
        bool(honesty_prune_leaves),
        int(ci_group_size),
        float(alpha),
        float(imbalance_penalty),
        bool(compute_oob_predictions),
        int(num_threads),
        int(seed),
        bool(legacy_seed),
    )


def predict_regression_forest(
    forest,
    X_test: np.ndarray,
    *,
    estimate_variance: bool = False,
    num_threads: int = 0,
):
    """Predict with a trained regression forest."""
    _ensure_available()
    X_arr = np.asarray(X_test, dtype=np.float64, order="C")
    return forest.predict(X_arr, bool(estimate_variance), int(num_threads))


def predict_regression_forest_oob(
    forest,
    *,
    estimate_variance: bool = False,
    num_threads: int = 0,
):
    """Return OOB predictions for a trained regression forest."""
    _ensure_available()
    return forest.predict_oob(bool(estimate_variance), int(num_threads))


def train_algorithm1(
    X1: np.ndarray,
    A1: np.ndarray,
    Y1: np.ndarray,
    X2: np.ndarray,
    A2: np.ndarray,
    Y2: np.ndarray,
    *,
    B: int,
    q: float,
    xi1: float,
    max_depth: int = 5,
    min_leaf: int = 10,
    ridge_eps: float = 1e-8,
    sample_fraction: float = 1.0,
    honesty: bool = False,
    honesty_fraction: float = 0.5,
    honesty_prune_leaves: bool = False,
    seed: Optional[int] = None,
):
    """Train the full Algorithm 1 pipeline using the GRF backend."""
    _ensure_available()
    if Algorithm1Model is None:
        raise RuntimeError("Algorithm1Model binding is unavailable in this build.")
    X1_arr = np.asarray(X1, dtype=np.float64, order="C")
    X2_arr = np.asarray(X2, dtype=np.float64, order="C")
    A1_arr = np.asarray(A1, dtype=np.float64, order="C").reshape(-1)
    Y1_arr = np.asarray(Y1, dtype=np.float64, order="C").reshape(-1)
    A2_arr = np.asarray(A2, dtype=np.float64, order="C").reshape(-1)
    Y2_arr = np.asarray(Y2, dtype=np.float64, order="C").reshape(-1)
    seed_arg = None if seed is None else int(seed)
    return _grf.train_algorithm1(
        X1_arr,
        A1_arr,
        Y1_arr,
        X2_arr,
        A2_arr,
        Y2_arr,
        int(B),
        float(q),
        float(xi1),
        int(max_depth),
        int(min_leaf),
        float(ridge_eps),
        float(sample_fraction),
        bool(honesty),
        float(honesty_fraction),
        bool(honesty_prune_leaves),
        seed_arg,
    )


def predict_algorithm1(model, X_new: np.ndarray):
    """Predict `(θ, ν)` using a GRF-backed Algorithm 1 model."""
    _ensure_available()
    if Algorithm1Model is None:
        raise RuntimeError("Algorithm1Model binding is unavailable in this build.")
    X_arr = np.asarray(X_new, dtype=np.float64, order="C")
    return model.predict_theta(X_arr)


def train_causal_forest(
    X: np.ndarray,
    y: np.ndarray,
    treatment: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    mtry: Optional[int] = None,
    num_trees: int = 2000,
    min_node_size: int = 5,
    sample_fraction: float = 0.5,
    honesty: bool = True,
    honesty_fraction: float = 0.5,
    honesty_prune_leaves: bool = True,
    ci_group_size: int = 1,
    reduced_form_weight: float = 0.0,
    alpha: float = 0.05,
    imbalance_penalty: float = 0.0,
    stabilize_splits: bool = True,
    compute_oob_predictions: bool = False,
    num_threads: int = 0,
    seed: int = 0,
    legacy_seed: bool = False,
):
    """Train a causal forest for heterogeneous treatment effect estimation."""
    _ensure_available()
    X_arr = np.asarray(X, dtype=np.float64, order="C")
    y_arr = np.asarray(y, dtype=np.float64, order="C").reshape(-1)
    t_arr = np.asarray(treatment, dtype=np.float64, order="C").reshape(-1)
    sample_w = None
    if sample_weight is not None:
        sample_w = np.asarray(sample_weight, dtype=np.float64, order="C").reshape(-1)
    return _grf.train_causal_forest(
        X_arr,
        y_arr,
        t_arr,
        sample_w,
        mtry,
        num_trees,
        min_node_size,
        float(sample_fraction),
        bool(honesty),
        float(honesty_fraction),
        bool(honesty_prune_leaves),
        int(ci_group_size),
        float(reduced_form_weight),
        float(alpha),
        float(imbalance_penalty),
        bool(stabilize_splits),
        bool(compute_oob_predictions),
        int(num_threads),
        int(seed),
        bool(legacy_seed),
    )


def predict_causal_forest(
    forest,
    X_test: np.ndarray,
    *,
    estimate_variance: bool = False,
    num_threads: int = 0,
):
    """Predict treatment effects using a trained causal forest."""
    _ensure_available()
    X_arr = np.asarray(X_test, dtype=np.float64, order="C")
    return forest.predict(X_arr, bool(estimate_variance), int(num_threads))


def predict_causal_forest_oob(
    forest,
    *,
    estimate_variance: bool = False,
    num_threads: int = 0,
):
    """Out-of-bag estimates for a causal forest."""
    _ensure_available()
    return forest.predict_oob(bool(estimate_variance), int(num_threads))


def best_split_regression(
    X: np.ndarray,
    rho: np.ndarray,
    node_indices: np.ndarray,
    min_leaf: int,
):
    """Wrapper around the C++ CART split search used in Algorithm 1."""
    _ensure_available()
    X_f = np.asarray(X, dtype=np.float64, order="F")
    rho_arr = np.asarray(rho, dtype=np.float64, order="C").reshape(-1)
    node_arr = np.asarray(node_indices, dtype=np.int64, order="C").reshape(-1)
    return _grf.best_split_regression(X_f, rho_arr, node_arr, int(min_leaf))


__all__ = [
    "RegressionForest",
    "CausalForest",
    "Algorithm1Model",
    "train_regression_forest",
    "predict_regression_forest",
    "predict_regression_forest_oob",
    "train_algorithm1",
    "predict_algorithm1",
    "train_causal_forest",
    "predict_causal_forest",
    "predict_causal_forest_oob",
    "best_split_regression",
    "_GRF_AVAILABLE",
]
