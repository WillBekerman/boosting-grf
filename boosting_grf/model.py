"""
High-level orchestration and public API for Algorithm 1 backed by GRF.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import grf


def _observations_to_arrays(obs: List[Dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert observation dictionaries into aligned arrays."""
    A = np.array([row["A"] for row in obs], dtype=float)
    Y = np.array([row["Y"] for row in obs], dtype=float)
    return A, Y


class GeneralizedBoostedKernels:
    """GRF-backed Algorithm 1 estimator."""

    def __init__(self, grf_model: grf.Algorithm1Model):
        self._grf_model = grf_model

    def predict_theta(self, Xnew: np.ndarray) -> Dict[str, np.ndarray]:
        """Estimate `(θ(x), ν(x))` for new feature points."""
        Xnew = np.asarray(Xnew, dtype=float, order="C")
        raw = grf.predict_algorithm1(self._grf_model, Xnew)
        theta = np.asarray(raw["theta"], dtype=float)
        nu = np.asarray(raw["nu"], dtype=float)
        return {"theta": theta, "nu": nu}

    @classmethod
    def fit(
        cls,
        X1: np.ndarray,
        O1: List[Dict[str, float]],
        X2: np.ndarray,
        O2: List[Dict[str, float]],
        B: int,
        M: Optional[int] = None,
        q: float = 0.6,
        xi1: float = 0.6,
        xi2: Optional[float] = None,
        max_depth: int = 5,
        min_leaf: int = 10,
        ridge_eps: float = 1e-8,
        seed: Optional[int] = None,
        sample_fraction: float = 1.0,
        honesty: bool = False,
        honesty_fraction: float = 1.0,
        honesty_prune_leaves: bool = False,
    ) -> "GeneralizedBoostedKernels":
        """Fit Algorithm 1 using the GRF backend."""
        if not grf._GRF_AVAILABLE or grf.Algorithm1Model is None:
            raise RuntimeError(
                "The GRF extension is required to train Algorithm 1. "
                "Please build the package with the bundled GRF core."
            )

        X1 = np.asarray(X1, dtype=float, order="C")
        X2 = np.asarray(X2, dtype=float, order="C")
        A1_vec, Y1_vec = _observations_to_arrays(O1)
        A2_vec, Y2_vec = _observations_to_arrays(O2)

        grf_model = grf.train_algorithm1(
            X1,
            A1_vec,
            Y1_vec,
            X2,
            A2_vec,
            Y2_vec,
            B=B,
            q=q,
            xi1=xi1,
            max_depth=max_depth,
            min_leaf=min_leaf,
            ridge_eps=ridge_eps,
            sample_fraction=sample_fraction,
            honesty=honesty,
            honesty_fraction=honesty_fraction,
            honesty_prune_leaves=honesty_prune_leaves,
            seed=seed,
        )
        return cls(grf_model)
