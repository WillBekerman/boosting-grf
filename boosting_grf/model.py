"""
High-level orchestration and public API for Algorithm 1.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .stage1 import fit_stage1_structures
from .stage2 import compute_beta_weights, fit_stage2_grf
from .utils import wls_theta_nu_cate


class Alg1GRFModel:
    """Convenience wrapper that exposes prediction helpers for Algorithm 1.

    Args:
        model: Dictionary containing Stage 1/Stage 2 artefacts plus raw data
            slices. The dictionary is typically returned by :func:`fit_alg1_grf`.
    """

    def __init__(self, model: Dict[str, Any]):
        self.__dict__.update(model)

    def predict_theta(self, Xnew: np.ndarray) -> Dict[str, np.ndarray]:
        """Estimate `(θ(x), ν(x))` for new feature points.

        Args:
            Xnew: Feature matrix of shape `(n, p)` for which predictions are
                required.

        Returns:
            A dictionary containing:
                - ``"theta"``: Estimated heterogeneous treatment effects.
                - ``"nu"``: Estimated nuisance intercept terms.
        """
        Xnew = np.asarray(Xnew, dtype=float)
        n = Xnew.shape[0]
        A2 = np.array([oi["A"] for oi in self.O2], dtype=float)
        Y2 = np.array([oi["Y"] for oi in self.O2], dtype=float)
        theta = np.full(n, np.nan, dtype=float)
        nu = np.full(n, np.nan, dtype=float)
        for i in range(n):
            # Compute β-weights for the query point using Stage 2 caches.
            beta = compute_beta_weights(self.__dict__, Xnew[i, :])
            idx = np.where(beta > 0)[0]
            if idx.size == 0:
                continue
            w = beta[idx]
            th, nv = wls_theta_nu_cate(A2[idx], Y2[idx], w)
            theta[i] = th
            nu[i] = nv
        return {"theta": theta, "nu": nu}


def fit_alg1_grf(
    X1: np.ndarray,
    O1: List[Dict[str, Any]],
    X2: np.ndarray,
    O2: List[Dict[str, Any]],
    B: int,
    M: Optional[int] = None,
    q: float = 0.6,
    xi1: float = 0.6,
    xi2: Optional[float] = None,
    max_depth: int = 5,
    min_leaf: int = 10,
    ridge_eps: float = 1e-8,
    seed: Optional[int] = None,
) -> Alg1GRFModel:
    """Fit Algorithm 1 using disjoint structure-learning and GRF samples.

    Args:
        X1: Feature matrix for the structure-learning sample `D1`.
        O1: Observation list for `D1` containing `"A"` and `"Y"` keys.
        X2: Feature matrix for the GRF sample `D2`.
        O2: Observation list for `D2` (mirrors `O1`).
        B: Number of boosting rounds to perform in Stage 1.
        M: Reserved for Boulevard-style extensions (unused in Algorithm 1).
        q: Dropout probability within each boosting round.
        xi1: Subsampling rate for `G_b` in Stage 1.
        xi2: Placeholder for Stage 2 subsampling (unused in Algorithm 1).
        max_depth: Maximum depth of each tree in Stage 1.
        min_leaf: Minimum number of examples per leaf.
        ridge_eps: Ridge regularisation passed to the ρ computation.
        seed: Optional seed applied to both stages for reproducibility.

    Returns:
        An :class:`Alg1GRFModel` instance capable of predicting CATEs.
    """
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    stage1 = fit_stage1_structures(
        X1,
        O1,
        B=B,
        q=q,
        xi1=xi1,
        max_depth=max_depth,
        min_leaf=min_leaf,
        ridge_eps=ridge_eps,
        seed=seed,
    )
    stage2 = fit_stage2_grf(
        X2,
        stage1["trees"],
        Sb_list=stage1["Sb_list"],
        M=M,
        xi2=xi2,
        seed=seed,
    )
    # Package artefacts into a lightweight model object.
    return Alg1GRFModel(
        {
            "X2": X2,
            "O2": O2,
            "stage1": stage1,
            "stage2": stage2,
        }
    )
