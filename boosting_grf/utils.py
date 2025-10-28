"""
Utility helpers shared across Algorithm 1 components.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg import pinv as nla_pinv
from numpy.linalg import solve as nla_solve


class RNG:
    """Lightweight wrapper around NumPy's generator to centralize seeding."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def bernoulli_mask(self, n: int, p: float) -> np.ndarray:
        """Draw n Bernoulli(p) trials and return a boolean mask."""
        return self.rng.random(n) < p

    def choice(self, n: int, size: int, replace: bool) -> np.ndarray:
        """Sample integer indices from {0, …, n-1}."""
        return self.rng.choice(np.arange(n), size=size, replace=replace)


def wls_theta_nu_cate(A: np.ndarray, Y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Solve the weighted CATE moment equations for θ and ν within a leaf."""
    X = np.column_stack([A, np.ones_like(A)])  # [A, 1]
    XtW = X.T * w
    XtWX = XtW @ X
    XtWy = XtW @ Y
    try:
        beta = nla_solve(XtWX, XtWy)
    except LinAlgError:
        beta = nla_pinv(XtWX) @ XtWy
    theta, nu = float(beta[0]), float(beta[1])
    return theta, nu
