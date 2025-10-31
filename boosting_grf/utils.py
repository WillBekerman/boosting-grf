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
    """Random-number convenience wrapper.

    Args:
        seed: Optional seed for the underlying `default_rng`.

    Attributes:
        rng: The NumPy `Generator` used for sampling.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def bernoulli_mask(self, n: int, p: float) -> np.ndarray:
        """Sample an indicator mask from a Bernoulli distribution.

        Args:
            n: Number of trials to draw.
            p: Success probability in the open interval (0, 1).

        Returns:
            A boolean NumPy array of shape `(n,)` indicating selected elements.
        """
        return self.rng.random(n) < p

    def choice(self, n: int, size: int, replace: bool) -> np.ndarray:
        """Sample indices without managing manual range checks.

        Args:
            n: Upper-bound (exclusive) for the population integers.
            size: Number of samples to draw.
            replace: Whether to sample with replacement.

        Returns:
            A NumPy array of integer indices sampled from `{0, …, n-1}`.
        """
        return self.rng.choice(np.arange(n), size=size, replace=replace)


def wls_theta_nu_cate(A: np.ndarray, Y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Solve the local weighted least-squares problem for CATE parameters.

    Args:
        A: Treatment indicators for the nodes being considered.
        Y: Observed outcomes for the same nodes.
        w: Non-negative weights applied to each observation.

    Returns:
        A tuple `(theta, nu)` representing the slope and intercept of the local
        CATE estimating equation.

    Raises:
        ValueError: If the observation arrays are not one-dimensional or have
            mismatched shapes.
    """
    if A.ndim != 1 or Y.ndim != 1 or w.ndim != 1:
        raise ValueError("A, Y, and w must be one-dimensional arrays.")
    if not (A.shape == Y.shape == w.shape):
        raise ValueError("A, Y, and w must share the same shape.")

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
