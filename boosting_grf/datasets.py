"""Data generation utilities matching benchmarks from generalized random forest literature."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import math
import numpy as np

DGPName = Literal["aw1", "aw2", "aw3"]


@dataclass
class CausalData:
    X: np.ndarray
    Y: np.ndarray
    W: np.ndarray
    tau: np.ndarray
    m: np.ndarray
    e: np.ndarray
    dgp: str


def _beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate the Beta(a, b) PDF for the supplied vector.

    Args:
        x: Points in the unit interval.
        a: First shape parameter (>0).
        b: Second shape parameter (>0).

    Returns:
        The pointwise PDF evaluations.
    """
    coef = math.gamma(a + b) / (math.gamma(a) * math.gamma(b))
    return coef * np.power(x, a - 1) * np.power(1.0 - x, b - 1)


def _zeta(u: np.ndarray) -> np.ndarray:
    """Logistic bump used in the AW2/AW3 DGPs."""
    return 1.0 + 1.0 / (1.0 + np.exp(-20.0 * (u - (1.0 / 3.0))))


def generate_causal_data(
    n: int,
    p: int,
    dgp: DGPName,
    sigma_m: float = 1.0,
    sigma_tau: float = 0.1,
    sigma_noise: float = 1.0,
    seed: Optional[int] = None,
) -> CausalData:
    """Generate synthetic data mirroring the GRF benchmark DGPs.

    Args:
        n: Number of observations to draw.
        p: Number of covariates.
        dgp: Identifier (``"aw1"``, ``"aw2"``, or ``"aw3"``).
        sigma_m: Target standard deviation for the baseline function.
        sigma_tau: Target standard deviation for the treatment effect.
        sigma_noise: Conditional noise standard deviation.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A :class:`CausalData` instance containing covariates, treatment, outcomes,
        and nuisance components used in the benchmarking scripts.

    Raises:
        ValueError: If an unsupported DGP identifier is supplied.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, p))

    if dgp == "aw1":
        tau = np.zeros(n)
        e = 0.25 * (1.0 + _beta_pdf(X[:, 0], 2.0, 4.0))
    else:
        zeta1 = _zeta(X[:, 0])
        zeta2 = _zeta(X[:, 1])
        tau = zeta1 * zeta2
        if dgp == "aw2":
            e = np.full(n, 0.5)
        elif dgp == "aw3":
            e = 0.25 * (1.0 + _beta_pdf(X[:, 0], 2.0, 4.0))
        else:
            raise ValueError(f"Unsupported DGP '{dgp}'.")

    if dgp == "aw1":
        m = 2.0 * X[:, 0] - 1.0 + e * tau
    elif dgp == "aw2":
        m = e * tau
    elif dgp == "aw3":
        m = 2.0 * X[:, 0] - 1.0 + e * tau
    else:
        raise ValueError(f"Unsupported DGP '{dgp}'.")

    tau = _rescale_component(tau, sigma_tau)
    m = _rescale_component(m, sigma_m)
    V = np.ones(n) * (sigma_noise ** 2)

    W = rng.binomial(1, np.clip(e, 1e-6, 1 - 1e-6))
    noise = rng.normal(scale=np.sqrt(V))
    Y = m + (W - e) * tau + noise

    return CausalData(X=X, Y=Y, W=W, tau=tau, m=m, e=e, dgp=dgp)


def _rescale_component(component: np.ndarray, target_sd: float) -> np.ndarray:
    """Rescale a vector to a requested standard deviation.

    Args:
        component: Input array to be rescaled.
        target_sd: Desired standard deviation.

    Returns:
        The rescaled array (or the original if the variance is zero).
    """
    std = np.std(component)
    if std == 0:
        return component
    return component / std * target_sd
