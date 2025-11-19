"""
Backward-compatible entry point that delegates to the refactored boosting_grf package.

Public API:
    model = GeneralizedBoostedKernels.fit(...)
    pred  = model.predict_theta(...)
"""
from __future__ import annotations

import numpy as np

from boosting_grf import GeneralizedBoostedKernels

__all__ = ["GeneralizedBoostedKernels"]


if __name__ == "__main__":
    seed = 0
    rng = np.random.default_rng(seed)
    n = 2000
    p = 3
    X = rng.normal(size=(n, p))

    def tau_true(x):
        return 1.0 + x[0] - 0.5 * x[1]

    def f_base(x):
        return np.sin(x[0]) + x[1] ** 2 - x[2]

    pis = 1.0 / (1.0 + np.exp(-(0.2 + 0.5 * X[:, 0] - 0.3 * X[:, 2])))
    A = rng.binomial(1, pis)
    Y = np.apply_along_axis(tau_true, 1, X) * A + np.apply_along_axis(f_base, 1, X) + rng.normal(size=n)

    idx = rng.permutation(n)
    train = idx[: int(0.8 * n)]
    test = idx[int(0.8 * n) :]

    D1 = train[: len(train) // 2]
    D2 = np.setdiff1d(train, D1, assume_unique=False)

    X1, X2 = X[D1, :], X[D2, :]
    O_list = [{"Y": float(Y[i]), "A": float(A[i])} for i in range(n)]
    O1 = [O_list[int(i)] for i in D1]
    O2 = [O_list[int(i)] for i in D2]

    B = 16
    model = GeneralizedBoostedKernels.fit(
        X1,
        O1,
        X2,
        O2,
        B=B,
        q=0.6,
        xi1=0.6,
        max_depth=3,
        min_leaf=20,
        seed=seed,
    )
    pred = model.predict_theta(X[test, :])
    tau_hat = pred["theta"]
    tau_true_vec = np.apply_along_axis(lambda r: 1.0 + r[0] - 0.5 * r[1], 1, X[test, :])
    rmse = float(np.sqrt(np.mean((tau_hat - tau_true_vec) ** 2)))
    print("[Alg1:] Test RMSE for tau(x):", f"{rmse:.4f}")
