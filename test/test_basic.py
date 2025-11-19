import numpy as np

from boosting_grf import GeneralizedBoostedKernels
from boosting_grf.datasets import generate_causal_data


def build_observations(y, w):
    return [{"Y": float(y_i), "A": float(w_i)} for y_i, w_i in zip(y, w)]


def test_small_fit_runs():
    data = generate_causal_data(n=200, p=6, dgp="aw3", sigma_tau=1.0, seed=0)
    idx = np.random.default_rng(1).permutation(data.X.shape[0])
    mid = data.X.shape[0] // 2
    d1, d2 = idx[:mid], idx[mid:]
    model = GeneralizedBoostedKernels.fit(
        data.X[d1],
        build_observations(data.Y[d1], data.W[d1]),
        data.X[d2],
        build_observations(data.Y[d2], data.W[d2]),
        B=8,
        q=0.5,
        xi1=0.8,
        max_depth=2,
        min_leaf=15,
        ridge_eps=1e-6,
        seed=5,
    )
    preds = model.predict_theta(data.X[:5])
    assert preds["theta"].shape == (5,)
    assert np.isfinite(preds["theta"]).any()


def test_generate_causal_data_shapes():
    data = generate_causal_data(n=50, p=5, dgp="aw2", sigma_tau=1.0, seed=2)
    assert data.X.shape == (50, 5)
    assert data.Y.shape == (50,)
    assert data.W.shape == (50,)
    assert np.all((data.X >= 0.0) & (data.X <= 1.0))
