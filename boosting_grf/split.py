"""
Split search routines for the boosted tree learner.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .grf import _GRF_AVAILABLE, best_split_regression


def best_split_for_node(
    node_rows: np.ndarray,
    XGb: np.ndarray,
    rhos: np.ndarray,
    min_leaf: int,
) -> Optional[Dict[str, Any]]:
    """Dispatch to the fast GRF-backed splitter when available."""
    if _GRF_AVAILABLE:
        result = best_split_regression(XGb, rhos, node_rows, int(min_leaf))
        if result is None:
            return None
        return {
            "feature": int(result["feature"]),
            "threshold": float(result["threshold"]),
            "left": np.asarray(result["left_indices"], dtype=np.int64),
            "right": np.asarray(result["right_indices"], dtype=np.int64),
        }
    return _best_split_numpy(node_rows, XGb, rhos, min_leaf)


def _best_split_numpy(
    node_rows: np.ndarray,
    XGb: np.ndarray,
    rhos: np.ndarray,
    min_leaf: int,
) -> Optional[Dict[str, Any]]:
    """Pure NumPy fallback split search."""
    if node_rows.shape[0] < 2 * min_leaf:
        return None
    best = None
    best_obj = -np.inf
    p = XGb.shape[1]
    for j in range(p):
        xj = XGb[node_rows, j]
        ord_idx = np.argsort(xj, kind="mergesort")
        rho_sorted = rhos[node_rows][ord_idx]
        x_sorted = xj[ord_idx]
        if x_sorted[0] == x_sorted[-1]:
            continue
        csum = np.cumsum(rho_sorted)
        total = csum[-1]
        n = rho_sorted.shape[0]
        left_counts = np.arange(1, n, dtype=np.int64)
        right_counts = n - left_counts
        diff_mask = np.diff(x_sorted) > 0
        valid_mask = diff_mask & (left_counts >= min_leaf) & (right_counts >= min_leaf)
        if not np.any(valid_mask):
            continue
        sumL = csum[:-1]
        sumR = total - sumL
        obj = np.full(n - 1, -np.inf, dtype=float)
        valid_idx = np.where(valid_mask)[0]
        obj_vals = (sumL[valid_idx] * sumL[valid_idx]) / left_counts[valid_idx]
        obj_vals += (sumR[valid_idx] * sumR[valid_idx]) / right_counts[valid_idx]
        obj[valid_idx] = obj_vals
        local_best = int(np.argmax(obj))
        local_best_obj = float(obj[local_best])
        if local_best_obj > best_obj:
            nL = local_best + 1
            thr = 0.5 * (x_sorted[local_best] + x_sorted[local_best + 1])
            left_idx = node_rows[ord_idx[:nL]]
            right_idx = node_rows[ord_idx[nL:]]
            best_obj = local_best_obj
            best = {"feature": j, "threshold": float(thr), "left": left_idx, "right": right_idx}
    return best
