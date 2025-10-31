"""
Leaf-level sufficient statistics and dropout cache utilities.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .tree import Tree, compute_leaf_ids


def summarize_leaf_contrib(
    member_idx: np.ndarray,
    weight: float,
    A_all: np.ndarray,
    Y_all: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Compute per-leaf aggregates used in Algorithm 1 updates.

    Args:
        member_idx: Indices of `G_b` observations contained in the leaf.
        weight: Per-observation α-weight contribution from the leaf.
        A_all: Treatment indicators for all rows in `G_b`.
        Y_all: Outcome values for all rows in `G_b`.

    Returns:
        A dictionary containing weighted sums and Jacobian contributions, or
        ``None`` if the leaf has no members or zero weight.
    """
    count = int(member_idx.size)
    if count == 0 or weight <= 0.0:
        return None
    A_slice = A_all[member_idx]
    Y_slice = Y_all[member_idx]
    sum_A = float(np.sum(A_slice))
    sum_Y = float(np.sum(Y_slice))
    sum_A2 = float(np.sum(A_slice * A_slice))
    sum_AY = float(np.sum(A_slice * Y_slice))
    jac = weight * np.array(
        [[-sum_A2, -sum_A],
         [-sum_A,  -count]],
        dtype=float,
    )
    return {
        "weight": weight,
        "count": count,
        "sum_A": sum_A,
        "sum_Y": sum_Y,
        "sum_A2": sum_A2,
        "sum_AY": sum_AY,
        "M": jac,
        "members": member_idx,
    }


def build_prev_tree_cache(
    prev_tree_info: Dict[str, Any],
    XGb: np.ndarray,
    A_all: np.ndarray,
    Y_all: np.ndarray,
) -> Dict[str, Any]:
    """Pre-compute previous-tree statistics on the current subsample.

    Args:
        prev_tree_info: Metadata dictionary produced during Stage 1.
        XGb: Feature matrix for the current subsample `G_b`.
        A_all: Treatment indicators for the subsample.
        Y_all: Outcomes for the subsample.

    Returns:
        A dictionary mapping leaf identifiers to cached statistics required to
        build dropout-weight contributions.
    """
    tree = prev_tree_info["tree"]
    leaf_ids_on_cur = compute_leaf_ids(tree, XGb)
    leaf_counts_map = prev_tree_info.get("leaf_counts_on_Gb", {})
    leaf_stats: Dict[int, Dict[str, Any]] = {}
    unique_leaf_ids = np.unique(leaf_ids_on_cur)
    for lid in unique_leaf_ids:
        denom = int(leaf_counts_map.get(int(lid), 0))
        if denom <= 0:
            continue
        members = np.where(leaf_ids_on_cur == lid)[0]
        stats = summarize_leaf_contrib(members, 1.0 / float(denom), A_all, Y_all)
        if stats is not None:
            leaf_stats[int(lid)] = stats
    return {
        "leaf_ids_on_cur": leaf_ids_on_cur,
        "leaf_stats": leaf_stats,
    }


def build_current_tree_cache(
    partial_tree: Tree,
    XGb: np.ndarray,
    A_all: np.ndarray,
    Y_all: np.ndarray,
) -> Dict[str, Any]:
    """Cache α-weight statistics for the partially grown tree.

    Args:
        partial_tree: The tree being grown in the current boosting round.
        XGb: Feature matrix of the subsample.
        A_all: Treatment vector aligned with `XGb`.
        Y_all: Outcome vector aligned with `XGb`.

    Returns:
        A dictionary with per-leaf membership arrays and aggregated statistics.
    """
    leaf_ids = compute_leaf_ids(partial_tree, XGb)
    leaf_stats: Dict[int, Dict[str, Any]] = {}
    unique_leaf_ids = np.unique(leaf_ids)
    for lid in unique_leaf_ids:
        members = np.where(leaf_ids == lid)[0]
        count = int(members.size)
        if count == 0:
            continue
        weight = 1.0 / float(count)
        stats = summarize_leaf_contrib(members, weight, A_all, Y_all)
        if stats is not None:
            leaf_stats[int(lid)] = stats
    return {
        "leaf_ids": leaf_ids,
        "leaf_stats": leaf_stats,
    }


def aggregate_signature_stats(
    signature: np.ndarray,
    prev_structs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate statistics across a unique dropout signature.

    Args:
        signature: One row of the signature matrix produced by
            :func:`prepare_prev_round_cache`.
        prev_structs: Cached statistics for the previous trees included in the
            dropout set `S_b`.

    Returns:
        A dictionary containing cumulative moments and Jacobian contributions.
    """
    w_sum = 0.0
    wA_sum = 0.0
    wA2_sum = 0.0
    wY_sum = 0.0
    wAY_sum = 0.0
    jac = np.zeros((2, 2), dtype=float)
    for t_idx, leaf_id in enumerate(signature):
        stats = prev_structs[t_idx]["leaf_stats"].get(int(leaf_id))
        if stats is None:
            continue
        weight = stats["weight"]
        count = stats["count"]
        w_sum += weight * count
        wA_sum += weight * stats["sum_A"]
        wA2_sum += weight * stats["sum_A2"]
        wY_sum += weight * stats["sum_Y"]
        wAY_sum += weight * stats["sum_AY"]
        jac += stats["M"]
    XtWX = np.array([[wA2_sum, wA_sum], [wA_sum, w_sum]], dtype=float)
    XtWy = np.array([wAY_sum, wY_sum], dtype=float)
    return {
        "XtWX": XtWX,
        "XtWy": XtWy,
        "M": jac,
        "has_contrib": bool(w_sum > 0.0),
    }


def prepare_prev_round_cache(
    prev_structs: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build a cache for unique dropout signatures.

    Args:
        prev_structs: List of per-tree caches produced by
            :func:`build_prev_tree_cache`.

    Returns:
        A dictionary containing aggregated statistics keyed by signature, or
        ``None`` if the dropout set is empty for the current round.
    """
    if not prev_structs:
        return None
    leaf_id_matrix = np.column_stack([s["leaf_ids_on_cur"] for s in prev_structs])
    unique_signatures, inverse = np.unique(leaf_id_matrix, axis=0, return_inverse=True)
    signature_stats = [aggregate_signature_stats(sig, prev_structs) for sig in unique_signatures]
    signature_M = np.stack([s["M"] for s in signature_stats], axis=0)
    XtWX = np.stack([s["XtWX"] for s in signature_stats], axis=0)
    XtWy = np.stack([s["XtWy"] for s in signature_stats], axis=0)
    contrib_mask = np.array([s["has_contrib"] for s in signature_stats], dtype=bool)
    return {
        "structures": prev_structs,
        "unique_signatures": unique_signatures,
        "signature_indices": inverse,
        "XtWX": XtWX,
        "XtWy": XtWy,
        "signature_M": signature_M,
        "contrib_mask": contrib_mask,
    }


def compute_rho_vector(
    A_all: np.ndarray,
    Y_all: np.ndarray,
    theta_nu_per_i: List[Tuple[float, float]],
    curr_cache: Dict[str, Any],
    prev_cache: Optional[Dict[str, Any]],
    xi_selector: np.ndarray,
    ridge_eps: float = 1e-8,
) -> np.ndarray:
    """Compute the gradient-based splitting residuals `ρ_i`.

    Implements Equation (9) in *causalgrf.pdf* using cached statistics to avoid
    repeated traversals.

    Args:
        A_all: Treatment indicators for the current subsample.
        Y_all: Outcome vector for the same subsample.
        theta_nu_per_i: List of `(theta, nu)` tuples from the estimating step.
        curr_cache: Cache produced by :func:`build_current_tree_cache`.
        prev_cache: Cache produced by :func:`prepare_prev_round_cache` or ``None``.
        xi_selector: Selector vector for the component of the score used when
            projecting onto the boosting direction.
        ridge_eps: Ridge regularisation strength added to the diagonal of the
            Jacobian to preserve numerical stability.

    Returns:
        A NumPy array of shape `(m,)` containing the residuals `ρ_i` for all rows
        in the current subsample.
    """
    m = A_all.shape[0]
    theta_arr = np.array([tn[0] for tn in theta_nu_per_i], dtype=float)
    nu_arr = np.array([tn[1] for tn in theta_nu_per_i], dtype=float)
    resid = Y_all - A_all * theta_arr - nu_arr
    v = np.column_stack((resid * A_all, resid))

    if prev_cache is not None:
        prev_M = prev_cache["signature_M"][prev_cache["signature_indices"]]
    else:
        prev_M = np.zeros((m, 2, 2), dtype=float)

    curr_leaf_stats = curr_cache["leaf_stats"]
    curr_M = np.zeros_like(prev_M)
    for stats in curr_leaf_stats.values():
        curr_M[stats["members"]] = stats["M"]

    M_total = prev_M + curr_M
    if ridge_eps > 0.0:
        M_total[:, 0, 0] += ridge_eps
        M_total[:, 1, 1] += ridge_eps

    a = M_total[:, 0, 0]
    b = M_total[:, 0, 1]
    c = M_total[:, 1, 0]
    d = M_total[:, 1, 1]
    det = a * d - b * c
    det[np.abs(det) < ridge_eps] = ridge_eps

    v0 = v[:, 0]
    v1 = v[:, 1]
    u0 = (d * v0 - b * v1) / det
    u1 = (-c * v0 + a * v1) / det

    rhos = -(xi_selector[0] * u0 + xi_selector[1] * u1)
    return rhos.astype(float, copy=False)
