"""
Stage 1 of Algorithm 1: boosted tree structure learning.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg import pinv as nla_pinv
from numpy.linalg import solve as nla_solve

from .caches import (
    build_current_tree_cache,
    build_prev_tree_cache,
    compute_rho_vector,
    prepare_prev_round_cache,
)
from .split import best_split_for_node
from .tree import Tree, TreeNode, compute_leaf_ids
from .utils import RNG, wls_theta_nu_cate


def fit_stage1_structures(
    X1: np.ndarray,
    O1: List[Dict[str, Any]],
    B: int,
    q: float,
    xi1: float,
    max_depth: int = 5,
    min_leaf: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Learn tree structures using boosting on the split score.
    Returns the learned trees together with dropout bookkeeping.
    """
    rng = RNG(seed)
    n1 = X1.shape[0]
    BT = B
    trees: List[Dict[str, Any]] = [None] * BT
    Sb_list: List[np.ndarray] = [None] * BT

    for b in range(BT):
        # Subsample G_b
        keep = rng.bernoulli_mask(n1, xi1)
        Gb_idx = np.where(keep)[0]
        if Gb_idx.shape[0] < 2 * min_leaf:
            raise RuntimeError("G_b too small; increase xi1 or reduce min_leaf.")
        XGb = X1[Gb_idx, :]
        OGb = [O1[int(i)] for i in Gb_idx]
        m = XGb.shape[0]

        # Dropout S_b \subset {0..b-1}
        cand = np.arange(b)
        if cand.size == 0:
            Sb_idx = np.array([], dtype=int)
        else:
            mask = rng.bernoulli_mask(cand.shape[0], q)
            Sb_idx = cand[mask]
        Sb_list[b] = Sb_idx

        # -------- Estimating step (E): fit thetahat, nuhat using ONLY previous-tree weights --------
        theta_nu_per_i: List[Tuple[float, float]] = [None] * m
        A_all = np.array([og["A"] for og in OGb], dtype=float)
        Y_all = np.array([og["Y"] for og in OGb], dtype=float)
        uniform_theta_nu = wls_theta_nu_cate(A_all, Y_all, np.ones(m, dtype=float))

        Sb_prev_structs: List[Dict[str, Any]] = []
        if Sb_idx.size > 0:
            for bp in Sb_idx:
                prev_info = trees[int(bp)]
                Sb_prev_structs.append(build_prev_tree_cache(prev_info, XGb, A_all, Y_all))

        prev_cache = prepare_prev_round_cache(Sb_prev_structs) if Sb_prev_structs else None

        if prev_cache is not None:
            XtWX = prev_cache["XtWX"]
            XtWy = prev_cache["XtWy"]
            contrib_mask = prev_cache["contrib_mask"]
            signature_betas = np.tile(np.asarray(uniform_theta_nu, dtype=float), (XtWX.shape[0], 1))
            for sig_idx in range(XtWX.shape[0]):
                if not contrib_mask[sig_idx]:
                    continue
                try:
                    beta = nla_solve(XtWX[sig_idx], XtWy[sig_idx])
                except LinAlgError:
                    beta = nla_pinv(XtWX[sig_idx]) @ XtWy[sig_idx]
                signature_betas[sig_idx] = beta
            sig_indices = prev_cache["signature_indices"]
            theta_vals = signature_betas[sig_indices, 0]
            nu_vals = signature_betas[sig_indices, 1]
            for kk in range(m):
                theta_nu_per_i[kk] = (float(theta_vals[kk]), float(nu_vals[kk]))
        else:
            for kk in range(m):
                theta_nu_per_i[kk] = uniform_theta_nu

        # Start with partial tree
        partial = Tree(TreeNode(indices=np.arange(m), depth=0))

        depth = 0
        while True:
            # -------- Split step (ρ): use w_prev + alpha_current for ρ --------
            curr_cache = build_current_tree_cache(partial, XGb, A_all, Y_all)
            rhos = compute_rho_vector(
                A_all,
                Y_all,
                theta_nu_per_i,
                curr_cache,
                prev_cache,
                xi_selector=np.array([1.0, 0.0], dtype=float),
                ridge_eps=1e-8,
            )

            # Frontier at this depth
            frontier: List[TreeNode] = []

            def collect(n: TreeNode) -> None:
                if n.depth == depth:
                    frontier.append(n)
                if not n.is_leaf:
                    collect(n.left)
                    collect(n.right)

            collect(partial.root)

            # Split nodes
            any_split = False
            for node in frontier:
                best = best_split_for_node(node.indices, XGb, rhos, min_leaf)
                if best is None or depth >= max_depth:
                    node.is_leaf = True
                else:
                    node.is_leaf = False
                    node.split_feature = int(best["feature"])
                    node.split_threshold = float(best["threshold"])
                    node.left = TreeNode(indices=best["left"], depth=depth + 1)
                    node.right = TreeNode(indices=best["right"], depth=depth + 1)
                    any_split = True
            if (not any_split) or (depth >= max_depth):
                break
            depth += 1
            # DO NOT re-estimate (thetahat, nuhat) after splits.

        # Finalize this tree; cache leaf IDs on its own G_b for denominators
        leaf_ids_on_Gb = compute_leaf_ids(partial, XGb)
        unique_leaf_ids, counts = np.unique(leaf_ids_on_Gb, return_counts=True)
        leaf_count_map = {int(l): int(c) for l, c in zip(unique_leaf_ids.tolist(), counts.tolist())}
        trees[b] = {
            "tree": partial,
            "Gb_idx": Gb_idx,
            "leaf_ids_on_Gb": leaf_ids_on_Gb,
            "leaf_counts_on_Gb": leaf_count_map,
        }

    return {"trees": trees, "Sb_list": Sb_list}
