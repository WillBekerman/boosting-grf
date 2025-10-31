"""
Stage 2 of Algorithm 1: build β-weights over the GRF sample.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .tree import compute_leaf_ids, locate_leaf_id


def fit_stage2_grf(
    X2: np.ndarray,
    trees: List[Dict[str, Any]],
    Sb_list: List[np.ndarray],
    M: Optional[int] = None,
    xi2: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Pre-compute Stage 2 leaf membership and weights.

    Args:
        X2: Feature matrix for the GRF sample `D2`.
        trees: List of dictionaries returned by Stage 1.
        Sb_list: Dropout sets corresponding to each Stage 1 tree.
        M: Unused placeholder for future extensions (mirrors literature).
        xi2: Optional Stage 2 subsampling rate (currently unused for Algorithm 1).
        seed: Optional seed for reproducibility (unused but kept for symmetry).

    Returns:
        A dictionary containing per-tree leaf-membership indices and weights that
        can be reused when computing β-weights at prediction time.

    Raises:
        RuntimeError: If Stage 1 produced no trees.
    """
    n2 = X2.shape[0]
    BT = len(trees)  # equals B
    if BT == 0:
        raise RuntimeError("No trees from Stage 1.")
    leaf_members_per_tree: List[Dict[int, np.ndarray]] = []
    leaf_weights_per_tree: List[Dict[int, float]] = []
    for b in range(BT):
        lids = compute_leaf_ids(trees[b]["tree"], X2)
        members_map: Dict[int, List[int]] = {}
        for idx, lid in enumerate(lids):
            lid_int = int(lid)
            members_map.setdefault(lid_int, []).append(idx)
        weight_map: Dict[int, float] = {}
        members_map_np: Dict[int, np.ndarray] = {}
        for lid_int, members in members_map.items():
            arr = np.asarray(members, dtype=np.int64)
            members_map_np[lid_int] = arr
            weight_map[lid_int] = 1.0 / float(arr.size)
        leaf_members_per_tree.append(members_map_np)
        leaf_weights_per_tree.append(weight_map)
    return {
        "leaf_members_per_tree": leaf_members_per_tree,
        "leaf_weights_per_tree": leaf_weights_per_tree,
        "Sb_list": Sb_list,
    }


def compute_beta_weights(model: Dict[str, Any], x_new: np.ndarray) -> np.ndarray:
    """Compute β-weights for a new query point following Algorithm 1.

    Args:
        model: A model dictionary constructed by :func:`fit_alg1_grf`.
        x_new: Feature vector for which the β-weights are to be evaluated.

    Returns:
        A weight vector over the Stage 2 dataset `D2`, normalised to sum to one.
    """
    X2 = model["X2"]
    stage1 = model["stage1"]
    trees = stage1["trees"]
    Sb_list = stage1["Sb_list"]
    stage2 = model["stage2"]
    leaf_members_per_tree = stage2["leaf_members_per_tree"]
    leaf_weights_per_tree = stage2["leaf_weights_per_tree"]

    n2 = X2.shape[0]
    BT = len(trees)  # equals B
    betas = np.zeros(n2, dtype=float)

    def add_alpha_from_tree(t_index: int, lid_q: int):
        members_map = leaf_members_per_tree[t_index]
        weight_map = leaf_weights_per_tree[t_index]
        members = members_map.get(int(lid_q))
        if members is not None:
            # Uniform contribution within the leaf; accumulate in-place.
            betas[members] += weight_map[int(lid_q)]

    for b in range(BT):
        tree_b = trees[b]["tree"]
        lid_q_b = locate_leaf_id(tree_b, x_new)
        add_alpha_from_tree(b, lid_q_b)
        Sb = Sb_list[b]
        if Sb is not None and Sb.size > 0:
            for bp in Sb:
                tree_bp = trees[int(bp)]["tree"]
                lid_q_bp = locate_leaf_id(tree_bp, x_new)
                add_alpha_from_tree(int(bp), lid_q_bp)

    s = float(np.sum(betas))
    if s > 0:
        betas /= s
    return betas
