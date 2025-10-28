"""
Tree data structures and traversal helpers.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class TreeNode:
    """Minimal tree node storing split metadata and member indices in G_b."""

    __slots__ = ("is_leaf", "split_feature", "split_threshold", "left", "right", "indices", "depth")

    def __init__(self, indices: np.ndarray, depth: int):
        self.is_leaf: bool = True
        self.split_feature: Optional[int] = None
        self.split_threshold: Optional[float] = None
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None
        self.indices: np.ndarray = indices
        self.depth: int = depth


class Tree:
    """Binary tree with a single root node used for structure learning."""

    __slots__ = ("root",)

    def __init__(self, root: TreeNode):
        self.root = root


def locate_leaf_id(tree: Tree, x_row: np.ndarray) -> int:
    """Return the integer leaf identifier reached by a feature vector."""
    node = tree.root
    path = 1
    while not node.is_leaf:
        j = node.split_feature
        thr = node.split_threshold
        go_left = x_row[j] <= thr
        node = node.left if go_left else node.right
        path = 2 * path + (0 if go_left else 1)
    return path


def compute_leaf_ids(tree: Tree, X_mat: np.ndarray) -> np.ndarray:
    """Vectorized helper returning leaf ids for every row of X_mat."""
    return np.array([locate_leaf_id(tree, X_mat[i, :]) for i in range(X_mat.shape[0])], dtype=np.int64)
