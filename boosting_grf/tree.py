"""
Tree data structures and traversal helpers.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class TreeNode:
    """Tree node storing split metadata and the member indices for Algorithm 1.

    Args:
        indices: Integer indices local to the current `G_b` subsample.
        depth: Depth of the node in the tree (root depth == 0).
    """

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
    """Binary tree container for structure learning."""

    __slots__ = ("root",)

    def __init__(self, root: TreeNode):
        self.root = root


def locate_leaf_id(tree: Tree, x_row: np.ndarray) -> int:
    """Traverse a tree and obtain a deterministic integer leaf identifier.

    The identifier encodes the traversal path (`1 -> 2 -> ...`) so that it is
    stable across repeated calls and amenable to hashing.

    Args:
        tree: The tree whose partitions are being queried.
        x_row: Feature row (`shape == (p,)`) whose path is evaluated.

    Returns:
        An integer representing the path from the root to the reached leaf.
    """
    node = tree.root
    path = 1
    while not node.is_leaf:
        # A binary heap style encoding for consistent leaf IDs.
        j = node.split_feature
        thr = node.split_threshold
        go_left = x_row[j] <= thr
        node = node.left if go_left else node.right
        path = 2 * path + (0 if go_left else 1)
    return path


def compute_leaf_ids(tree: Tree, X_mat: np.ndarray) -> np.ndarray:
    """Vectorized helper returning leaf IDs for every row of a matrix.

    Args:
        tree: The tree used to evaluate the partition.
        X_mat: Feature matrix `(n, p)` subject to traversal.

    Returns:
        A NumPy array of shape `(n,)` containing integer leaf identifiers.
    """
    return np.array([locate_leaf_id(tree, X_mat[i, :]) for i in range(X_mat.shape[0])], dtype=np.int64)
