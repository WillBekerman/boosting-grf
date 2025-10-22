"""
Algorithm 1 -- Boosting algorithm only for tree structure

Public API:
  model = fit_alg1_grf(X1, O1, X2, O2, B, q, xi1, max_depth, min_leaf, seed)
  pred  = model.predict_theta(Xnew)  # {"theta": ..., "nu": ...}
"""
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg import solve as nla_solve, pinv as nla_pinv


# ------------------------------ RNG helper ------------------------------
class RNG:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    def bernoulli_mask(self, n: int, p: float) -> np.ndarray:
        return (self.rng.random(n) < p)
    def choice(self, n: int, size: int, replace: bool) -> np.ndarray:
        return self.rng.choice(np.arange(n), size=size, replace=replace)


# ------------------------------ Tree node ------------------------------
class TreeNode:
    __slots__ = ("is_leaf","split_feature","split_threshold","left","right","indices","depth")
    def __init__(self, indices: np.ndarray, depth: int):
        self.is_leaf: bool = True
        self.split_feature: Optional[int] = None
        self.split_threshold: Optional[float] = None
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None
        # indices within local G_b
        self.indices: np.ndarray = indices
        self.depth: int = depth

class Tree:
    __slots__ = ("root",)
    def __init__(self, root: TreeNode):
        self.root = root


# ---------------------------- Leaf ID helpers ---------------------------
def locate_leaf_id(tree: Tree, x_row: np.ndarray) -> int:
    node = tree.root
    path = 1
    while not node.is_leaf:
        j = node.split_feature
        thr = node.split_threshold
        go_left = (x_row[j] <= thr)
        node = node.left if go_left else node.right
        path = 2 * path + (0 if go_left else 1)
    return path

def compute_leaf_ids(tree: Tree, X_mat: np.ndarray) -> np.ndarray:
    return np.array([locate_leaf_id(tree, X_mat[i, :]) for i in range(X_mat.shape[0])], dtype=np.int64)


# ------------------------- Weighted least squares ------------------------
def wls_theta_nu_cate(A: np.ndarray, Y: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
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


# ------------------------ alpha and rho construction --------------------------
def build_prev_alpha_weights_for_query(
    XGb: np.ndarray,
    x_query: np.ndarray,
    prev_trees: List[Dict[str, Any]],  # each: {"tree","Gb_idx","leaf_ids_on_Gt"}
) -> np.ndarray:
    """
    Accumulate alpha^{(b')}(x_query) for rows in current G_b, with denominators over each prior tree's own G_t.
    """
    m = XGb.shape[0]
    acc = np.zeros(m, dtype=float)
    for t in prev_trees:
        tree = t["tree"]
        lid_on_Gb = compute_leaf_ids(tree, XGb)  # leaves of current G_b under prior tree
        lid_q = locate_leaf_id(tree, x_query)
        lid_on_Gt = t["leaf_ids_on_Gt"]         # leaves of that tree's own G_t
        denom = int(np.sum(lid_on_Gt == lid_q))
        if denom == 0:
            continue
        mask = (lid_on_Gb == lid_q)
        acc[mask] += 1.0 / denom
    return acc

def build_curr_alpha_weights_for_query(
    XGb: np.ndarray, partial_tree: Tree, x_query: np.ndarray
) -> np.ndarray:
    """
    alpha^{(b)} for current partial tree over G_b (uniform within leaf).
    """
    lids = compute_leaf_ids(partial_tree, XGb)
    lid_q = locate_leaf_id(partial_tree, x_query)
    mask = (lids == lid_q)
    denom = int(np.sum(mask))
    if denom == 0:
        return np.zeros_like(mask, dtype=float)
    return mask.astype(float) / float(denom)

def compute_rho_vector(
    XGb: np.ndarray,
    O_Gb: List[Dict[str, Any]],    # {'Y','A'}
    Sb_prev_trees: List[Dict[str, Any]],
    partial_tree: Tree,
    theta_nu_per_i: List[Tuple[float, float]],  # (thetahat, nuhat) fixed for this round
    xi_selector: np.ndarray,  # [1,0]
    ridge_eps: float = 1e-8,
) -> np.ndarray:
    """
    rho_i = -psi^T [sum_j w^(rho)_j(X_i) nabla_psi(O_j; X_i)]^{-1} psi_{thetahat, nuhat}(O_i; X_i), with
    w^(rho) = w_prev + alpha_current.
    Uses thetahat, nuhat from the estimating step (fit with w_prev only) and keeps them fixed.
    """
    m = XGb.shape[0]
    rhos = np.zeros(m, dtype=float)
    for kk in range(m):
        x_i = XGb[kk, :]
        # Build weights for rho: previous + current
        w_prev = build_prev_alpha_weights_for_query(XGb, x_i, Sb_prev_trees)
        w_curr = build_curr_alpha_weights_for_query(XGb, partial_tree, x_i)
        w_all = w_prev + w_curr

        # Jacobian sum
        M = np.zeros((2, 2), dtype=float)
        for jj in range(m):
            A = float(O_Gb[jj]["A"])
            J = np.array([[-A*A, -A], [-A, -1.0]], dtype=float)
            M += w_all[jj] * J

        Ai = float(O_Gb[kk]["A"]); Yi = float(O_Gb[kk]["Y"])
        th, nu = theta_nu_per_i[kk]
        resid = Yi - Ai*th - nu
        v = np.array([resid * Ai, resid], dtype=float)

        if ridge_eps > 0:
            M[0,0] += ridge_eps; M[1,1] += ridge_eps
        try:
            u = nla_solve(M, v)
        except LinAlgError:
            u = nla_pinv(M) @ v
        rhos[kk] = - float(xi_selector @ u)
    return rhos


# --------------------------- Best split search ---------------------------
def best_split_for_node(node_rows: np.ndarray, XGb: np.ndarray, rhos: np.ndarray, min_leaf: int) -> Optional[Dict[str, Any]]:
    if node_rows.shape[0] < 2 * min_leaf:
        return None
    best = None; best_obj = -np.inf
    p = XGb.shape[1]
    for j in range(p):
        xj = XGb[node_rows, j]
        ord_idx = np.argsort(xj)
        rho_sorted = rhos[node_rows][ord_idx]
        x_sorted = xj[ord_idx]
        if np.unique(x_sorted).shape[0] < 2:
            continue
        csum = np.cumsum(rho_sorted); total = csum[-1]
        n = rho_sorted.shape[0]
        for k in range(n - 1):
            if x_sorted[k] == x_sorted[k + 1]:
                continue
            nL = k + 1; nR = n - nL
            if nL < min_leaf or nR < min_leaf:
                continue
            sumL = csum[k]; sumR = total - csum[k]
            obj = (sumL*sumL)/nL + (sumR*sumR)/nR
            if obj > best_obj:
                thr = 0.5 * (x_sorted[k] + x_sorted[k + 1])
                left_idx = node_rows[ord_idx[:nL]]
                right_idx = node_rows[ord_idx[nL:]]
                best_obj = obj
                best = {"feature": j, "threshold": float(thr), "left": left_idx, "right": right_idx}
    return best


# ---------------------------- Stage 1 training ---------------------------
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
        cand = np.arange(b)   # empty when b=0
        if cand.size == 0:
            Sb_idx = np.array([], dtype=int)
            Sb_prev_trees: List[Dict[str, Any]] = []
        else:
            mask = rng.bernoulli_mask(cand.shape[0], q)
            Sb_idx = cand[mask]
            Sb_prev_trees = []
            for bp in Sb_idx:
                t = trees[int(bp)]
                Sb_prev_trees.append({
                    "tree": t["tree"],
                    "Gb_idx": t["Gb_idx"],
                    "leaf_ids_on_Gt": t["leaf_ids_on_Gb"],  # denominators on each prior tree's G_t
                })
        Sb_list[b] = Sb_idx

        # -------- Estimating step (E): fit thetahat, nuhat using ONLY previous-tree weights --------
        theta_nu_per_i: List[Tuple[float, float]] = [None] * m
        A_all = np.array([og["A"] for og in OGb], dtype=float)
        Y_all = np.array([og["Y"] for og in OGb], dtype=float)

        if Sb_idx.shape[0] > 0:
            for kk in range(m):
                x_i = XGb[kk, :]
                w_prev = build_prev_alpha_weights_for_query(XGb, x_i, Sb_prev_trees)  # ONLY previous trees
                if not np.any(w_prev > 0):
                    w_prev = np.ones(m, dtype=float)
                th, nu = wls_theta_nu_cate(A_all, Y_all, w_prev)
                theta_nu_per_i[kk] = (th, nu)
        else:
            # No previous trees: shared uniform-weight fit (root node equivalent)
            th, nu = wls_theta_nu_cate(A_all, Y_all, np.ones(m, dtype=float))
            for kk in range(m):
                theta_nu_per_i[kk] = (th, nu)

        # Start with partial
        partial = Tree(TreeNode(indices=np.arange(m), depth=0))

        depth = 0
        while True:
            # -------- Split step (rho): use w_prev + alpha_current for rho --------
            rhos = compute_rho_vector(
                XGb, OGb, Sb_prev_trees, partial, theta_nu_per_i,
                xi_selector=np.array([1.0, 0.0], dtype=float), ridge_eps=1e-8
            )

            # Frontier at this depth
            frontier: List[TreeNode] = []
            def collect(n: TreeNode):
                if n.depth == depth:
                    frontier.append(n)
                if not n.is_leaf:
                    collect(n.left); collect(n.right)
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
        trees[b] = {"tree": partial, "Gb_idx": Gb_idx, "leaf_ids_on_Gb": leaf_ids_on_Gb}

    return {"trees": trees, "Sb_list": Sb_list}


# ---------------------------- Stage 2 (beta cumulative-alpha) -----------------------------
def fit_stage2_grf(
    X2: np.ndarray,
    trees: List[Dict[str, Any]],
    Sb_list: List[np.ndarray],
    M: Optional[int] = None,
    xi2: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    n2 = X2.shape[0]
    BT = len(trees)  # equals B
    if BT == 0:
        raise RuntimeError("No trees from Stage 1.")
    leaf_ids_per_tree = [compute_leaf_ids(trees[b]["tree"], X2) for b in range(BT)]
    return {"leaf_ids_per_tree": leaf_ids_per_tree, "Sb_list": Sb_list}


def compute_beta_weights(model: Dict[str, Any], x_new: np.ndarray) -> np.ndarray:
    """
    beta_i(x) = sum_{b=0}^{B-1} [ alpha^{(b)}_i(x) + sum_{b'\in S_b} alpha^{(b')}_i(x) ], i \in D2.
    alpha^{(t)}_i(x) is uniform within the tree-t leaf containing x on D2.
    Normalize beta to sum to 1 over D2.
    """
    X2 = model["X2"]
    stage1 = model["stage1"]
    trees = stage1["trees"]
    Sb_list = stage1["Sb_list"]
    leaf_ids_per_tree = model["stage2"]["leaf_ids_per_tree"]

    n2 = X2.shape[0]
    BT = len(trees)  # equals B
    betas = np.zeros(n2, dtype=float)

    def add_alpha_from_tree(t_index: int, lid_q: int):
        lids = leaf_ids_per_tree[t_index]
        mask = (lids == lid_q)
        denom = int(np.sum(mask))
        if denom > 0:
            betas[mask] += 1.0 / denom

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


# ------------------------------ Public API ------------------------------
class Alg1GRFModel:
    def __init__(self, model: Dict[str, Any]):
        self.__dict__.update(model)

    def predict_theta(self, Xnew: np.ndarray) -> Dict[str, np.ndarray]:
        Xnew = np.asarray(Xnew, dtype=float)
        n = Xnew.shape[0]
        A2 = np.array([oi["A"] for oi in self.O2], dtype=float)
        Y2 = np.array([oi["Y"] for oi in self.O2], dtype=float)
        theta = np.full(n, np.nan, dtype=float)
        nu = np.full(n, np.nan, dtype=float)
        for i in range(n):
            beta = compute_beta_weights(self.__dict__, Xnew[i, :])
            idx = np.where(beta > 0)[0]
            if idx.size == 0:
                continue
            w = beta[idx]
            th, nv = wls_theta_nu_cate(A2[idx], Y2[idx], w)
            theta[i] = th; nu[i] = nv
        return {"theta": theta, "nu": nu}


def fit_alg1_grf(
    X1: np.ndarray,
    O1: List[Dict[str, Any]],
    X2: np.ndarray,
    O2: List[Dict[str, Any]],
    B: int,
    M: Optional[int] = None,
    q: float = 0.6,
    xi1: float = 0.6,
    xi2: Optional[float] = None,
    max_depth: int = 5,
    min_leaf: int = 10,
    seed: Optional[int] = None,
) -> Alg1GRFModel:
    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)
    stage1 = fit_stage1_structures(
        X1, O1, B=B, q=q, xi1=xi1,
        max_depth=max_depth, min_leaf=min_leaf, seed=seed
    )
    stage2 = fit_stage2_grf(
        X2, stage1["trees"],
        Sb_list=stage1["Sb_list"], M=M, xi2=xi2, seed=seed
    )
    return Alg1GRFModel({
        "X2": X2,
        "O2": O2,
        "stage1": stage1,
        "stage2": stage2,
    })


# ----------------------------- Test ------------------------------
if __name__ == "__main__":
    seed = 0
    rng = np.random.default_rng(seed)
    n = 2000; p = 3
    X = rng.normal(size=(n, p))

    def tau_true(x):
        return 1.0 + x[0] - 0.5 * x[1]

    def f_base(x):
        return np.sin(x[0]) + x[1]**2 - x[2]

    pis = 1.0 / (1.0 + np.exp(-(0.2 + 0.5*X[:,0] - 0.3*X[:,2])))
    A = rng.binomial(1, pis)
    Y = np.apply_along_axis(tau_true, 1, X) * A + np.apply_along_axis(f_base, 1, X) + rng.normal(size=n)

    idx = rng.permutation(n)
    train = idx[: int(0.8*n)]
    test = idx[int(0.8*n):]

    D1 = train[: len(train)//2]
    D2 = np.setdiff1d(train, D1, assume_unique=False)

    X1, X2 = X[D1,:], X[D2,:]
    O_list = [{"Y": float(Y[i]), "A": float(A[i])} for i in range(n)]
    O1 = [O_list[int(i)] for i in D1]; O2 = [O_list[int(i)] for i in D2]

    B = 16
    model = fit_alg1_grf(X1, O1, X2, O2, B=B, q=0.6, xi1=0.6, max_depth=3, min_leaf=20, seed=seed)
    pred = model.predict_theta(X[test,:])
    tau_hat = pred["theta"]
    tau_true_vec = np.apply_along_axis(lambda r: 1.0 + r[0] - 0.5*r[1], 1, X[test,:])
    rmse = float(np.sqrt(np.mean((tau_hat - tau_true_vec)**2)))
    print("[Alg1:] Test RMSE for tau(x):", f"{rmse:.4f}")
