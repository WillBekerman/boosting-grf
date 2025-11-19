// -----------------------------------------------------------------------------
//  _grf_bindings.cpp
//
//  This translation unit exposes the GRF C++ core to Python and reimplements
//  Algorithm 1 from *causalgrf.pdf* (Athey et al.) in a way that matches the
//  paper verbatim:
//    • Stage 1 learns tree structures via gradient-boosted score splits, using
//      the dropout-set recursion to reuse prior trees.
//    • Stage 2 computes β(x) exactly as in Eq. (10), carrying the Stage‑1
//      denominators and the λ_b schedule through to prediction.
//
//  The comments below walk through each stage so the connection to the paper
//  remains explicit.
// -----------------------------------------------------------------------------
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <limits>
#include <functional>
#include <cstring>

#include <Eigen/Dense>

#include "commons/Data.h"
#include "commons/globals.h"
#include "forest/Forest.h"
#include "forest/ForestOptions.h"
#include "forest/ForestPredictor.h"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainer.h"
#include "forest/ForestTrainers.h"
#include "prediction/Prediction.h"
#include "splitting/RegressionSplittingRule.h"
#include "tree/Tree.h"

namespace py = pybind11;

namespace boosting_grf {

using grf::Forest;
using grf::ForestOptions;
using grf::ForestPredictor;
using grf::Prediction;

namespace detail {

inline py::value_error shape_error(const std::string& msg);
std::vector<double> to_column_major(const py::array_t<double, py::array::c_style | py::array::forcecast>& X_in);

struct ColumnMajorMatrix {
  std::shared_ptr<std::vector<double>> values;
  size_t n_rows = 0;
  size_t n_cols = 0;

  inline double get(size_t row, size_t col) const {
    return (*values)[col * n_rows + row];
  }

  inline double* data() {
    return values->data();
  }

  inline const double* data() const {
    return values->data();
  }
};

/**
 * @brief Wrap a numpy matrix in column-major storage for GRF.
 *
 * @param arr Input 2-D array (any strides) to be copied.
 * @return ColumnMajorMatrix owning a shared column-major buffer.
 *
 * @behavior Copies the matrix so future subsamples can share the same storage
 *           while remaining contiguous for grf::Data.
 */
inline ColumnMajorMatrix column_major_matrix(const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
  auto buf = arr.request();
  if (buf.ndim != 2) {
    throw shape_error("Input array must be 2D");
  }
  ColumnMajorMatrix mat;
  mat.n_rows = static_cast<size_t>(buf.shape[0]);
  mat.n_cols = static_cast<size_t>(buf.shape[1]);
  mat.values = std::make_shared<std::vector<double>>(to_column_major(arr));
  return mat;
}

/**
 * @brief Build a column-major view for a subsampled set of rows.
 *
 * @param src Source column-major matrix.
 * @param indices Row indices to keep (ascending order assumed for cache hits).
 * @return ColumnMajorMatrix referencing a new shared buffer for the subset.
 *
 * @behavior Copies only the requested rows so Stage 1 can reuse X without
 *           touching the original Python-owned memory.
 */
inline ColumnMajorMatrix subset_matrix(const ColumnMajorMatrix& src,
                                       const std::vector<int>& indices) {
  ColumnMajorMatrix sub;
  sub.n_rows = indices.size();
  sub.n_cols = src.n_cols;
  sub.values = std::make_shared<std::vector<double>>(sub.n_rows * sub.n_cols);
  for (size_t col = 0; col < sub.n_cols; ++col) {
    for (size_t row = 0; row < sub.n_rows; ++row) {
      int src_row = indices[row];
      (*sub.values)[col * sub.n_rows + row] = (*src.values)[col * src.n_rows + src_row];
    }
  }
  return sub;
}

/**
 * @brief Extract a subset of a numeric vector.
 *
 * @param src Source vector.
 * @param indices Positions to keep (interpreted as ints).
 * @return std::vector<double> Copy of src at the requested positions.
 *
 * @behavior Maintains the ordering of the subsample so indices line up with
 *           Stage 1 caches.
 */
inline std::vector<double> subset_vector(const std::vector<double>& src,
                                         const std::vector<int>& indices) {
  std::vector<double> out(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    out[i] = src[indices[i]];
  }
  return out;
}

struct SplitResult {
  bool found = false;
  int feature = -1;
  double threshold = 0.0;
  bool send_missing_left = true;
  std::vector<int> left_indices;
  std::vector<int> right_indices;
};

/**
 * @brief Find the best CART split for the current node.
 *
 * @param X_ptr Column-major feature matrix (shared with grf::Data).
 * @param n_rows Number of rows in the matrix.
 * @param n_cols Number of columns in the matrix.
 * @param rhos Gradient residuals (ρ_i) for all rows.
 * @param node_indices Indices of samples residing in the node.
 * @param min_leaf Minimum number of samples per child.
 * @return SplitResult containing the best split or found=false if unsplittable.
 *
 * @behavior Mirrors Eq. (8) by delegating to GRF's regression splitter, so we
 *           maintain identical split criteria between Algorithm 1 and GRF.
 */
inline SplitResult regression_best_split(const double* X_ptr,
                                         size_t n_rows,
                                         size_t n_cols,
                                         const std::vector<double>& rhos,
                                         const std::vector<int>& node_indices,
                                         int min_leaf) {
  SplitResult result;
  size_t node_size = node_indices.size();
  if (node_size < static_cast<size_t>(2 * std::max(1, min_leaf))) {
    return result;
  }
  grf::Data data(X_ptr, n_rows, n_cols);
  Eigen::ArrayXXd responses(n_rows, 1);
  for (size_t i = 0; i < n_rows; ++i) {
    responses(i, 0) = rhos[i];
  }

  std::vector<std::vector<size_t>> samples(1);
  samples[0].reserve(node_size);
  for (int idx : node_indices) {
    if (idx < 0 || static_cast<size_t>(idx) >= n_rows) {
      throw shape_error("node_indices contain out-of-range entries");
    }
    samples[0].push_back(static_cast<size_t>(idx));
  }

  double alpha = std::min(
      0.5,
      std::max(1e-9, static_cast<double>(min_leaf) / static_cast<double>(node_size)));
  grf::RegressionSplittingRule rule(node_size, alpha, 0.0);

  std::vector<size_t> split_vars(1, 0);
  std::vector<double> split_values(1, 0.0);
  std::vector<bool> send_missing_left(1, true);
  std::vector<size_t> possible_vars(n_cols);
  std::iota(possible_vars.begin(), possible_vars.end(), 0);

  bool stop = rule.find_best_split(
      data,
      0,
      possible_vars,
      responses,
      samples,
      split_vars,
      split_values,
      send_missing_left);
  if (stop) {
    return result;
  }

  result.found = true;
  result.feature = static_cast<int>(split_vars[0]);
  result.threshold = split_values[0];
  result.send_missing_left = send_missing_left[0];

  result.left_indices.reserve(node_size / 2 + 1);
  result.right_indices.reserve(node_size / 2 + 1);
  for (size_t sample : samples[0]) {
    double value = data.get(sample, split_vars[0]);
    bool go_left = (value <= split_values[0]) ||
        (send_missing_left[0] && std::isnan(value)) ||
        (std::isnan(split_values[0]) && std::isnan(value));
    if (go_left) {
      result.left_indices.push_back(static_cast<int>(sample));
    } else {
      result.right_indices.push_back(static_cast<int>(sample));
    }
  }

  return result;
}

struct ThetaNu {
  double theta = 0.0;
  double nu = 0.0;
};

/**
 * @brief Solve a 2x2 weighted least squares system.
 *
 * @param XtWX Symmetric design matrix.
 * @param XtWy Right-hand side vector.
 * @return ThetaNu Pair (θ, ν) solving XtWX * beta = XtWy.
 *
 * @behavior Uses a numerically stable complete orthogonal decomposition so the
 *           linear system stays well behaved even when nearly singular.
 */
inline ThetaNu solve_from_matrix(const Eigen::Matrix2d& XtWX, const Eigen::Vector2d& XtWy) {
  Eigen::Vector2d beta = Eigen::Vector2d::Zero();
  Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix2d> cod(XtWX);
  beta = cod.solve(XtWy);
  return {beta[0], beta[1]};
}

/**
 * @brief Solve for (θ, ν) given raw moment statistics.
 *
 * @param sum_A2 Sum of A_i^2.
 * @param sum_A Sum of A_i.
 * @param sum_w Sum of weights (count or normalized mass).
 * @param sum_AY Sum of A_i * Y_i.
 * @param sum_Y Sum of Y_i.
 * @return ThetaNu Weighted least-squares solution.
 *
 * @behavior Constructs the 2x2 system directly from the sufficient statistics
 *           and delegates to solve_from_matrix.
 */
inline ThetaNu solve_from_stats(double sum_A2,
                                double sum_A,
                                double sum_w,
                                double sum_AY,
                                double sum_Y) {
  Eigen::Matrix2d XtWX;
  XtWX << sum_A2, sum_A,
          sum_A, sum_w;
  Eigen::Vector2d XtWy;
  XtWy << sum_AY, sum_Y;
  return solve_from_matrix(XtWX, XtWy);
}

/**
 * @brief Compute a fallback (θ, ν) using uniform weights.
 *
 * @param A Treatment indicator vector.
 * @param Y Outcome vector.
 * @return ThetaNu Uniform-weight weighted least-squares solution.
 *
 * @behavior Used when no prior trees exist so Stage 1 always has an initial
 *           plug-in estimate.
 */
inline ThetaNu uniform_wls(const std::vector<double>& A,
                           const std::vector<double>& Y) {
  double sum_A = 0.0;
  double sum_A2 = 0.0;
  double sum_Y = 0.0;
  double sum_AY = 0.0;
  double sum_w = static_cast<double>(A.size());
  for (size_t i = 0; i < A.size(); ++i) {
    sum_A += A[i];
    sum_A2 += A[i] * A[i];
    sum_Y += Y[i];
    sum_AY += A[i] * Y[i];
  }
  return solve_from_stats(sum_A2, sum_A, sum_w, sum_AY, sum_Y);
}

/**
 * @brief Copy a 1-D numpy array into std::vector<double>.
 *
 * @param arr Input array expected to be one-dimensional.
 * @return std::vector<double> Copy of the data.
 *
 * @behavior Validates dimensionality so callers in Stage 1/2 always work with
 *           contiguous C++ vectors.
 */
inline std::vector<double> array1d_to_vector(const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
  auto buf = arr.request();
  if (buf.ndim != 1) {
    throw shape_error("Expected a 1D array.");
  }
  size_t n = static_cast<size_t>(buf.shape[0]);
  std::vector<double> out(n);
  const double* ptr = static_cast<const double*>(buf.ptr);
  std::copy(ptr, ptr + n, out.begin());
  return out;
}

struct TreeNode {
  bool is_leaf = true;
  int split_feature = -1;
  double split_threshold = 0.0;
  bool send_missing_left = true;
  int left = -1;
  int right = -1;
  int depth = 0;
  std::vector<int> indices;
};

struct BoostedTree {
  std::vector<TreeNode> nodes;
  std::unordered_map<int, int> leaf_counts;
};

/**
 * @brief Evaluate a tree's decision rule for a scalar value.
 *
 * @param node Split descriptor.
 * @param value Feature value (may be NaN).
 * @return True if the traversal should go left, false otherwise.
 *
 * @behavior Follows the GRF missing-value convention: NaNs are routed based on
 *           send_missing_left unless the split threshold itself is NaN.
 */
inline bool go_left(const TreeNode& node, double value) {
  if (std::isnan(value)) {
    if (std::isnan(node.split_threshold)) {
      return true;
    }
    return node.send_missing_left;
  }
  if (std::isnan(node.split_threshold)) {
    return node.send_missing_left;
  }
  return value <= node.split_threshold;
}

/**
 * @brief Traverse a BoostedTree to get a leaf identifier.
 *
 * @param tree Tree to traverse.
 * @param X Column-major feature matrix.
 * @param row Row index within X.
 * @return Integer leaf id (encoded via heap index).
 *
 * @behavior Uses a binary-heap encoding so leaf IDs are deterministic and can
 *           be looked up in stage-2 caches.
 */
inline int locate_leaf(const BoostedTree& tree,
                       const ColumnMajorMatrix& X,
                       size_t row) {
  int node_idx = 0;
  int path = 1;
  while (true) {
    const TreeNode& node = tree.nodes[node_idx];
    if (node.is_leaf || node.left < 0 || node.right < 0) {
      break;
    }
    double value = X.get(row, static_cast<size_t>(node.split_feature));
    bool left_child = go_left(node, value);
    node_idx = left_child ? node.left : node.right;
    path = 2 * path + (left_child ? 0 : 1);
  }
  return path;
}

struct LeafStats {
  double weight = 0.0;
  int count = 0;
  double sum_A = 0.0;
  double sum_Y = 0.0;
  double sum_A2 = 0.0;
  double sum_AY = 0.0;
  double M00 = 0.0;
  double M01 = 0.0;
  double M10 = 0.0;
  double M11 = 0.0;
  std::vector<int> members;
};

// Aggregate sufficient statistics for a single leaf. These feed both the
// plug-in (θ, ν) solves and the Jacobian term in Eq. (9).
inline bool summarize_leaf_contrib(const std::vector<int>& members,
                                   double weight,
                                   const std::vector<double>& A,
                                   const std::vector<double>& Y,
                                   LeafStats& out) {
  int count = static_cast<int>(members.size());
  if (count == 0 || weight <= 0.0) {
    return false;
  }
  double sum_A = 0.0;
  double sum_Y = 0.0;
  double sum_A2 = 0.0;
  double sum_AY = 0.0;
  for (int idx : members) {
    double a = A[idx];
    double y = Y[idx];
    sum_A += a;
    sum_Y += y;
    sum_A2 += a * a;
    sum_AY += a * y;
  }
  out.weight = weight;
  out.count = count;
  out.sum_A = sum_A;
  out.sum_Y = sum_Y;
  out.sum_A2 = sum_A2;
  out.sum_AY = sum_AY;
  out.M00 = -weight * sum_A2;
  out.M01 = -weight * sum_A;
  out.M10 = -weight * sum_A;
  out.M11 = -weight * static_cast<double>(count);
  return true;
}

struct PrevTreeCache {
  const BoostedTree* tree = nullptr;
  std::vector<int> leaf_ids_on_cur;
  std::unordered_map<int, LeafStats> leaf_stats;
};

struct CurrentTreeCache {
  std::vector<int> leaf_ids;
  std::unordered_map<int, LeafStats> stats;
};

struct PrevRoundCache {
  std::vector<PrevTreeCache> structures;
  std::vector<int> signature_indices;
  std::vector<Eigen::Matrix2d> XtWX;
  std::vector<Eigen::Vector2d> XtWy;
  std::vector<Eigen::Matrix2d> signature_M;
  std::vector<bool> contrib_mask;
};

inline std::vector<int> compute_leaf_ids(const BoostedTree& tree,
                                         const ColumnMajorMatrix& X) {
  std::vector<int> leaf_ids(X.n_rows);
  for (size_t i = 0; i < X.n_rows; ++i) {
    leaf_ids[i] = locate_leaf(tree, X, i);
  }
  return leaf_ids;
}

inline PrevTreeCache build_prev_tree_cache(const BoostedTree& tree,
                                           const ColumnMajorMatrix& XGb,
                                           const std::vector<double>& A,
                                           const std::vector<double>& Y) {
  /**
   * @brief Assemble cached statistics for a previous tree on the current Gb.
   *
   * @param tree Previously trained tree.
   * @param XGb Current subsample features.
   * @param A Treatments for Gb.
   * @param Y Outcomes for Gb.
   * @return PrevTreeCache storing leaf IDs and sufficient stats.
   *
   * @behavior Replays the tree on Gb and stores α-weights (1 / |Gb ∩ leaf|)
   *           with aggregated score terms used in Eq. (9).
   */
  PrevTreeCache cache;
  cache.tree = &tree;
  cache.leaf_ids_on_cur = compute_leaf_ids(tree, XGb);  // Locate every Gb point inside this older tree.
  std::unordered_map<int, std::vector<int>> members;
  for (size_t i = 0; i < cache.leaf_ids_on_cur.size(); ++i) {
    members[cache.leaf_ids_on_cur[i]].push_back(static_cast<int>(i));
  }
  for (const auto& kv : members) {
    int leaf_id = kv.first;
    auto denom_it = tree.leaf_counts.find(leaf_id);
    if (denom_it == tree.leaf_counts.end() || denom_it->second <= 0) {
      continue;
    }
    LeafStats stats;
    if (summarize_leaf_contrib(kv.second,
                               1.0 / static_cast<double>(denom_it->second),
                               A,
                               Y,
                               stats)) {
      cache.leaf_stats.emplace(leaf_id, std::move(stats));
    }
  }
  return cache;
}

inline CurrentTreeCache build_current_tree_cache(const BoostedTree& tree,
                                                 const ColumnMajorMatrix& XGb,
                                                 const std::vector<double>& A,
                                                 const std::vector<double>& Y) {
  /**
   * @brief Cache per-leaf stats for the partially grown tree.
   *
   * @param tree Tree being grown this round.
   * @param XGb Current subsample.
   * @param A Treatments for Gb.
   * @param Y Outcomes for Gb.
   * @return CurrentTreeCache with leaf membership and stats.
   *
   * @behavior Needed so compute_rho_vector can reuse aggregated Jacobians
   *           without re-traversing the tree for every split candidate.
   */
  CurrentTreeCache cache;
  cache.leaf_ids = compute_leaf_ids(tree, XGb);  // Evaluate the partial tree on its own subsample.
  std::unordered_map<int, std::vector<int>> members;
  for (size_t i = 0; i < cache.leaf_ids.size(); ++i) {
    members[cache.leaf_ids[i]].push_back(static_cast<int>(i));
  }
  for (auto& kv : members) {
    LeafStats stats;
    stats.members = kv.second;
    if (summarize_leaf_contrib(stats.members,
                               1.0 / static_cast<double>(stats.members.size()),
                               A,
                               Y,
                               stats)) {
      cache.stats.emplace(kv.first, std::move(stats));
    }
  }
  return cache;
}

struct VectorHash {
  size_t operator()(const std::vector<int>& vec) const {
    size_t seed = vec.size();
    for (int v : vec) {
      seed ^= std::hash<int>()(v) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

inline bool signature_has_contrib(const std::vector<int>& signature,
                                  const std::vector<PrevTreeCache>& prev_structs) {
  for (size_t t = 0; t < prev_structs.size(); ++t) {
    int leaf_id = signature[t];
    const auto& stats_map = prev_structs[t].leaf_stats;
    auto it = stats_map.find(leaf_id);
    if (it != stats_map.end() && it->second.count > 0) {
      return true;
    }
  }
  return false;
}

inline std::unique_ptr<PrevRoundCache> prepare_prev_round_cache(
    const std::vector<PrevTreeCache>& prev_structs) {
  /**
   * @brief Deduplicate dropout signatures and precompute their moments.
   *
   * @param prev_structs Caches for trees in S_b.
   * @return Unique signature cache or nullptr if S_b is empty.
   *
   * @behavior Groups Gb rows that share identical (leaf_id^b') tuples so we
   *           solve each plug-in system once per signature.
   */
  if (prev_structs.empty()) {
    return nullptr;
  }
  size_t m = prev_structs.front().leaf_ids_on_cur.size();  // Number of Gb rows.
  size_t k = prev_structs.size();
  auto cache = std::make_unique<PrevRoundCache>();
  cache->structures = prev_structs;
  cache->signature_indices.assign(m, -1);

  std::unordered_map<std::vector<int>, int, VectorHash> signature_map;
  std::vector<std::vector<int>> signatures;

  for (size_t i = 0; i < m; ++i) {
    std::vector<int> signature(k);
    for (size_t t = 0; t < k; ++t) {
      signature[t] = prev_structs[t].leaf_ids_on_cur[i];
    }
    auto iter = signature_map.find(signature);
    if (iter == signature_map.end()) {
      int idx = static_cast<int>(signatures.size());
      signatures.push_back(signature);
      signature_map.emplace(signature, idx);
      cache->signature_indices[i] = idx;
    } else {
      cache->signature_indices[i] = iter->second;
    }
  }

  cache->XtWX.resize(signatures.size());
  cache->XtWy.resize(signatures.size());
  cache->signature_M.resize(signatures.size());
  cache->contrib_mask.resize(signatures.size(), false);

  for (size_t idx = 0; idx < signatures.size(); ++idx) {
    const auto& signature = signatures[idx];
    double w_sum = 0.0;
    double wA_sum = 0.0;
    double wA2_sum = 0.0;
    double wY_sum = 0.0;
    double wAY_sum = 0.0;
    Eigen::Matrix2d jac = Eigen::Matrix2d::Zero();
    for (size_t t = 0; t < k; ++t) {
      int leaf_id = signature[t];
      const auto& stats_map = prev_structs[t].leaf_stats;
      auto it = stats_map.find(leaf_id);
      if (it == stats_map.end()) {
        continue;
      }
      const LeafStats& stats = it->second;
      double weight = stats.weight;
      double count = static_cast<double>(stats.count);
      w_sum += weight * count;
      wA_sum += weight * stats.sum_A;
      wA2_sum += weight * stats.sum_A2;
      wY_sum += weight * stats.sum_Y;
      wAY_sum += weight * stats.sum_AY;
      Eigen::Matrix2d contrib;
      contrib << stats.M00, stats.M01,
                 stats.M10, stats.M11;
      jac += contrib;
    }
    cache->contrib_mask[idx] = (w_sum > 0.0);
    Eigen::Matrix2d XtWX;
    XtWX << wA2_sum, wA_sum,
            wA_sum, w_sum;
    Eigen::Vector2d XtWy;
    XtWy << wAY_sum, wY_sum;
    cache->XtWX[idx] = XtWX;
    cache->XtWy[idx] = XtWy;
    cache->signature_M[idx] = jac;
  }
  return cache;
}

inline std::vector<double> compute_rho_vector(
    const std::vector<double>& A,
    const std::vector<double>& Y,
    const std::vector<ThetaNu>& theta_nu,
    const CurrentTreeCache& curr_cache,
    const PrevRoundCache* prev_cache,
    double ridge_eps) {
  /**
   * @brief Compute the gradient residual ρ_i (Eq. 9) for the current tree.
   *
   * @param A Treatments for Gb.
   * @param Y Outcomes for Gb.
   * @param theta_nu Plug-in (θ̂, ν̂) for each Gb point.
   * @param curr_cache Per-leaf stats for the partial tree.
   * @param prev_cache Aggregated stats from dropout trees (may be null).
   * @param ridge_eps Diagonal ridge to stabilize 2x2 inversions.
   * @return std::vector<double> Residual vector used by the splitter.
   *
   * @behavior Sums the Jacobians from S_b and the current tree, solves the
   *           2x2 system, and returns the negative θ-component per Eq. (9).
   */
  size_t m = A.size();
  std::vector<Eigen::Matrix2d> total_M(m, Eigen::Matrix2d::Zero());

  if (prev_cache != nullptr) {
    // Each signature contributes its Jacobian block to the rows matching it.
    for (size_t i = 0; i < m; ++i) {
      int sig_idx = prev_cache->signature_indices[i];
      if (sig_idx >= 0) {
        total_M[i] += prev_cache->signature_M[sig_idx];
      }
    }
  }

  for (const auto& kv : curr_cache.stats) {
    const LeafStats& stats = kv.second;
    Eigen::Matrix2d contrib;
    contrib << stats.M00, stats.M01,
               stats.M10, stats.M11;
    for (int idx : stats.members) {
      total_M[idx] += contrib;
    }
  }

  std::vector<double> rhos(m, 0.0);
  for (size_t i = 0; i < m; ++i) {
    total_M[i](0, 0) += ridge_eps;  // Diagonal ridge to keep the 2x2 invertible.
    total_M[i](1, 1) += ridge_eps;
    double theta = theta_nu[i].theta;
    double nu = theta_nu[i].nu;
    double resid = Y[i] - A[i] * theta - nu;
    double v0 = resid * A[i];
    double v1 = resid;
    double a = total_M[i](0, 0);
    double b = total_M[i](0, 1);
    double c = total_M[i](1, 0);
    double d = total_M[i](1, 1);
    double det = a * d - b * c;
    if (std::abs(det) < ridge_eps) {
      det = (det >= 0 ? ridge_eps : -ridge_eps);
    }
    double u0 = (d * v0 - b * v1) / det;
    // double u1 = (-c * v0 + a * v1) / det;
    rhos[i] = -u0;
  }
  return rhos;
}

struct Stage2Cache {
  std::shared_ptr<std::vector<std::unordered_map<int, std::vector<int>>>> leaf_members;
  std::shared_ptr<std::vector<std::unordered_map<int, double>>> leaf_weights;
};

inline Stage2Cache build_stage2_cache(const std::vector<BoostedTree>& trees,
                                      const ColumnMajorMatrix& X2) {
  /**
   * @brief Pre-compute Stage 2 leaf memberships and α weights.
   *
   * @param trees All Stage 1 trees.
   * @param X2 Stage 2 feature matrix.
   * @return Stage2Cache containing leaf membership maps and denominators.
   *
   * @behavior For each tree, locates every D2 sample and stores both the
   *           member list and the Stage 1 denominator (|Gb ∩ leaf|).
   */
  Stage2Cache cache;
  cache.leaf_members = std::make_shared<std::vector<std::unordered_map<int, std::vector<int>>>>(trees.size());
  cache.leaf_weights = std::make_shared<std::vector<std::unordered_map<int, double>>>(trees.size());
  for (size_t b = 0; b < trees.size(); ++b) {
    auto& members_map = cache.leaf_members->at(b);
    const auto& count_map = trees[b].leaf_counts;
    for (size_t i = 0; i < X2.n_rows; ++i) {
      int leaf_id = locate_leaf(trees[b], X2, i);  // Push every Stage-2 point through tree b.
      members_map[leaf_id].push_back(static_cast<int>(i));
    }
    for (const auto& kv : members_map) {
      auto denom_it = count_map.find(kv.first);
      if (denom_it != count_map.end() && denom_it->second > 0) {
        (*cache.leaf_weights)[b][kv.first] = 1.0 / static_cast<double>(denom_it->second);
      }
    }
  }
  return cache;
}

/**
 * @brief Bernoulli subsample of indices.
 *
 * @param n Population size.
 * @param rate Inclusion probability per element.
 * @param rng Random number generator.
 * @return Vector of selected indices.
 *
 * @behavior Implements Algorithm 1 Step 3 (and honesty splits) by drawing
 *           without replacement using a Bernoulli mask.
 */
inline std::vector<int> subsample_indices(size_t n,
                                          double rate,
                                          std::mt19937_64& rng) {
  std::bernoulli_distribution dist(rate);
  std::vector<int> indices;
  indices.reserve(static_cast<size_t>(n * rate) + 4);
  for (size_t i = 0; i < n; ++i) {
    if (dist(rng)) {
      indices.push_back(static_cast<int>(i));
    }
  }
  return indices;
}

/**
 * @brief Sample dropout indices S_b ⊂ {0,…,b-1}.
 *
 * @param current_tree Current tree index b.
 * @param q Dropout probability.
 * @param rng Random number generator.
 * @return Vector of previous tree indices included in S_b.
 *
 * @behavior Mirrors Algorithm 1 Step 4 using independent Bernoulli(q) draws.
 */
inline std::vector<int> dropout_indices(size_t current_tree,
                                        double q,
                                        std::mt19937_64& rng) {
  std::bernoulli_distribution dist(q);
  std::vector<int> result;
  result.reserve(current_tree);
  for (size_t i = 0; i < current_tree; ++i) {
    if (dist(rng)) {
      result.push_back(static_cast<int>(i));
    }
  }
  return result;
}

class Algorithm1Model {
public:
  /**
   * @brief Owns Stage‑2 caches and evaluates Algorithm 1 predictions.
   *
   * @param num_features Feature dimension (for validation).
   * @param A2 Stage‑2 treatment vector.
   * @param Y2 Stage‑2 outcome vector.
   * @param trees Learned Stage‑1 trees.
   * @param dropout_sets Dropout indices per tree.
   * @param stage2 Precomputed Stage‑2 cache.
   */
  Algorithm1Model(size_t num_features,
                  std::vector<double>&& A2,
                  std::vector<double>&& Y2,
                  std::vector<BoostedTree>&& trees,
                  std::vector<std::vector<int>>&& dropout_sets,
                  Stage2Cache&& stage2)
      : num_features_(num_features),
        A2_(std::move(A2)),
        Y2_(std::move(Y2)),
        trees_(std::move(trees)),
        dropout_sets_(std::move(dropout_sets)),
        stage2_(std::move(stage2)) {}

  /**
   * @brief Release cached resources (shared_ptrs reset automatically).
   */
  ~Algorithm1Model() {
    trees_.clear();
    dropout_sets_.clear();
    stage2_.leaf_members.reset();
    stage2_.leaf_weights.reset();
    A2_.clear();
    Y2_.clear();
  }

  /**
   * @brief Predict (θ, ν) for new points via Algorithm 1 Stage 2.
   *
   * @param Xnew Feature matrix of query points.
   * @return Dict with arrays "theta" and "nu".
   *
   * @behavior Executes the β recursion from Eq. (10) using the stored trees
   *           and dropout sets, then runs a weighted least-squares solve.
   */
  py::dict predict_theta(const py::array_t<double, py::array::c_style | py::array::forcecast>& Xnew) const {
    auto Xmat = column_major_matrix(Xnew);
    size_t n = Xmat.n_rows;
    size_t p = Xmat.n_cols;
    if (p != num_features_) {
      throw shape_error("Feature dimension mismatch during prediction.");
    }
    std::vector<double> theta(n, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> nu(n, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> betas(A2_.size(), 0.0);

    auto add_alpha_scaled = [&](size_t tree_idx, int leaf_id, double coeff) {
      const auto& members_map = stage2_.leaf_members->at(tree_idx);
      auto mit = members_map.find(leaf_id);
      if (mit == members_map.end()) {
        return;
      }
      const auto& weight_map = stage2_.leaf_weights->at(tree_idx);
      auto wit = weight_map.find(leaf_id);
      if (wit == weight_map.end()) {
        return;
      }
      double w = coeff * wit->second;
      for (int idx : mit->second) {
        betas[static_cast<size_t>(idx)] += w;
      }
    };

    for (size_t row = 0; row < n; ++row) {
      std::fill(betas.begin(), betas.end(), 0.0);
      for (size_t b = 0; b < trees_.size(); ++b) {
        // Algorithm 1 Eq. 10: shrink old β and add λ_b-weighted dropout α's.
        double shrink = (b == 0) ? 0.0 : static_cast<double>(b) / static_cast<double>(b + 1);
        for (double& val : betas) {
          val *= shrink;
        }
        const auto& dropout = dropout_sets_[b];
        double lambda_b = 1.0 / (static_cast<double>(b + 1) * static_cast<double>(dropout.size() + 1));
        int leaf = locate_leaf(trees_[b], Xmat, row);
        add_alpha_scaled(b, leaf, lambda_b);
        for (int prev : dropout) {
          int prev_leaf = locate_leaf(trees_[static_cast<size_t>(prev)], Xmat, row);
          add_alpha_scaled(static_cast<size_t>(prev), prev_leaf, lambda_b);
        }
      }
      double sum_w = std::accumulate(betas.begin(), betas.end(), 0.0);
      if (sum_w <= 0.0) {
        continue;
      }
      double inv = 1.0 / sum_w;
      double sA = 0.0;
      double sA2 = 0.0;
      double sY = 0.0;
      double sAY = 0.0;
      double norm_sum = 0.0;
      for (size_t i = 0; i < betas.size(); ++i) {
        double w = betas[i] * inv;
        if (w <= 0.0) {
          continue;
        }
        double Ai = A2_[i];
        double Yi = Y2_[i];
        sA += w * Ai;
        sA2 += w * Ai * Ai;
        sY += w * Yi;
        sAY += w * Ai * Yi;
        norm_sum += w;
      }
      if (norm_sum <= 0.0) {
        continue;
      }
      ThetaNu est = solve_from_stats(sA2, sA, norm_sum, sAY, sY);
      theta[row] = est.theta;
      nu[row] = est.nu;
    }

    py::array_t<double> theta_arr(theta.size());
    std::memcpy(theta_arr.mutable_data(), theta.data(), theta.size() * sizeof(double));
    py::array_t<double> nu_arr(nu.size());
    std::memcpy(nu_arr.mutable_data(), nu.data(), nu.size() * sizeof(double));

    py::dict result;
    result["theta"] = theta_arr;
    result["nu"] = nu_arr;
    return result;
  }

  size_t num_trees() const {
    return trees_.size();
  }

private:
  size_t num_features_;
  std::vector<double> A2_;
  std::vector<double> Y2_;
  std::vector<BoostedTree> trees_;
  std::vector<std::vector<int>> dropout_sets_;
  Stage2Cache stage2_;
};

// ---------------------------------------------------------------------------
// Stage 1: faithful implementation of Algorithm 1 tree-structure boosting.
//
// The code below mirrors the pseudocode:
//   • Step 3: subsample G_b using xi1 * sample_fraction.
//   • Step 4: draw dropout set S_b.
//   • Step 5–6: compute (θ̂, ν̂) using cached moments for unique signatures.
//   • Step 7–9: grow the depth-limited tree using ρ_i residuals.
//   • Step 10: store α weights needed for Stage 2.
// ---------------------------------------------------------------------------
/**
 * @brief Train Algorithm 1 (Stage 1 + cached Stage 2) using GRF primitives.
 *
 * @param X1_in Structure-learning features (D1).
 * @param A1_in Treatments for D1.
 * @param Y1_in Outcomes for D1.
 * @param X2_in GRF-sample features (D2).
 * @param A2_in Treatments for D2.
 * @param Y2_in Outcomes for D2.
 * @param B Number of boosting rounds.
 * @param q Dropout probability.
 * @param xi1 Stage-1 subsample rate.
 * @param max_depth Maximum tree depth.
 * @param min_leaf Minimum samples per leaf.
 * @param ridge_eps Ridge regularization in Eq. (9).
 * @param sample_fraction Additional sampling factor (e.g., honesty splits).
 * @param honesty Whether to use honesty when growing trees.
 * @param honesty_fraction Fraction of Gb used for structure (rest for leaves).
 * @param honesty_prune_leaves Whether to prune empty leaves after honesty split.
 * @param seed Optional RNG seed.
 * @return Shared pointer owning the trained Algorithm 1 model.
 *
 * @behavior Implements Algorithm 1 verbatim: subsample Gb, sample dropout set,
 *           compute (θ̂, ν̂), grow tree with ρ residuals, store α-weights, and
 *           cache Stage‑2 information for prediction.
 */
std::shared_ptr<Algorithm1Model> train_algorithm1(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& X1_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& A1_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& Y1_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& X2_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& A2_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& Y2_in,
    int B,
    double q,
    double xi1,
    int max_depth,
    int min_leaf,
    double ridge_eps,
    double sample_fraction,
    bool honesty,
    double honesty_fraction,
    bool honesty_prune_leaves,
    std::optional<int> seed) {
  if (B <= 0) {
    throw shape_error("Number of boosting rounds must be positive.");
  }
  if (xi1 <= 0.0 || xi1 > 1.0) {
    throw shape_error("xi1 must lie in (0, 1].");
  }
  if (sample_fraction <= 0.0 || sample_fraction > 1.0) {
    throw shape_error("sample_fraction must lie in (0, 1].");
  }
  if (honesty) {
    if (honesty_fraction <= 0.0 || honesty_fraction >= 1.0) {
      throw shape_error("honesty_fraction must lie in (0, 1).");
    }
  }
  ColumnMajorMatrix X1 = column_major_matrix(X1_in);
  ColumnMajorMatrix X2 = column_major_matrix(X2_in);
  auto A1 = array1d_to_vector(A1_in);
  auto Y1 = array1d_to_vector(Y1_in);
  auto A2 = array1d_to_vector(A2_in);
  auto Y2 = array1d_to_vector(Y2_in);
  if (X1.n_rows != A1.size() || X1.n_rows != Y1.size()) {
    throw shape_error("D1 arrays must share the same number of rows.");
  }
  if (X2.n_rows != A2.size() || X2.n_rows != Y2.size()) {
    throw shape_error("D2 arrays must share the same number of rows.");
  }
  if (X1.n_cols != X2.n_cols) {
    throw shape_error("Feature dimension mismatch between X1 and X2.");
  }

  std::mt19937_64 rng(seed.has_value() ? static_cast<uint64_t>(seed.value()) : std::random_device{}());
  double draw_rate = std::min(1.0, xi1 * sample_fraction);
  if (draw_rate <= 0.0) {
    throw shape_error("Effective sample fraction is too small; adjust xi1 or sample_fraction.");
  }

  std::vector<BoostedTree> trees;
  trees.reserve(static_cast<size_t>(B));
  std::vector<std::vector<int>> dropout_sets(static_cast<size_t>(B));

  for (int b = 0; b < B; ++b) {
    // Step 3: draw the Gb subsample used to grow tree b.
    auto Gb_idx = subsample_indices(X1.n_rows, draw_rate, rng);
    if (Gb_idx.size() < static_cast<size_t>(2 * std::max(1, min_leaf))) {
      throw std::runtime_error("Subsample G_b too small; increase xi1 or reduce min_leaf.");
    }

    ColumnMajorMatrix XGb = subset_matrix(X1, Gb_idx);
    auto A_Gb = subset_vector(A1, Gb_idx);
    auto Y_Gb = subset_vector(Y1, Gb_idx);

    // Step 4: sample the dropout set S_b.
    dropout_sets[static_cast<size_t>(b)] = dropout_indices(static_cast<size_t>(b), q, rng);
    const auto& dropout = dropout_sets[static_cast<size_t>(b)];

    ThetaNu uniform = uniform_wls(A_Gb, Y_Gb);  // Fallback plug-in when no history exists.
    std::vector<ThetaNu> theta_nu(A_Gb.size(), uniform);

    std::vector<PrevTreeCache> prev_structs;
    prev_structs.reserve(dropout.size());
    for (int idx : dropout) {
      prev_structs.push_back(build_prev_tree_cache(trees[static_cast<size_t>(idx)], XGb, A_Gb, Y_Gb));
    }
    auto prev_cache = prepare_prev_round_cache(prev_structs);
    if (prev_cache) {
      // Step 5: solve for θ̂, ν̂ for each unique dropout signature.
      std::vector<ThetaNu> signature_betas(prev_cache->XtWX.size(), uniform);
      for (size_t sig = 0; sig < prev_cache->XtWX.size(); ++sig) {
        if (!prev_cache->contrib_mask[sig]) {
          continue;
        }
        signature_betas[sig] = solve_from_matrix(prev_cache->XtWX[sig], prev_cache->XtWy[sig]);
      }
      for (size_t i = 0; i < theta_nu.size(); ++i) {
        int sig_idx = prev_cache->signature_indices[i];
        if (sig_idx >= 0 && static_cast<size_t>(sig_idx) < signature_betas.size()) {
          theta_nu[i] = signature_betas[static_cast<size_t>(sig_idx)];
        }
      }
    }

    BoostedTree tree;
    std::vector<int> structure_indices(XGb.n_rows);
    std::iota(structure_indices.begin(), structure_indices.end(), 0);
    if (honesty) {
      std::shuffle(structure_indices.begin(), structure_indices.end(), rng);
      size_t grow_count = static_cast<size_t>(std::round(honesty_fraction * structure_indices.size()));
      grow_count = std::max(grow_count, static_cast<size_t>(2 * std::max(1, min_leaf)));
      grow_count = std::min(grow_count, structure_indices.size());
      if (grow_count < static_cast<size_t>(2 * std::max(1, min_leaf))) {
        throw std::runtime_error("Honesty settings produced too few samples for tree growing.");
      }
      structure_indices.resize(grow_count);
      std::sort(structure_indices.begin(), structure_indices.end());
    }

    TreeNode root;
    root.depth = 0;
    root.indices = structure_indices;
    tree.nodes.push_back(std::move(root));

    int depth = 0;
    while (true) {
      // Step 6/7: compute ρ_i for the current partial tree.
      CurrentTreeCache curr_cache = build_current_tree_cache(tree, XGb, A_Gb, Y_Gb);
      auto rhos = compute_rho_vector(A_Gb, Y_Gb, theta_nu, curr_cache, prev_cache.get(), ridge_eps);
      std::vector<int> frontier;
      for (size_t idx = 0; idx < tree.nodes.size(); ++idx) {
        if (tree.nodes[idx].depth == depth) {
          frontier.push_back(static_cast<int>(idx));
        }
      }
      bool any_split = false;
      for (int node_idx : frontier) {
        TreeNode& node = tree.nodes[static_cast<size_t>(node_idx)];
        SplitResult split = regression_best_split(
            XGb.data(),
            XGb.n_rows,
            XGb.n_cols,
            rhos,
            node.indices,
            min_leaf);
        if (!split.found || depth >= max_depth) {
          node.is_leaf = true;
          node.indices.clear();
          node.indices.shrink_to_fit();
          continue;
        }
        int child_depth = node.depth + 1;
        TreeNode left_child;
        left_child.depth = child_depth;
        left_child.indices = split.left_indices;
        TreeNode right_child;
        right_child.depth = child_depth;
        right_child.indices = split.right_indices;
        int left_idx = static_cast<int>(tree.nodes.size());
        tree.nodes.push_back(std::move(left_child));
        int right_idx = static_cast<int>(tree.nodes.size());
        tree.nodes.push_back(std::move(right_child));
        TreeNode& mut_node = tree.nodes[static_cast<size_t>(node_idx)];
        mut_node.is_leaf = false;
        mut_node.split_feature = split.feature;
        mut_node.split_threshold = split.threshold;
        mut_node.send_missing_left = split.send_missing_left;
        mut_node.left = left_idx;
        mut_node.right = right_idx;
        mut_node.indices.clear();
        mut_node.indices.shrink_to_fit();
        any_split = true;
      }
      if (!any_split || depth >= max_depth) {
        break;
      }
      depth += 1;
    }

    auto leaf_ids = compute_leaf_ids(tree, XGb);
    for (int lid : leaf_ids) {
      tree.leaf_counts[lid] += 1;  // Step 10: record |Gb ∩ leaf| for Stage 2 α weights.
    }
    trees.push_back(std::move(tree));
  }

Stage2Cache stage2 = build_stage2_cache(trees, X2);
  auto model = std::make_shared<Algorithm1Model>(
      X1.n_cols,
      std::move(A2),
      std::move(Y2),
      std::move(trees),
      std::move(dropout_sets),
      std::move(stage2));
  return model;
}

struct TrainingMatrix {
  std::vector<double> values;
  size_t n_rows;
  size_t num_cols;
  size_t outcome_index;
  std::optional<size_t> weight_index;
  std::optional<size_t> treatment_index;
  std::optional<size_t> instrument_index;
};

inline py::value_error shape_error(const std::string& msg) {
  return py::value_error("boosting_grf._grf: " + msg);
}

std::vector<double> to_column_major(const py::array_t<double, py::array::c_style | py::array::forcecast>& X_in);

/**
 * @brief Build a GRF training matrix [X | y | w] in column-major order.
 *
 * @param X_in Feature matrix (n × p).
 * @param y_in Outcome vector (n or n × 1).
 * @param sample_weight Optional weights vector.
 * @return TrainingMatrix describing the column-major storage and indices.
 *
 * @behavior Packages X, y, and (optionally) weights into a single buffer so
 *           the C++ GRF trainers can interpret it via grf::Data.
 */
TrainingMatrix build_training_matrix(const py::array_t<double, py::array::c_style | py::array::forcecast>& X_in,
                                     const py::array_t<double, py::array::c_style | py::array::forcecast>& y_in,
                                     const std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>>& sample_weight) {
  auto xbuf = X_in.request();
  auto ybuf = y_in.request();
  if (xbuf.ndim != 2) {
    throw shape_error("X must be a 2D array");
  }
  if (ybuf.ndim != 1 && !(ybuf.ndim == 2 && ybuf.shape[1] == 1)) {
    throw shape_error("y must be a 1D array or column vector");
  }

  size_t n = static_cast<size_t>(xbuf.shape[0]);
  size_t p = static_cast<size_t>(xbuf.shape[1]);
  if (static_cast<size_t>(ybuf.shape[0]) != n) {
    throw shape_error("X and y must have matching row counts");
  }
  if (n == 0 || p == 0) {
    throw shape_error("X must be non-empty");
  }

  std::optional<py::buffer_info> wbuf;
  if (sample_weight.has_value()) {
    wbuf.emplace(sample_weight->request());
    if (wbuf->ndim != 1 && !(wbuf->ndim == 2 && wbuf->shape[1] == 1)) {
      throw shape_error("sample_weight must be 1D or a column vector");
    }
    if (static_cast<size_t>(wbuf->shape[0]) != n) {
      throw shape_error("sample_weight must align with X rows");
    }
  }

  size_t extra_cols = 1 + (wbuf.has_value() ? 1 : 0);
  size_t num_cols = p + extra_cols;
  std::vector<double> values(n * num_cols, 0.0);

  const double* X_ptr = static_cast<const double*>(xbuf.ptr);
  const double* y_ptr = static_cast<const double*>(ybuf.ptr);

  for (size_t j = 0; j < p; ++j) {
    for (size_t i = 0; i < n; ++i) {
      values[j * n + i] = X_ptr[i * p + j];
    }
  }

  size_t outcome_index = p;
  for (size_t i = 0; i < n; ++i) {
    values[outcome_index * n + i] = (ybuf.ndim == 1)
        ? y_ptr[i]
        : y_ptr[i * ybuf.shape[1]];
  }

  std::optional<size_t> weight_index;
  if (wbuf.has_value()) {
    size_t column = p + 1;
    const double* w_ptr = static_cast<const double*>(wbuf->ptr);
    for (size_t i = 0; i < n; ++i) {
      values[column * n + i] = (wbuf->ndim == 1)
          ? w_ptr[i]
          : w_ptr[i * wbuf->shape[1]];
    }
    weight_index = column;
  }

  return TrainingMatrix{
      std::move(values),
      n,
      num_cols,
      outcome_index,
      weight_index,
      std::nullopt,
      std::nullopt
  };
}

/**
 * @brief Build a training matrix [X | y | A | w] for causal forests.
 *
 * @param X_in Feature matrix.
 * @param y_in Outcome vector.
 * @param treatment_in Treatment vector.
 * @param sample_weight Optional weight vector.
 * @return TrainingMatrix ready for causal forest trainers.
 *
 * @behavior Extends build_training_matrix by appending a treatment column
 *           (and optional weights) in column-major form.
 */
TrainingMatrix build_causal_training_matrix(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& X_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& y_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& treatment_in,
    const std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>>& sample_weight) {
  auto xbuf = X_in.request();
  auto ybuf = y_in.request();
  auto tbuf = treatment_in.request();

  if (xbuf.ndim != 2) {
    throw shape_error("X must be a 2D array");
  }
  if (ybuf.ndim != 1 && !(ybuf.ndim == 2 && ybuf.shape[1] == 1)) {
    throw shape_error("y must be 1D or a column vector");
  }
  if (tbuf.ndim != 1 && !(tbuf.ndim == 2 && tbuf.shape[1] == 1)) {
    throw shape_error("treatment must be 1D or a column vector");
  }

  size_t n = static_cast<size_t>(xbuf.shape[0]);
  size_t p = static_cast<size_t>(xbuf.shape[1]);
  if (static_cast<size_t>(ybuf.shape[0]) != n || static_cast<size_t>(tbuf.shape[0]) != n) {
    throw shape_error("X, y, and treatment must share the same number of rows");
  }
  if (n == 0 || p == 0) {
    throw shape_error("X must be non-empty");
  }

  std::optional<py::buffer_info> wbuf;
  if (sample_weight.has_value()) {
    wbuf.emplace(sample_weight->request());
    if (wbuf->ndim != 1 && !(wbuf->ndim == 2 && wbuf->shape[1] == 1)) {
      throw shape_error("sample_weight must be 1D or a column vector");
    }
    if (static_cast<size_t>(wbuf->shape[0]) != n) {
      throw shape_error("sample_weight must align with X rows");
    }
  }

  size_t extra_cols = 2 + (wbuf.has_value() ? 1 : 0);
  size_t num_cols = p + extra_cols;
  std::vector<double> values(n * num_cols, 0.0);

  const double* X_ptr = static_cast<const double*>(xbuf.ptr);
  const double* y_ptr = static_cast<const double*>(ybuf.ptr);
  const double* t_ptr = static_cast<const double*>(tbuf.ptr);

  for (size_t j = 0; j < p; ++j) {
    for (size_t i = 0; i < n; ++i) {
      values[j * n + i] = X_ptr[i * p + j];
    }
  }

  size_t outcome_index = p;
  for (size_t i = 0; i < n; ++i) {
    values[outcome_index * n + i] = (ybuf.ndim == 1)
        ? y_ptr[i]
        : y_ptr[i * ybuf.shape[1]];
  }

  size_t treatment_index = outcome_index + 1;
  for (size_t i = 0; i < n; ++i) {
    values[treatment_index * n + i] = (tbuf.ndim == 1)
        ? t_ptr[i]
        : t_ptr[i * tbuf.shape[1]];
  }

  std::optional<size_t> weight_index;
  if (wbuf.has_value()) {
    size_t column = treatment_index + 1;
    const double* w_ptr = static_cast<const double*>(wbuf->ptr);
    for (size_t i = 0; i < n; ++i) {
      values[column * n + i] = (wbuf->ndim == 1)
          ? w_ptr[i]
          : w_ptr[i * wbuf->shape[1]];
    }
    weight_index = column;
  }

  return TrainingMatrix{
      std::move(values),
      n,
      num_cols,
      outcome_index,
      weight_index,
      treatment_index,
      treatment_index
  };
}

/**
 * @brief Convert GRF Prediction objects into Python arrays.
 *
 * @param predictions Vector of predictions returned by GRF.
 * @return Dict containing "predictions", "variance", "error", "excess_error".
 *
 * @behavior Allocates numpy arrays of appropriate shape (or empty arrays if
 *           the forest did not compute those estimates).
 */
py::dict pack_predictions(const std::vector<Prediction>& predictions) {
  py::dict result;
  if (predictions.empty()) {
    result["predictions"] = py::array_t<double>(py::array::ShapeContainer{0});
    result["variance"] = py::array_t<double>(py::array::ShapeContainer{0});
    result["error"] = py::array_t<double>(py::array::ShapeContainer{0});
    result["excess_error"] = py::array_t<double>(py::array::ShapeContainer{0});
    return result;
  }

  size_t n = predictions.size();
  size_t dim = predictions.front().size();

  py::array_t<double> mean(py::array::ShapeContainer{static_cast<py::ssize_t>(n),
                                                     static_cast<py::ssize_t>(dim)});
  py::buffer_info mean_buf = mean.request();
  double* mean_ptr = static_cast<double*>(mean_buf.ptr);

  py::array_t<double> variance;
  py::array_t<double> error;
  py::array_t<double> excess_error;

  bool has_variance = predictions.front().contains_variance_estimates();
  bool has_error = predictions.front().contains_error_estimates();

  if (has_variance) {
    variance = py::array_t<double>(py::array::ShapeContainer{
        static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(dim)});
  } else {
    variance = py::array_t<double>(py::array::ShapeContainer{0});
  }

  if (has_error) {
    error = py::array_t<double>(
        py::array::ShapeContainer{static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(1)});
    excess_error = py::array_t<double>(
        py::array::ShapeContainer{static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(1)});
  } else {
    error = py::array_t<double>(py::array::ShapeContainer{0});
    excess_error = py::array_t<double>(py::array::ShapeContainer{0});
  }

  double* var_ptr = has_variance ? static_cast<double*>(variance.request().ptr) : nullptr;
  double* err_ptr = has_error ? static_cast<double*>(error.request().ptr) : nullptr;
  double* excess_ptr = has_error ? static_cast<double*>(excess_error.request().ptr) : nullptr;

  for (size_t i = 0; i < n; ++i) {
    const Prediction& pred = predictions[i];
    const auto& pred_vals = pred.get_predictions();
    for (size_t j = 0; j < dim; ++j) {
      mean_ptr[i * dim + j] = pred_vals[j];
    }

    if (has_variance) {
      const auto& var_vals = pred.get_variance_estimates();
      for (size_t j = 0; j < dim; ++j) {
        var_ptr[i * dim + j] = var_vals[j];
      }
    }

    if (has_error) {
      const auto& err_vals = pred.get_error_estimates();
      const auto& excess_vals = pred.get_excess_error_estimates();
      if (!err_vals.empty()) {
        err_ptr[i] = err_vals.front();
      }
      if (!excess_vals.empty()) {
        excess_ptr[i] = excess_vals.front();
      }
    }
  }

  result["predictions"] = mean;
  result["variance"] = variance;
  result["error"] = error;
  result["excess_error"] = excess_error;
  return result;
}

std::vector<double> to_column_major(const py::array_t<double, py::array::c_style | py::array::forcecast>& X_in) {
  auto buf = X_in.request();
  if (buf.ndim != 2) {
    throw shape_error("Input array must be 2D");
  }
  size_t n = static_cast<size_t>(buf.shape[0]);
  size_t p = static_cast<size_t>(buf.shape[1]);
  std::vector<double> values(n * p);
  const double* src = static_cast<const double*>(buf.ptr);
  for (size_t j = 0; j < p; ++j) {
    for (size_t i = 0; i < n; ++i) {
      values[j * n + i] = src[i * p + j];
    }
  }
  return values;
}

/**
 * @brief Owning wrapper for a trained GRF regression forest.
 *
 * Stores the serialized forest plus the column-major training matrix so we can
 * run predictions (including OOB) later from Python.
 */
class RegressionForestHandle {
public:
  /**
   * @brief Construct a handle with training storage references.
   *
    * @param forest_ptr Pointer to the trained forest.
    * @param train_matrix Column-major training data (shared with grf::Data).
    * @param n_rows Number of training rows.
    * @param num_cols Number of columns (features + outcome + weights).
    * @param outcome_index Column index for the outcome.
    * @param weight_index Optional column index for sample weights.
    */
  RegressionForestHandle(std::shared_ptr<Forest> forest_ptr,
                         std::vector<double>&& train_matrix,
                         size_t n_rows,
                         size_t num_cols,
                         size_t outcome_index,
                         std::optional<size_t> weight_index):
      forest(std::move(forest_ptr)),
      train_data_storage(std::move(train_matrix)),
      n_train(n_rows),
      num_train_cols(num_cols),
      y_index(outcome_index),
      w_index(std::move(weight_index)) {}

  /**
   * @brief Return the number of trees stored in the forest.
   */
  size_t num_trees() const {
    return forest ? forest->get_trees().size() : 0;
  }

  /**
   * @brief Predict on a test matrix.
   *
   * @param X_test Feature matrix (n × p).
   * @param estimate_variance Whether to request variance estimates.
   * @param num_threads Number of threads for GRF prediction.
   * @return Dict with prediction outputs (means/variance/error/etc).
   */
  py::dict predict(const py::array_t<double, py::array::c_style | py::array::forcecast>& X_test,
                   bool estimate_variance,
                   grf::uint num_threads) const {
    ensure_forest();
    auto cm = to_column_major(X_test);
    auto buf = X_test.request();
    size_t n = static_cast<size_t>(buf.shape[0]);
    size_t p = static_cast<size_t>(buf.shape[1]);
    if (p != y_index) {
      throw shape_error("X_test column mismatch: expected " + std::to_string(y_index) + " features");
    }

    grf::Data train_data(train_data_storage.data(), n_train, num_train_cols);
    train_data.set_outcome_index(y_index);
    if (w_index.has_value()) {
      train_data.set_weight_index(w_index.value());
    }

    grf::Data test_data(cm.data(), n, p);
    ForestPredictor predictor = grf::regression_predictor(num_threads);
    auto preds = predictor.predict(*forest, train_data, test_data, estimate_variance);
    return pack_predictions(preds);
  }

  /**
   * @brief Compute out-of-bag predictions on the training data.
   *
   * @param estimate_variance Whether to request variance estimates.
   * @param num_threads Number of threads.
   * @return Dict with OOB predictions.
   */
  py::dict predict_oob(bool estimate_variance,
                       grf::uint num_threads) const {
    ensure_forest();
    grf::Data data(train_data_storage.data(), n_train, num_train_cols);
    data.set_outcome_index(y_index);
    if (w_index.has_value()) {
      data.set_weight_index(w_index.value());
    }

    ForestPredictor predictor = grf::regression_predictor(num_threads);
    auto preds = predictor.predict_oob(*forest, data, estimate_variance);
    return pack_predictions(preds);
  }

  /**
   * @brief Access the underlying forest (throws if empty).
   */
  const Forest& get_forest() const {
    ensure_forest();
    return *forest;
  }

private:
  /** @brief Throw if the forest pointer is null. */
  void ensure_forest() const {
    if (!forest) {
      throw std::runtime_error("Forest handle is empty");
    }
  }

  std::shared_ptr<Forest> forest;
  std::vector<double> train_data_storage;
  size_t n_train;
  size_t num_train_cols;
  size_t y_index;
  std::optional<size_t> w_index;
};

/**
 * @brief Handle for trained causal (instrumental) forests.
 *
 * Similar to RegressionForestHandle but also stores treatment/instrument
 * indices so causal predictions can be made.
 */
class CausalForestHandle {
public:
  /**
   * @brief Construct the causal forest handle.
   *
   * @param forest_ptr Pointer to forest.
   * @param train_matrix Column-major training matrix.
   * @param n_rows Number of training rows.
   * @param num_cols Columns in the training matrix.
   * @param outcome_index Outcome column index.
   * @param treatment_index Treatment column index.
   * @param weight_index Optional weight column.
   */
  CausalForestHandle(std::shared_ptr<Forest> forest_ptr,
                     std::vector<double>&& train_matrix,
                     size_t n_rows,
                     size_t num_cols,
                     size_t outcome_index,
                     size_t treatment_index,
                     std::optional<size_t> weight_index):
      forest(std::move(forest_ptr)),
      train_data_storage(std::move(train_matrix)),
      n_train(n_rows),
      num_train_cols(num_cols),
      y_index(outcome_index),
      t_index(treatment_index),
      w_index(std::move(weight_index)) {}

  size_t num_trees() const {
    return forest ? forest->get_trees().size() : 0;
  }

  /**
   * @brief Predict causal effects for new data.
   *
   * @param X_test Feature matrix.
   * @param estimate_variance Whether to output variance estimates.
   * @param num_threads Number of threads.
   * @return Dict with predictions.
   */
  py::dict predict(const py::array_t<double, py::array::c_style | py::array::forcecast>& X_test,
                   bool estimate_variance,
                   grf::uint num_threads) const {
    ensure_forest();
    auto cm = to_column_major(X_test);
    auto buf = X_test.request();
    size_t n = static_cast<size_t>(buf.shape[0]);
    size_t p = static_cast<size_t>(buf.shape[1]);
    if (p != y_index) {
      throw shape_error("X_test column mismatch: expected " + std::to_string(y_index) + " features");
    }

    grf::Data train_data(train_data_storage.data(), n_train, num_train_cols);
    train_data.set_outcome_index(y_index);
    train_data.set_treatment_index(t_index);
    train_data.set_instrument_index(t_index);
    if (w_index.has_value()) {
      train_data.set_weight_index(w_index.value());
    }

    grf::Data test_data(cm.data(), n, p);
    ForestPredictor predictor = grf::instrumental_predictor(num_threads);
    auto preds = predictor.predict(*forest, train_data, test_data, estimate_variance);
    return pack_predictions(preds);
  }

  /**
   * @brief Return out-of-bag predictions for causal forest.
   *
   * @param estimate_variance Flag for variance estimates.
   * @param num_threads Number of threads.
   * @return Dict mirroring predict() but using OOB samples.
   */
  py::dict predict_oob(bool estimate_variance,
                       grf::uint num_threads) const {
    ensure_forest();
    grf::Data data(train_data_storage.data(), n_train, num_train_cols);
    data.set_outcome_index(y_index);
    data.set_treatment_index(t_index);
    data.set_instrument_index(t_index);
    if (w_index.has_value()) {
      data.set_weight_index(w_index.value());
    }

    ForestPredictor predictor = grf::instrumental_predictor(num_threads);
    auto preds = predictor.predict_oob(*forest, data, estimate_variance);
    return pack_predictions(preds);
  }

private:
  void ensure_forest() const {
    if (!forest) {
      throw std::runtime_error("Forest handle is empty");
    }
  }

  std::shared_ptr<Forest> forest;
  std::vector<double> train_data_storage;
  size_t n_train;
  size_t num_train_cols;
  size_t y_index;
  size_t t_index;
  std::optional<size_t> w_index;
};

}  // namespace detail

/**
 * @brief Python entrypoint for training regression forests.
 *
 * @param X_in Feature matrix.
 * @param y_in Outcome vector.
 * @param sample_weight Optional weights.
 * @param mtry Optional feature subsampling size.
 * @param num_trees Number of trees.
 * @param min_node_size Minimum node size.
 * @param sample_fraction Subsample fraction per tree.
 * @param honesty Honesty flag.
 * @param honesty_fraction Honest split fraction.
 * @param honesty_prune_leaves Whether to prune empty leaves.
 * @param ci_group_size CI group size.
 * @param alpha Regularization parameter.
 * @param imbalance_penalty Penalty for unbalanced splits.
 * @param compute_oob_predictions Whether to precompute OOB predictions.
 * @param num_threads Thread count.
 * @param seed Seed.
 * @param legacy_seed Legacy seed flag.
 * @return RegressionForestHandle owning the trained forest.
 */
detail::RegressionForestHandle train_regression_forest(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& X_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& y_in,
    std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>> sample_weight,
    std::optional<unsigned int> mtry,
    unsigned int num_trees,
    unsigned int min_node_size,
    double sample_fraction,
    bool honesty,
    double honesty_fraction,
    bool honesty_prune_leaves,
    size_t ci_group_size,
    double alpha,
    double imbalance_penalty,
    bool compute_oob_predictions,
    unsigned int num_threads,
    unsigned int seed,
    bool legacy_seed) {
  auto tm = detail::build_training_matrix(X_in, y_in, sample_weight);

  grf::Data data(tm.values.data(), tm.n_rows, tm.num_cols);
  data.set_outcome_index(tm.outcome_index);
  if (tm.weight_index.has_value()) {
    data.set_weight_index(tm.weight_index.value());
  }

  size_t p = tm.outcome_index;
  unsigned int resolved_mtry = mtry.has_value()
      ? mtry.value()
      : static_cast<unsigned int>(std::max<size_t>(1, std::sqrt(static_cast<double>(p))));

  grf::ForestTrainer trainer = grf::regression_trainer();
  ForestOptions options(
      num_trees,
      ci_group_size,
      sample_fraction,
      resolved_mtry,
      min_node_size,
      honesty,
      honesty_fraction,
      honesty_prune_leaves,
      alpha,
      imbalance_penalty,
      num_threads,
      seed,
      legacy_seed,
      std::vector<size_t>(),
      0);

  Forest forest = trainer.train(data, options);
  std::shared_ptr<Forest> forest_ptr(new Forest(std::move(forest)));

  detail::RegressionForestHandle handle(
      std::move(forest_ptr),
      std::move(tm.values),
      tm.n_rows,
      tm.num_cols,
      tm.outcome_index,
      tm.weight_index);

  if (compute_oob_predictions) {
    // Pre-compute OOB predictions to ensure the C++ code path is exercised.
    (void) handle.predict_oob(false, num_threads);
  }

  return handle;
}

/**
 * @brief Predict using a regression forest handle.
 *
 * @param handle Trained handle.
 * @param X_test Feature matrix.
 * @param estimate_variance Whether to compute variance.
 * @param num_threads Thread count.
 * @return Dict of predictions.
 */
py::dict predict_regression_forest(const detail::RegressionForestHandle& handle,
                                   const py::array_t<double, py::array::c_style | py::array::forcecast>& X_test,
                                   bool estimate_variance,
                                   unsigned int num_threads) {
  return handle.predict(X_test, estimate_variance, num_threads);
}

/**
 * @brief Out-of-bag predictions for regression forests.
 *
 * @param handle Trained handle.
 * @param estimate_variance Whether to compute variance.
 * @param num_threads Thread count.
 * @return Dict with OOB predictions.
 */
py::dict oob_predict_regression_forest(const detail::RegressionForestHandle& handle,
                                       bool estimate_variance,
                                       unsigned int num_threads) {
  return handle.predict_oob(estimate_variance, num_threads);
}

/**
 * @brief Python entrypoint for training causal (instrumental) forests.
 *
 * @param X_in Feature matrix.
 * @param y_in Outcome vector.
 * @param treatment_in Treatment assignments.
 * @param sample_weight Optional weights.
 * @param mtry Optional feature subsampling size.
 * @param num_trees Number of trees.
 * @param min_node_size Minimum node size.
 * @param sample_fraction Subsample fraction.
 * @param honesty Honesty flag.
 * @param honesty_fraction Honest split fraction.
 * @param honesty_prune_leaves Whether to prune empty leaves.
 * @param ci_group_size CI group size.
 * @param reduced_form_weight Reduced-form mixing parameter.
 * @param alpha Regularization parameter.
 * @param imbalance_penalty Split penalty.
 * @param stabilize_splits Whether to use instrumental split rules.
 * @param compute_oob_predictions Whether to precompute OOB predictions.
 * @param num_threads Thread count.
 * @param seed Seed.
 * @param legacy_seed Legacy seed flag.
 * @return CausalForestHandle owning the trained forest.
 */
detail::CausalForestHandle train_causal_forest(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& X_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& y_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& treatment_in,
    std::optional<py::array_t<double, py::array::c_style | py::array::forcecast>> sample_weight,
    std::optional<unsigned int> mtry,
    unsigned int num_trees,
    unsigned int min_node_size,
    double sample_fraction,
    bool honesty,
    double honesty_fraction,
    bool honesty_prune_leaves,
    size_t ci_group_size,
    double reduced_form_weight,
    double alpha,
    double imbalance_penalty,
    bool stabilize_splits,
    bool compute_oob_predictions,
    unsigned int num_threads,
    unsigned int seed,
    bool legacy_seed) {
  auto tm = detail::build_causal_training_matrix(X_in, y_in, treatment_in, sample_weight);

  grf::Data data(tm.values.data(), tm.n_rows, tm.num_cols);
  data.set_outcome_index(tm.outcome_index);
  data.set_treatment_index(tm.treatment_index.value());
  data.set_instrument_index(tm.instrument_index.value());
  if (tm.weight_index.has_value()) {
    data.set_weight_index(tm.weight_index.value());
  }

  size_t p = tm.outcome_index;
  unsigned int resolved_mtry = mtry.has_value()
      ? mtry.value()
      : static_cast<unsigned int>(std::max<size_t>(1, std::sqrt(static_cast<double>(p))));

  grf::ForestTrainer trainer = grf::instrumental_trainer(reduced_form_weight, stabilize_splits);
  ForestOptions options(
      num_trees,
      ci_group_size,
      sample_fraction,
      resolved_mtry,
      min_node_size,
      honesty,
      honesty_fraction,
      honesty_prune_leaves,
      alpha,
      imbalance_penalty,
      num_threads,
      seed,
      legacy_seed,
      std::vector<size_t>(),
      0);

  Forest forest = trainer.train(data, options);
  std::shared_ptr<Forest> forest_ptr(new Forest(std::move(forest)));

  detail::CausalForestHandle handle(
      std::move(forest_ptr),
      std::move(tm.values),
      tm.n_rows,
      tm.num_cols,
      tm.outcome_index,
      tm.treatment_index.value(),
      tm.weight_index);

  if (compute_oob_predictions) {
    (void) handle.predict_oob(false, num_threads);
  }

  return handle;
}

/**
 * @brief Predict causal effects with a causal forest handle.
 *
 * @param handle Handle produced by train_causal_forest.
 * @param X_test Feature matrix.
 * @param estimate_variance Whether to compute variance.
 * @param num_threads Thread count.
 * @return Dict of predictions.
 */
py::dict predict_causal_forest(const detail::CausalForestHandle& handle,
                               const py::array_t<double, py::array::c_style | py::array::forcecast>& X_test,
                               bool estimate_variance,
                               unsigned int num_threads) {
  return handle.predict(X_test, estimate_variance, num_threads);
}

/**
 * @brief Out-of-bag predictions for causal forests.
 *
 * @param handle Handle produced by train_causal_forest.
 * @param estimate_variance Whether to compute variance.
 * @param num_threads Thread count.
 * @return Dict of OOB predictions.
 */
py::dict oob_predict_causal_forest(const detail::CausalForestHandle& handle,
                                   bool estimate_variance,
                                   unsigned int num_threads) {
  return handle.predict_oob(estimate_variance, num_threads);
}

/**
 * @brief Python wrapper for GRF's best split search on dense arrays.
 *
 * @param X_in Feature matrix (Fortran contiguous).
 * @param rho_in Residual vector.
 * @param node_indices_in Indices of samples in the node.
 * @param min_leaf Minimum samples per child.
 * @return Dict describing the split or None if unsplittable.
 *
 * @behavior Convenience helper that exposes Algorithm 1's CART procedure to
 *           Python for debugging/analysis.
 */
py::object best_split_regression(
    const py::array_t<double, py::array::f_style | py::array::forcecast>& X_in,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& rho_in,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& node_indices_in,
    int min_leaf) {
  auto xbuf = X_in.request();
  if (xbuf.ndim != 2) {
    throw detail::shape_error("X must be a 2D Fortran-contiguous array");
  }
  size_t n = static_cast<size_t>(xbuf.shape[0]);
  size_t p = static_cast<size_t>(xbuf.shape[1]);
  if (n == 0 || p == 0) {
    return py::none();
  }

  auto rbuf = rho_in.request();
  if (rbuf.ndim != 1 || static_cast<size_t>(rbuf.shape[0]) != n) {
    throw detail::shape_error("rho must be a vector aligned with X rows");
  }

  auto ibuf = node_indices_in.request();
  if (ibuf.ndim != 1) {
    throw detail::shape_error("node_indices must be 1D");
  }
  const double* X_ptr = static_cast<const double*>(xbuf.ptr);
  const double* rho_ptr = static_cast<const double*>(rbuf.ptr);
  const int64_t* idx_ptr = static_cast<const int64_t*>(ibuf.ptr);

  std::vector<double> rho_vec(n);
  for (size_t i = 0; i < n; ++i) {
    rho_vec[i] = rho_ptr[i];
  }
  std::vector<int> node_indices(static_cast<size_t>(ibuf.shape[0]));
  for (size_t i = 0; i < node_indices.size(); ++i) {
    int64_t raw_idx = idx_ptr[i];
    if (raw_idx < 0 || static_cast<size_t>(raw_idx) >= n) {
      throw detail::shape_error("node_indices contain out-of-range entries");
    }
    node_indices[i] = static_cast<int>(raw_idx);
  }

  auto split = detail::regression_best_split(X_ptr, n, p, rho_vec, node_indices, min_leaf);
  if (!split.found) {
    return py::none();
  }

  auto to_array = [](const std::vector<int>& vec) {
    py::array_t<int64_t> arr(vec.size());
    auto buf = arr.request();
    auto* ptr = static_cast<int64_t*>(buf.ptr);
    for (size_t i = 0; i < vec.size(); ++i) {
      ptr[i] = static_cast<int64_t>(vec[i]);
    }
    return arr;
  };

  py::dict result;
  result["feature"] = split.feature;
  result["threshold"] = split.threshold;
  result["send_missing_left"] = split.send_missing_left;
  result["left_indices"] = to_array(split.left_indices);
  result["right_indices"] = to_array(split.right_indices);
  return result;
}

PYBIND11_MODULE(_grf, m) {
  m.doc() = "Pybind11 bindings for the GRF C++ core";

  py::class_<detail::RegressionForestHandle>(m, "RegressionForest")
      .def("predict",
           &detail::RegressionForestHandle::predict,
           py::arg("X_test"),
           py::arg("estimate_variance") = false,
           py::arg("num_threads") = grf::DEFAULT_NUM_THREADS)
      .def("predict_oob",
           &detail::RegressionForestHandle::predict_oob,
           py::arg("estimate_variance") = false,
           py::arg("num_threads") = grf::DEFAULT_NUM_THREADS)
      .def_property_readonly("num_trees", &detail::RegressionForestHandle::num_trees);

  py::class_<detail::CausalForestHandle>(m, "CausalForest")
      .def("predict",
           &detail::CausalForestHandle::predict,
           py::arg("X_test"),
           py::arg("estimate_variance") = false,
           py::arg("num_threads") = grf::DEFAULT_NUM_THREADS)
      .def("predict_oob",
           &detail::CausalForestHandle::predict_oob,
           py::arg("estimate_variance") = false,
           py::arg("num_threads") = grf::DEFAULT_NUM_THREADS)
      .def_property_readonly("num_trees", &detail::CausalForestHandle::num_trees);

  py::class_<detail::Algorithm1Model, std::shared_ptr<detail::Algorithm1Model>>(m, "Algorithm1Model")
      .def("predict_theta",
           &detail::Algorithm1Model::predict_theta,
           py::arg("X_new"))
      .def_property_readonly("num_trees", &detail::Algorithm1Model::num_trees);

  m.def(
      "train_regression_forest",
      &train_regression_forest,
      py::arg("X"),
      py::arg("y"),
      py::arg("sample_weight") = py::none(),
      py::arg("mtry") = py::none(),
      py::arg("num_trees") = 2000,
      py::arg("min_node_size") = 5,
      py::arg("sample_fraction") = 0.5,
      py::arg("honesty") = true,
      py::arg("honesty_fraction") = 0.5,
      py::arg("honesty_prune_leaves") = true,
      py::arg("ci_group_size") = 1,
      py::arg("alpha") = 0.05,
      py::arg("imbalance_penalty") = 0.0,
      py::arg("compute_oob_predictions") = false,
      py::arg("num_threads") = grf::DEFAULT_NUM_THREADS,
      py::arg("seed") = 0,
      py::arg("legacy_seed") = false,
      R"pbdoc(
        Train a regression forest using the GRF C++ core.

        Parameters mirror the R package defaults. Arrays are copied into column-major
        storage required by GRF, so X and y are left untouched.
      )pbdoc");

  m.def(
      "predict_regression_forest",
      &predict_regression_forest,
      py::arg("forest"),
      py::arg("X_test"),
      py::arg("estimate_variance") = false,
      py::arg("num_threads") = grf::DEFAULT_NUM_THREADS);

  m.def(
      "predict_regression_forest_oob",
      &oob_predict_regression_forest,
      py::arg("forest"),
      py::arg("estimate_variance") = false,
      py::arg("num_threads") = grf::DEFAULT_NUM_THREADS);

  m.def(
      "train_algorithm1",
      &detail::train_algorithm1,
      py::arg("X1"),
      py::arg("A1"),
      py::arg("Y1"),
      py::arg("X2"),
      py::arg("A2"),
      py::arg("Y2"),
      py::arg("B"),
      py::arg("q"),
      py::arg("xi1"),
      py::arg("max_depth") = 5,
      py::arg("min_leaf") = 10,
      py::arg("ridge_eps") = 1e-8,
      py::arg("sample_fraction") = 1.0,
      py::arg("honesty") = false,
      py::arg("honesty_fraction") = 0.5,
      py::arg("honesty_prune_leaves") = false,
      py::arg("seed") = py::none());

  m.def(
      "best_split_regression",
      &best_split_regression,
      py::arg("X"),
      py::arg("rho"),
      py::arg("node_indices"),
      py::arg("min_leaf"));

  m.def(
      "train_causal_forest",
      &train_causal_forest,
      py::arg("X"),
      py::arg("y"),
      py::arg("treatment"),
      py::arg("sample_weight") = py::none(),
      py::arg("mtry") = py::none(),
      py::arg("num_trees") = 2000,
      py::arg("min_node_size") = 5,
      py::arg("sample_fraction") = 0.5,
      py::arg("honesty") = true,
      py::arg("honesty_fraction") = 0.5,
      py::arg("honesty_prune_leaves") = true,
      py::arg("ci_group_size") = 1,
      py::arg("reduced_form_weight") = 0.0,
      py::arg("alpha") = 0.05,
      py::arg("imbalance_penalty") = 0.0,
      py::arg("stabilize_splits") = true,
      py::arg("compute_oob_predictions") = false,
      py::arg("num_threads") = grf::DEFAULT_NUM_THREADS,
      py::arg("seed") = 0,
      py::arg("legacy_seed") = false);

  m.def(
      "predict_causal_forest",
      &predict_causal_forest,
      py::arg("forest"),
      py::arg("X_test"),
      py::arg("estimate_variance") = false,
      py::arg("num_threads") = grf::DEFAULT_NUM_THREADS);

  m.def(
      "predict_causal_forest_oob",
      &oob_predict_causal_forest,
      py::arg("forest"),
      py::arg("estimate_variance") = false,
      py::arg("num_threads") = grf::DEFAULT_NUM_THREADS);
}

}  // namespace boosting_grf
