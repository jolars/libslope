#include "slope_threshold.h"
#include "../math.h"

namespace slope {

std::tuple<double, int>
slopeThreshold(const double x,
               const int j,
               const Eigen::ArrayXd lambdas,
               const Clusters& clusters)
{
  using std::size_t;

  assert(j >= 0 && j < clusters.size());

  const size_t cluster_size = clusters.cluster_size(j);
  const double abs_x = std::abs(x);
  const int sign_x = sign(x);
  const size_t n_lambda = lambdas.size();

  // Prepare a lazy cumulative sum of lambdas.
  // cumsum[i] holds sum_{k=0}^{i-1} lambdas(k) with cumsum[0] = 0.
  std::vector<double> cumsum(n_lambda + 1, 0.0);
  size_t computed = 0; // Last index for which cum has been computed.

  // getCum(i) computes and returns cumsum[i] on demand.
  auto getCumSum = [&](size_t i) -> double {
    while (computed < i) {
      computed++;
      cumsum[computed] = cumsum[computed - 1] + lambdas(computed - 1);
    }
    return cumsum[i];
  };

  // getLambdaSum(start, len) returns sum of lambdas from start to start+len-1
  auto getLambdaSum = [&](size_t start, size_t len) -> double {
    return getCumSum(start + len) - getCumSum(start);
  };

  // Determine whether the update moves upward.
  int ptr_j = clusters.pointer(j);
  const bool direction_up =
    abs_x - getLambdaSum(ptr_j, cluster_size) > clusters.coeff(j);

  if (direction_up) {
    size_t start = clusters.pointer(j);
    double lo = getLambdaSum(start, cluster_size);

    for (int k = j - 1; k >= 0; --k) {
      double c_k = clusters.coeff(k);

      if (abs_x - lo < c_k && k < j) {
        return { x - sign_x * lo, k + 1 };
      }

      start = clusters.pointer(k);
      double hi = getLambdaSum(start, cluster_size);

      if (abs_x - hi <= c_k) {
        return { sign_x * c_k, k };
      }

      lo = hi;
    }

    return { x - sign_x * lo, 0 };
  } else {
    // Moving down in the cluster ordering
    int end = clusters.pointer(j + 1);
    double hi = getLambdaSum(end - cluster_size, cluster_size);

    for (int k = j + 1; k < clusters.size(); ++k) {
      end = clusters.pointer(k + 1);

      double c_k = clusters.coeff(k);

      if (abs_x > hi + c_k) {
        return { x - sign_x * hi, k - 1 };
      }

      double lo = getLambdaSum(end - cluster_size, cluster_size);

      if (abs_x >= lo + c_k) {
        return { sign_x * c_k, k };
      }

      hi = lo;
    }

    if (abs_x > hi) {
      return { x - sign_x * hi, clusters.size() - 1 };
    } else {
      // Zero cluster case
      return { 0, clusters.size() };
    }
  }
}

}
