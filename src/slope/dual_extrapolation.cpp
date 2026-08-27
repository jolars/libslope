#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <slope/dual_extrapolation.h>

namespace slope::detail {

std::optional<Eigen::MatrixXd>
DualExtrapolator::push(const Eigen::MatrixXd& point, const bool extrapolate)
{
  if (!point.allFinite()) {
    reset();
    return std::nullopt;
  }

  if (count > 0 && (point.rows() != rows || point.cols() != cols)) {
    reset();
  }

  if (count == 0) {
    rows = point.rows();
    cols = point.cols();
  }

  history[next] = point;
  next = (next + 1) % history_size;
  count = std::min(count + 1, history_size);

  if (count < history_size || !extrapolate) {
    return std::nullopt;
  }

  differences.resize(point.size(), order);
  for (int i = 0; i < order; ++i) {
    differences.col(i) =
      Eigen::Map<const Eigen::VectorXd>(pointAt(i + 1).data(), point.size()) -
      Eigen::Map<const Eigen::VectorXd>(pointAt(i).data(), point.size());
  }

  const double scale = differences.cwiseAbs().maxCoeff();
  if (!std::isfinite(scale) || scale == 0.0) {
    return std::nullopt;
  }
  differences.array() /= scale;

  using GramMatrix = Eigen::Matrix<double, order, order>;
  const GramMatrix gram = differences.transpose() * differences;
  if (!gram.allFinite()) {
    return std::nullopt;
  }

  Eigen::LLT<GramMatrix> factorization(gram);
  if (factorization.info() != Eigen::Success) {
    return std::nullopt;
  }

  const Eigen::Matrix<double, order, 1> coefficients =
    factorization.solve(Eigen::Matrix<double, order, 1>::Ones());
  if (factorization.info() != Eigen::Success || !coefficients.allFinite()) {
    return std::nullopt;
  }

  const double denominator = coefficients.sum();
  const double denominator_tolerance =
    64.0 * std::numeric_limits<double>::epsilon() *
    std::max(1.0, coefficients.cwiseAbs().sum());
  if (!std::isfinite(denominator) ||
      std::abs(denominator) <= denominator_tolerance) {
    return std::nullopt;
  }

  const Eigen::Matrix<double, order, 1> weights = coefficients / denominator;
  if (!weights.allFinite()) {
    return std::nullopt;
  }

  Eigen::MatrixXd extrapolated = Eigen::MatrixXd::Zero(rows, cols);
  for (int i = 0; i < order; ++i) {
    extrapolated.noalias() += weights(i) * pointAt(i);
  }

  if (!extrapolated.allFinite()) {
    return std::nullopt;
  }

  return extrapolated;
}

void
DualExtrapolator::reset()
{
  rows = 0;
  cols = 0;
  count = 0;
  next = 0;
}

const Eigen::MatrixXd&
DualExtrapolator::pointAt(const int index) const
{
  const int oldest = count == history_size ? next : 0;
  return history[(oldest + index) % history_size];
}

bool
dualExtrapolationEnabled(const std::string& loss_type,
                         const std::string& resolved_solver_type)
{
  const bool supported_loss =
    loss_type == "logistic" || loss_type == "quadratic";
  const bool supported_solver =
    resolved_solver_type == "hybrid" || resolved_solver_type == "pgd";
  return supported_loss && supported_solver;
}

} // namespace slope::detail
