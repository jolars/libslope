/**
 * @file
 * @brief Internal helpers for extrapolating dual certificates.
 */

#pragma once

#include <Eigen/Core>
#include <array>
#include <optional>
#include <string>

namespace slope::detail {

/**
 * @brief Extrapolates the fixed point of a sequence of linear predictors.
 *
 * The five-step construction uses six consecutive predictors, following the
 * vector autoregressive dual extrapolation used by Celer.
 */
class DualExtrapolator
{
public:
  std::optional<Eigen::MatrixXd> push(const Eigen::MatrixXd& point,
                                      bool extrapolate = true);
  void reset();

private:
  static constexpr int order = 5;
  static constexpr int history_size = order + 1;

  const Eigen::MatrixXd& pointAt(int index) const;

  std::array<Eigen::MatrixXd, history_size> history;
  Eigen::MatrixXd differences;
  Eigen::Index rows = 0;
  Eigen::Index cols = 0;
  int count = 0;
  int next = 0;
};

bool
dualExtrapolationEnabled(const std::string& loss_type,
                         const std::string& resolved_solver_type);

} // namespace slope::detail
