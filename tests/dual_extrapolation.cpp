#include <Eigen/Core>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <slope/dual_extrapolation.h>
#include <slope/solvers/setup_solver.h>

namespace {

Eigen::MatrixXd
contractingPoint(const int iteration)
{
  constexpr std::array<double, 5> rates{ 0.05, 0.2, 0.4, 0.6, 0.8 };
  Eigen::MatrixXd point(3, 2);

  for (Eigen::Index i = 0; i < point.size(); ++i) {
    point(i) = std::pow(rates[i % rates.size()], iteration);
  }

  return point;
}

} // namespace

TEST_CASE("Dual extrapolation accelerates a fixed-point sequence",
          "[dual][extrapolation]")
{
  slope::detail::DualExtrapolator extrapolator;

  for (int iteration = 0; iteration < 5; ++iteration) {
    REQUIRE_FALSE(extrapolator.push(contractingPoint(iteration)).has_value());
  }

  const Eigen::MatrixXd newest = contractingPoint(5);
  const auto extrapolated = extrapolator.push(newest);

  REQUIRE(extrapolated.has_value());
  REQUIRE(extrapolated->rows() == newest.rows());
  REQUIRE(extrapolated->cols() == newest.cols());
  REQUIRE(extrapolated->norm() < 0.5 * newest.norm());
}

TEST_CASE("Dual extrapolation fails closed for unusable history",
          "[dual][extrapolation]")
{
  slope::detail::DualExtrapolator extrapolator;
  const Eigen::MatrixXd constant = Eigen::MatrixXd::Ones(3, 2);

  for (int iteration = 0; iteration < 6; ++iteration) {
    REQUIRE_FALSE(extrapolator.push(constant).has_value());
  }

  Eigen::MatrixXd non_finite = constant;
  non_finite(0, 0) = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_FALSE(extrapolator.push(non_finite).has_value());

  for (int iteration = 0; iteration < 5; ++iteration) {
    REQUIRE_FALSE(extrapolator.push(contractingPoint(iteration)).has_value());
  }
}

TEST_CASE("Reset discards dual extrapolation history", "[dual][extrapolation]")
{
  slope::detail::DualExtrapolator extrapolator;

  for (int iteration = 0; iteration < 6; ++iteration) {
    extrapolator.push(contractingPoint(iteration));
  }

  extrapolator.reset();

  for (int iteration = 0; iteration < 5; ++iteration) {
    REQUIRE_FALSE(extrapolator.push(contractingPoint(iteration)).has_value());
  }
  REQUIRE(extrapolator.push(contractingPoint(5)).has_value());
}

TEST_CASE("Dual extrapolation can be sampled without dropping history",
          "[dual][extrapolation]")
{
  slope::detail::DualExtrapolator extrapolator;

  for (int iteration = 0; iteration < 5; ++iteration) {
    extrapolator.push(contractingPoint(iteration));
  }

  REQUIRE_FALSE(extrapolator.push(contractingPoint(5), false).has_value());
  REQUIRE(extrapolator.push(contractingPoint(6)).has_value());
}

TEST_CASE("Dual extrapolation is limited to supported paths",
          "[dual][extrapolation]")
{
  using slope::detail::dualExtrapolationEnabled;
  using slope::detail::resolveSolverType;

  REQUIRE(resolveSolverType("auto") == "hybrid");
  REQUIRE(dualExtrapolationEnabled("logistic", "hybrid"));
  REQUIRE(dualExtrapolationEnabled("logistic", "pgd"));
  REQUIRE(dualExtrapolationEnabled("quadratic", "hybrid"));
  REQUIRE(dualExtrapolationEnabled("quadratic", "pgd"));

  REQUIRE_FALSE(dualExtrapolationEnabled("logistic", "fista"));
  REQUIRE_FALSE(dualExtrapolationEnabled("quadratic", "fista"));
  REQUIRE_FALSE(dualExtrapolationEnabled("poisson", "hybrid"));
  REQUIRE_FALSE(dualExtrapolationEnabled("multinomial", "pgd"));
}
