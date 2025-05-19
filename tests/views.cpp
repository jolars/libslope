#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/losses/quadratic.h>
#include <slope/slope.h>

TEST_CASE("Views", "[views][quadratic]")
{
  using namespace Catch::Matchers;

  slope::Slope model;

  int n = 10;
  int p = 3;

  Eigen::MatrixXd x(n, p);
  Eigen::VectorXd beta(p);
  Eigen::VectorXd y(n);

  // clang-format off
  x <<  0.288,  -0.0452,  0.880,
        0.788,   0.576,  -0.305,
        1.510,   0.390,  -0.621,
       -2.210,  -1.120,  -0.0449,
       -0.0162,  0.944,   0.821,
        0.594,   0.919,   0.782,
        0.0746, -1.990,   0.620,
       -0.0561, -0.156,  -1.470,
       -0.478,   0.418,   1.360,
       -0.103,   0.388,  -0.0538;
  // clang-format on

  std::vector<int> subset = {
    0, 1, 2, 3, 4,
  };

  auto x_view = x(subset, Eigen::all);

  Eigen::VectorXd y_subset = y(subset);
  Eigen::MatrixXd x_subset(subset.size(), p);

  // clang-format off
  x_subset <<  0.288,  -0.0452,  0.880,
               0.788,   0.576,  -0.305,
               1.510,   0.390,  -0.621,
              -2.210,  -1.120,  -0.0449,
              -0.0162,  0.944,   0.821;
  // clang-format on

  auto fit_view = model.fit(x_view, y_subset);
  auto fit_subset = model.fit(x_subset, y_subset);

  Eigen::VectorXd coef_view = fit_view.getCoefs();
  Eigen::VectorXd coef_subset = fit_view.getCoefs();

  REQUIRE_THAT(coef_view, VectorApproxEqual(coef_subset));
}
