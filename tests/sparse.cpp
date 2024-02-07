#include "../src/slope/slope.h"
#include "../src/slope/standardize.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

TEST_CASE("Sparse and dense methods agree", "[gaussian][sparse]")
{
  using namespace Catch::Matchers;

  Eigen::Vector<double, 3> beta;
  Eigen::Vector<double, 10> y;
  Eigen::Matrix<double, 10, 3> x_dense;

  x_dense << 0.0, 0.13339576, 0.49361983, 0.17769259, 0.66565742, 0.36972579,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.94280368, 0.0, 0.0, 0.3499374, 0.0, 0.22377115,
    0.0, 0.0, 0.96893287, 0.95858229, 0.70486475, 0.60885162, 0.0, 0.0,
    0.92902639, 0.0, 0.4978676, 0.0, 0.50022619;
  Eigen::SparseMatrix<double> x_sparse = x_dense.sparseView().eval();

  auto [x_centers_sparse, x_scales_sparse] =
    slope::computeCentersAndScales(x_sparse, true);
  auto [x_centers_dense, x_scales_dense] =
    slope::computeCentersAndScales(x_dense, true);

  REQUIRE_THAT(x_centers_dense, VectorApproxEqual(x_centers_sparse));
  REQUIRE_THAT(x_scales_dense, VectorApproxEqual(x_scales_sparse));

  beta << 1, 2, -0.9;

  y = x_dense * beta;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Ones(1);
  Eigen::Array<double, 3, 1> lambda;
  lambda << 0.5, 0.5, 0.2;

  slope::Slope model;

  model.setIntercept(false);
  model.setStandardize(true);

  model.fit(x_sparse, y, alpha, lambda);
  Eigen::VectorXd coefs_sparse = model.getCoefs();

  model.fit(x_dense, y, alpha, lambda);
  Eigen::VectorXd coefs_dense = model.getCoefs();

  Eigen::VectorXd coef_sparse = coefs_sparse.col(0);
  Eigen::VectorXd coef_dense = coefs_dense.col(0);

  REQUIRE_THAT(coef_sparse, VectorApproxEqual(coef_dense, 1e-6));
}
