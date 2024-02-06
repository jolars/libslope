#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <slope/slope.h>

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
  Eigen::SparseMatrix<double> x_sparse = x_dense.sparseView();

  beta << 1, 2, -0.9;

  y = x_dense * beta;

  Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(1);
  Eigen::Array<double, 3, 1> lambda;
  lambda << 0.5, 0.5, 0.2;

  slope::Slope model;

  model.setIntercept(false);
  model.setStandardize(false);
  model.setPrintLevel(3);
  model.setPgdFreq(1);

  model.fit(x_sparse, y, alpha, lambda);
  auto coefs_sparse = model.getCoefs();

  model.fit(x_dense, y, alpha, lambda);
  auto coefs_dense = model.getCoefs();

  auto dual_gaps = model.getDualGaps();
  auto primals = model.getPrimals();

  Eigen::VectorXd coef_sparse = coefs_sparse.col(0);
  Eigen::VectorXd coef_dense = coefs_dense.col(0);

  REQUIRE_THAT(coef_sparse, VectorApproxEqual(coef_dense, 1e-4));
}
