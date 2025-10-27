#include "slope/normalize.h"
#include "slope/slope.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

TEST_CASE("Sparse and dense methods agree", "[quadratic][sparse]")
{
  using namespace Catch::Matchers;

  int p = 3;
  Eigen::VectorXd beta(p);
  Eigen::VectorXd y(10);
  Eigen::MatrixXd x_dense(10, p);
  Eigen::VectorXd x_centers(p);
  Eigen::VectorXd x_centers_dense(p);
  Eigen::VectorXd x_centers_sparse(p);
  Eigen::VectorXd x_scales(p);
  Eigen::VectorXd x_scales_dense(p);
  Eigen::VectorXd x_scales_sparse(p);

  // clang-format off
  x_dense << 0.0,        0.13339576, 0.49361983,
             0.17769259, 0.66565742, 0.36972579,
             0.0,        0.0,        0.0,
             0.0,        0.0,        0.94280368,
             0.0,        0.0,        0.3499374,
             0.0,        0.22377115, 0.0,
             0.0,        0.96893287, 0.95858229,
             0.70486475, 0.60885162, 0.0,
             0.0,        0.92902639, 0.0,
             0.4978676,  0.0,        0.50022619;
  // clang-format on
  Eigen::SparseMatrix<double> x_sparse = x_dense.sparseView().eval();

  slope::Slope model;

  SECTION("Standardization for sparse and dense")
  {
    slope::computeCenters(x_centers_sparse, x_sparse, "mean");
    slope::computeScales(x_scales_sparse, x_sparse, "sd");

    slope::computeCenters(x_centers_dense, x_dense, "mean");
    slope::computeScales(x_scales_dense, x_dense, "sd");

    REQUIRE_THAT(x_centers_dense, VectorApproxEqual(x_centers_sparse));
    REQUIRE_THAT(x_scales_dense, VectorApproxEqual(x_scales_sparse));

    beta << 1, 2, -0.9;

    y = x_dense * beta;

    Eigen::ArrayXd alpha = Eigen::ArrayXd::Ones(1);
    Eigen::Array<double, 3, 1> lambda;
    lambda << 0.5, 0.5, 0.2;

    model.setIntercept(false);
    model.setModifyX(false);
    // model.setScreening("none");
    model.setNormalization("standardization");

    auto fit = model.path(x_sparse, y, alpha, lambda);
    auto coefs_sparse = fit.getCoefs();

    fit = model.path(x_dense, y, alpha, lambda);
    auto coefs_dense = fit.getCoefs();

    Eigen::VectorXd coef_sparse = coefs_sparse.front();
    Eigen::VectorXd coef_dense = coefs_dense.front();

    REQUIRE_THAT(coef_sparse, VectorApproxEqual(coef_dense, 1e-6));
  }

  SECTION("Manual centers and scales")
  {
    x_centers << 0, 0, 0;
    x_scales << 1, 1, 1;
    model.setCentering(x_centers);
    model.setScaling(x_scales);

    auto fit_manual_dense = model.fit(x_dense, y);
    auto fit_manual_sparse = model.fit(x_sparse, y);

    model.setNormalization("none");

    auto fit_auto_dense = model.fit(x_dense, y);
    auto fit_auto_sparse = model.fit(x_sparse, y);

    Eigen::VectorXd coef_manual_dense = fit_manual_dense.getCoefs();
    Eigen::VectorXd coef_manual_sparse = fit_manual_dense.getCoefs();

    REQUIRE_THAT(coef_manual_dense, VectorApproxEqual(coef_manual_sparse));
  }
}
