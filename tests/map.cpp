#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/losses/quadratic.h>
#include <slope/slope.h>
#include <slope/threads.h>

TEST_CASE("Quadratic with Eigen::Map", "[quadratic][map]")
{
  using namespace Catch::Matchers;

  // Create a data matrix and vector directly
  Eigen::MatrixXd x_data(5, 3);
  Eigen::VectorXd y_data(5);

  // Fill with some test values
  x_data << 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.2, 0.4, 0.6, 1.2, 2.4, 3.6, 0.8,
    1.6, 2.4;

  // Create a simple relationship
  Eigen::Vector3d beta(1.5, -0.5, 0.25);
  y_data = x_data * beta;

  // Create maps to the data
  Eigen::Map<Eigen::MatrixXd> x_map(
    x_data.data(), x_data.rows(), x_data.cols());

  SECTION("Map vs direct matrix")
  {
    slope::Slope model;
    model.setNormalization("standardization");
    model.setIntercept(false);
    model.setTol(1e-8);
    model.setLambdaType("bh");

    double alpha = 0.01;

    // Fit with direct matrix
    auto fit_direct = model.fit(x_data, y_data, alpha);

    // Fit with mapped matrix
    auto fit_map = model.fit(x_map, y_data, alpha);

    Eigen::VectorXd coefs_direct = fit_direct.getCoefs();
    Eigen::VectorXd coefs_map = fit_direct.getCoefs();

    // Check that coefficients are the same
    REQUIRE_THAT(coefs_direct, VectorApproxEqual(coefs_map, 1e-8));
  }

  SECTION("Sparse map vs direct matrix")
  {
    slope::Slope model;

    model.setNormalization("standardization");
    model.setIntercept(false);
    model.setTol(1e-8);
    model.setLambdaType("bh");

    double alpha = 0.01;

    // Create a sparse matrix from the dense matrix
    Eigen::SparseMatrix<double> x_sparse = x_data.sparseView();

    Eigen::Map<Eigen::SparseMatrix<double>> x_sparse_map(
      x_sparse.rows(),
      x_sparse.cols(),
      x_sparse.nonZeros(),
      x_sparse.outerIndexPtr(),
      x_sparse.innerIndexPtr(),
      x_sparse.valuePtr());

    // Fit with direct matrix
    auto fit_direct = model.fit(x_sparse, y_data, alpha);

    // Fit with mapped matrix
    auto fit_map = model.fit(x_sparse_map, y_data, alpha);

    Eigen::VectorXd coefs_direct = fit_direct.getCoefs();
    Eigen::VectorXd coefs_map = fit_direct.getCoefs();

    // Check that coefficients are the same
    REQUIRE_THAT(coefs_direct, VectorApproxEqual(coefs_map, 1e-8));
  }

  SECTION("Path with mapped matrix")
  {
    slope::Slope model;
    model.setPathLength(3);
    model.setNormalization("none");

    Eigen::ArrayXd alpha(3);
    alpha << 1e-2, 1e-4, 1e-6;

    // Generate a path using mapped matrix
    auto path_map = model.path(x_map, y_data, alpha);
    auto path_direct = model.path(x_data, y_data, alpha);

    Eigen::VectorXd coefs_map = path_map(2).getCoefs();
    Eigen::VectorXd coefs_direct = path_direct(2).getCoefs();

    // Check that the last model in the path is close to the true coefficients
    REQUIRE_THAT(coefs_map, VectorApproxEqual(coefs_direct, 1e-6));
  }

  SECTION("Alpha estimation with mapped matrix")
  {
    slope::Slope model;
    model.setAlphaType("estimate");

    REQUIRE_NOTHROW(model.fit(x_map, y_data));
  }
}
