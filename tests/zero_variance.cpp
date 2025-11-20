#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/cv.h>
#include <slope/slope.h>

TEST_CASE("Zero-variance columns", "[zero_variance]")
{
  using Catch::Matchers::WithinAbs;

  SECTION("Single zero-variance column")
  {
    // Create a matrix with one zero-variance column (all 1s)
    Eigen::MatrixXd x(10, 3);
    x << 1, 2, 1, 2, 3, 1, 3, 4, 1, 4, 5, 1, 5, 6, 1, 6, 7, 1, 7, 8, 1, 8, 9,
      1, 9, 10, 1, 10, 11, 1;

    Eigen::VectorXd y(10);
    y << 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5;

    slope::Slope model;

    REQUIRE_NOTHROW(model.fit(x, y));

    auto fit = model.fit(x, y);
    Eigen::VectorXd coefs = fit.getCoefs();

    // The coefficient for the zero-variance column should be zero
    REQUIRE_THAT(coefs(2), WithinAbs(0.0, 1e-10));

    // Other coefficients should be non-zero
    REQUIRE(std::abs(coefs(0)) > 1e-10);
    REQUIRE(std::abs(coefs(1)) > 1e-10);
  }

  SECTION("Multiple zero-variance columns")
  {
    // Create a matrix with two zero-variance columns
    Eigen::MatrixXd x(10, 4);
    x << 1, 2, 1, 5, 2, 3, 1, 5, 3, 4, 1, 5, 4, 5, 1, 5, 5, 6, 1, 5, 6, 7, 1,
      5, 7, 8, 1, 5, 8, 9, 1, 5, 9, 10, 1, 5, 10, 11, 1, 5;

    Eigen::VectorXd y(10);
    y << 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5;

    slope::Slope model;

    REQUIRE_NOTHROW(model.fit(x, y));

    auto fit = model.fit(x, y);
    Eigen::VectorXd coefs = fit.getCoefs();

    // The coefficients for zero-variance columns should be zero
    REQUIRE_THAT(coefs(2), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(coefs(3), WithinAbs(0.0, 1e-10));
  }

  SECTION("Zero-variance column with different centering and scaling")
  {
    Eigen::MatrixXd x(10, 3);
    x << 1, 2, 1, 2, 3, 1, 3, 4, 1, 4, 5, 1, 5, 6, 1, 6, 7, 1, 7, 8, 1, 8, 9,
      1, 9, 10, 1, 10, 11, 1;

    Eigen::VectorXd y(10);
    y << 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5;

    std::vector<std::string> centering_types = { "mean", "none", "min" };
    std::vector<std::string> scaling_types = { "sd", "l1", "l2", "max_abs",
                                               "range" };

    for (const auto& centering_type : centering_types) {
      for (const auto& scaling_type : scaling_types) {
        slope::Slope model;
        model.setCentering(centering_type);
        model.setScaling(scaling_type);

        REQUIRE_NOTHROW(model.fit(x, y));

        auto fit = model.fit(x, y);
        Eigen::VectorXd coefs = fit.getCoefs();

        // The coefficient for the zero-variance column should be zero
        REQUIRE_THAT(coefs(2), WithinAbs(0.0, 1e-10));
      }
    }
  }

  SECTION("Cross-validation with zero-variance column")
  {
    Eigen::MatrixXd x(40, 3);
    Eigen::VectorXd y(40);

    // Create data with zero-variance column
    for (int i = 0; i < 40; i++) {
      x(i, 0) = i + 1.0;
      x(i, 1) = (i + 1.0) * 2.0;
      x(i, 2) = 1.0; // Zero variance column
      y(i) = 0.5 * x(i, 0) + 0.3 * x(i, 1) + 0.1;
    }

    slope::Slope model;
    auto cv_config = slope::CvConfig();
    cv_config.n_folds = 5;
    cv_config.hyperparams["q"] = { 0.1 };

    REQUIRE_NOTHROW(crossValidate(model, x, y, cv_config));
  }

  SECTION("Sparse matrix with zero-variance column")
  {
    Eigen::MatrixXd x_dense(10, 3);
    x_dense << 1, 2, 1, 2, 3, 1, 3, 4, 1, 4, 5, 1, 5, 6, 1, 6, 7, 1, 7, 8, 1,
      8, 9, 1, 9, 10, 1, 10, 11, 1;

    Eigen::SparseMatrix<double> x = x_dense.sparseView();

    Eigen::VectorXd y(10);
    y << 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5;

    slope::Slope model;

    REQUIRE_NOTHROW(model.fit(x, y));

    auto fit = model.fit(x, y);
    Eigen::VectorXd coefs = fit.getCoefs();

    // The coefficient for the zero-variance column should be zero
    REQUIRE_THAT(coefs(2), WithinAbs(0.0, 1e-10));
  }

  SECTION("Near-zero variance column")
  {
    // Create a matrix with a near-zero variance column
    Eigen::MatrixXd x(10, 3);
    x << 1, 2, 1.0, 2, 3, 1.0, 3, 4, 1.0, 4, 5, 1.0, 5, 6, 1.0, 6, 7, 1.0, 7,
      8, 1.0, 8, 9, 1.0, 9, 10, 1.0, 10, 11, 1.0;

    // Add tiny variation to last column
    x(3, 2) = 1.0000001;
    x(7, 2) = 0.9999999;

    Eigen::VectorXd y(10);
    y << 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5;

    slope::Slope model;

    // Should not throw even with near-zero variance
    REQUIRE_NOTHROW(model.fit(x, y));
  }
}
