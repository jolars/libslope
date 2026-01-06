#include "generate_data.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <limits>
#include <slope/slope.h>

TEST_CASE("NA/NaN/Inf validation in response", "[input_validation]")
{
  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd x(n, p);
  x.setRandom();

  slope::Slope model;
  model.setLoss("quadratic");

  SECTION("NaN in response")
  {
    Eigen::MatrixXd y(n, 1);
    y.setRandom();
    y(5, 0) = std::numeric_limits<double>::quiet_NaN();

    REQUIRE_THROWS_AS(model.path(x, y), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.path(x, y),
                        Catch::Matchers::ContainsSubstring("NaN") ||
                          Catch::Matchers::ContainsSubstring("Inf") ||
                          Catch::Matchers::ContainsSubstring("NA"));
  }

  SECTION("Positive Inf in response")
  {
    Eigen::MatrixXd y(n, 1);
    y.setRandom();
    y(3, 0) = std::numeric_limits<double>::infinity();

    REQUIRE_THROWS_AS(model.path(x, y), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.path(x, y),
                        Catch::Matchers::ContainsSubstring("NaN") ||
                          Catch::Matchers::ContainsSubstring("Inf") ||
                          Catch::Matchers::ContainsSubstring("NA"));
  }

  SECTION("Negative Inf in response")
  {
    Eigen::MatrixXd y(n, 1);
    y.setRandom();
    y(7, 0) = -std::numeric_limits<double>::infinity();

    REQUIRE_THROWS_AS(model.path(x, y), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.path(x, y),
                        Catch::Matchers::ContainsSubstring("NaN") ||
                          Catch::Matchers::ContainsSubstring("Inf") ||
                          Catch::Matchers::ContainsSubstring("NA"));
  }

  SECTION("Valid response (no NaN/Inf)")
  {
    Eigen::MatrixXd y(n, 1);
    y.setRandom();

    REQUIRE_NOTHROW(model.path(x, y));
  }
}

TEST_CASE("NA/NaN/Inf validation in dense features", "[input_validation]")
{
  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd y(n, 1);
  y.setRandom();

  slope::Slope model;
  model.setLoss("quadratic");

  SECTION("NaN in feature matrix")
  {
    Eigen::MatrixXd x(n, p);
    x.setRandom();
    x(5, 2) = std::numeric_limits<double>::quiet_NaN();

    REQUIRE_THROWS_AS(model.path(x, y), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.path(x, y),
                        Catch::Matchers::ContainsSubstring("NaN") ||
                          Catch::Matchers::ContainsSubstring("Inf") ||
                          Catch::Matchers::ContainsSubstring("NA"));
  }

  SECTION("Positive Inf in feature matrix")
  {
    Eigen::MatrixXd x(n, p);
    x.setRandom();
    x(3, 1) = std::numeric_limits<double>::infinity();

    REQUIRE_THROWS_AS(model.path(x, y), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.path(x, y),
                        Catch::Matchers::ContainsSubstring("NaN") ||
                          Catch::Matchers::ContainsSubstring("Inf") ||
                          Catch::Matchers::ContainsSubstring("NA"));
  }

  SECTION("Negative Inf in feature matrix")
  {
    Eigen::MatrixXd x(n, p);
    x.setRandom();
    x(1, 0) = -std::numeric_limits<double>::infinity();

    REQUIRE_THROWS_AS(model.path(x, y), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.path(x, y),
                        Catch::Matchers::ContainsSubstring("NaN") ||
                          Catch::Matchers::ContainsSubstring("Inf") ||
                          Catch::Matchers::ContainsSubstring("NA"));
  }

  SECTION("Valid dense features (no NaN/Inf)")
  {
    Eigen::MatrixXd x(n, p);
    x.setRandom();

    REQUIRE_NOTHROW(model.path(x, y));
  }
}

TEST_CASE("NA/NaN/Inf validation in sparse features", "[input_validation]")
{
  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd y(n, 1);
  y.setRandom();

  slope::Slope model;
  model.setLoss("quadratic");

  SECTION("NaN in sparse feature matrix")
  {
    Eigen::SparseMatrix<double> x(n, p);
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < p; ++j) {
        if ((i + j) % 3 == 0) {
          triplets.push_back(Eigen::Triplet<double>(i, j, i + j + 1.0));
        }
      }
    }
    triplets.push_back(
      Eigen::Triplet<double>(5, 2, std::numeric_limits<double>::quiet_NaN()));
    x.setFromTriplets(triplets.begin(), triplets.end());

    REQUIRE_THROWS_AS(model.path(x, y), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.path(x, y),
                        Catch::Matchers::ContainsSubstring("NaN") ||
                          Catch::Matchers::ContainsSubstring("Inf") ||
                          Catch::Matchers::ContainsSubstring("NA"));
  }

  SECTION("Inf in sparse feature matrix")
  {
    Eigen::SparseMatrix<double> x(n, p);
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < p; ++j) {
        if ((i + j) % 3 == 0) {
          triplets.push_back(Eigen::Triplet<double>(i, j, i + j + 1.0));
        }
      }
    }
    triplets.push_back(
      Eigen::Triplet<double>(3, 1, std::numeric_limits<double>::infinity()));
    x.setFromTriplets(triplets.begin(), triplets.end());

    REQUIRE_THROWS_AS(model.path(x, y), std::invalid_argument);
    REQUIRE_THROWS_WITH(model.path(x, y),
                        Catch::Matchers::ContainsSubstring("NaN") ||
                          Catch::Matchers::ContainsSubstring("Inf") ||
                          Catch::Matchers::ContainsSubstring("NA"));
  }

  SECTION("Valid sparse features (no NaN/Inf)")
  {
    Eigen::SparseMatrix<double> x(n, p);
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < p; ++j) {
        if ((i + j) % 3 == 0) {
          triplets.push_back(Eigen::Triplet<double>(i, j, i + j + 1.0));
        }
      }
    }
    x.setFromTriplets(triplets.begin(), triplets.end());

    REQUIRE_NOTHROW(model.path(x, y));
  }
}

TEST_CASE("Lambda sequence order validation", "[input_validation]")
{
  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd x(n, p);
  x.setRandom();
  Eigen::MatrixXd y(n, 1);
  y.setRandom();

  slope::Slope model;
  model.setLoss("quadratic");

  SECTION("Increasing lambda sequence")
  {
    Eigen::ArrayXd lambda(p);
    lambda << 0.1, 0.2, 0.3; // Increasing

    REQUIRE_THROWS_AS(model.path(x, y, Eigen::ArrayXd::Zero(0), lambda),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(model.path(x, y, Eigen::ArrayXd::Zero(0), lambda),
                        Catch::Matchers::ContainsSubstring("decreasing"));
  }

  SECTION("Non-monotonic lambda sequence")
  {
    Eigen::ArrayXd lambda(p);
    lambda << 0.2, 0.3, 0.1; // Non-monotonic

    REQUIRE_THROWS_AS(model.path(x, y, Eigen::ArrayXd::Zero(0), lambda),
                      std::invalid_argument);
    REQUIRE_THROWS_WITH(model.path(x, y, Eigen::ArrayXd::Zero(0), lambda),
                        Catch::Matchers::ContainsSubstring("decreasing"));
  }

  SECTION("Strictly decreasing lambda sequence (valid)")
  {
    Eigen::ArrayXd lambda(p);
    lambda << 0.3, 0.2, 0.1; // Decreasing

    REQUIRE_NOTHROW(model.path(x, y, Eigen::ArrayXd::Zero(0), lambda));
  }

  SECTION("Non-increasing lambda with equal values (valid)")
  {
    Eigen::ArrayXd lambda(p);
    lambda << 0.3, 0.3, 0.1; // Non-increasing (has equal values)

    REQUIRE_NOTHROW(model.path(x, y, Eigen::ArrayXd::Zero(0), lambda));
  }

  SECTION("All zeros lambda (valid)")
  {
    Eigen::ArrayXd lambda(p);
    lambda << 0.0, 0.0, 0.0; // All zeros

    REQUIRE_NOTHROW(model.path(x, y, Eigen::ArrayXd::Zero(0), lambda));
  }

  SECTION("All equal non-zero lambda (valid)")
  {
    Eigen::ArrayXd lambda(p);
    lambda << 0.5, 0.5, 0.5; // All equal

    REQUIRE_NOTHROW(model.path(x, y, Eigen::ArrayXd::Zero(0), lambda));
  }
}

TEST_CASE("Lambda validation with different loss types", "[input_validation]")
{
  const int n = 20;
  const int p = 3;

  Eigen::MatrixXd x(n, p);
  x.setRandom();

  SECTION("Logistic loss with increasing lambda")
  {
    Eigen::MatrixXd y(n, 1);
    y = (Eigen::VectorXd::Random(n).array() > 0).cast<double>();

    Eigen::ArrayXd lambda(p);
    lambda << 0.05, 0.1, 0.15; // Increasing

    slope::Slope model;
    model.setLoss("logistic");

    REQUIRE_THROWS_AS(model.path(x, y, Eigen::ArrayXd::Zero(0), lambda),
                      std::invalid_argument);
  }

  SECTION("Poisson loss with increasing lambda")
  {
    Eigen::MatrixXd y(n, 1);
    y = Eigen::VectorXd::Random(n).array().abs() * 10;

    Eigen::ArrayXd lambda(p);
    lambda << 0.05, 0.1, 0.15; // Increasing

    slope::Slope model;
    model.setLoss("poisson");

    REQUIRE_THROWS_AS(model.path(x, y, Eigen::ArrayXd::Zero(0), lambda),
                      std::invalid_argument);
  }
}
