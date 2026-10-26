#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>
#include <slope/slope.h>

TEST_CASE("Weighted loss gradients match finite differences", "[weights]")
{
  Eigen::VectorXd raw_y(5), weights(5);
  weights << 0, 2, 1, 3, 1;
  for (const std::string family :
       { "quadratic", "logistic", "poisson", "multinomial" }) {
    DYNAMIC_SECTION(family)
    {
      raw_y << 0, 1, 0, 1, 0;
      if (family == "multinomial") {
        raw_y << 0, 1, 2, 0, 1;
      }
      auto loss = slope::setupLoss(family);
      Eigen::MatrixXd y = loss->preprocessResponse(raw_y);
      Eigen::MatrixXd eta = Eigen::MatrixXd::Constant(y.rows(), y.cols(), 0.3);
      const auto residual = loss->residual(eta, y);
      const double epsilon = 1e-6;
      for (int i = 0; i < eta.rows(); ++i) {
        for (int k = 0; k < eta.cols(); ++k) {
          eta(i, k) += epsilon;
          const double upper = loss->loss(eta, y, weights);
          eta(i, k) -= 2 * epsilon;
          const double lower = loss->loss(eta, y, weights);
          eta(i, k) += epsilon;
          REQUIRE_THAT((upper - lower) / (2 * epsilon),
                       Catch::Matchers::WithinAbs(
                         weights(i) * residual(i, k) / weights.sum(), 1e-8));
        }
      }
    }
  }
}

TEST_CASE("Observation weights reproduce repeated observations", "[weights]")
{
  Eigen::MatrixXd x(10, 3);
  x << -2, 1, 0, -1, 0, 2, 0, 2, -1, 1, -1, 1, 2, 0, -2, 3, 1, 1, -3, 2, 2, 0.5,
    -2, 0, 1.5, 1, 3, -0.5, 0, -1;
  Eigen::VectorXd weights(10);
  weights << 1, 2, 0, 3, 1, 2, 1, 1, 2, 1;
  for (const std::string family :
       { "quadratic", "logistic", "poisson", "multinomial" }) {
    DYNAMIC_SECTION(family)
    {
      Eigen::MatrixXd y(10, 1);
      if (family == "quadratic") {
        y << -1.5, 1, 4, 1.5, 2, 3, -2, 0, 1, -1;
      } else if (family == "logistic") {
        y << 0, 0, 1, 1, 1, 1, 0, 0, 1, 0;
      } else if (family == "poisson") {
        y << 0, 1, 3, 2, 4, 2, 0, 1, 3, 0;
      } else {
        y << 0, 1, 2, 2, 1, 0, 0, 2, 1, 2;
      }
      Eigen::MatrixXd repeated_x(static_cast<int>(weights.sum()), x.cols());
      Eigen::MatrixXd repeated_y(static_cast<int>(weights.sum()), 1);
      int row = 0;
      for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < weights(i); ++j, ++row) {
          repeated_x.row(row) = x.row(i);
          repeated_y.row(row) = y.row(i);
        }
      }
      for (const std::string solver : { "pgd", "fista", "hybrid" }) {
        DYNAMIC_SECTION(solver)
        {
          slope::Slope model;
          model.setLoss(family);
          model.setSolver(solver);
          model.setTol(1e-8);
          model.setDiagnostics(true);
          auto weighted =
            model.fit(x, y, 0.08, {}, slope::defaultInterruptChecker, weights);
          REQUIRE(!weighted.getPrimals().empty());
          REQUIRE(weighted.getPrimals().size() == weighted.getDuals().size());
          for (size_t i = 0; i < weighted.getPrimals().size(); ++i) {
            REQUIRE(weighted.getDuals()[i] <= weighted.getPrimals()[i] + 1e-7);
          }
          REQUIRE(weighted.getPrimals().back() - weighted.getDuals().back() <
                  1e-6);
          auto repeated = model.fit(repeated_x, repeated_y, 0.08);
          REQUIRE((Eigen::MatrixXd(weighted.getCoefs()) -
                   Eigen::MatrixXd(repeated.getCoefs()))
                    .norm() < 2e-4);
          REQUIRE((weighted.getIntercepts() - repeated.getIntercepts()).norm() <
                  2e-4);
          REQUIRE_THAT(
            weighted.getDeviance(),
            Catch::Matchers::WithinAbs(repeated.getDeviance(), 1e-5));
          auto scaled = model.fit(
            x, y, 0.08, {}, slope::defaultInterruptChecker, 17 * weights);
          REQUIRE((Eigen::MatrixXd(weighted.getCoefs()) -
                   Eigen::MatrixXd(scaled.getCoefs()))
                    .norm() < 1e-7);
          Eigen::SparseMatrix<double> sparse_x = x.sparseView();
          auto sparse = model.fit(
            sparse_x, y, 0.08, {}, slope::defaultInterruptChecker, weights);
          REQUIRE((Eigen::MatrixXd(weighted.getCoefs()) -
                   Eigen::MatrixXd(sparse.getCoefs()))
                    .norm() < 2e-4);
        }
      }
    }
  }
}

TEST_CASE("Observation weights validate before modifying predictors",
          "[weights]")
{
  Eigen::MatrixXd x = Eigen::MatrixXd::Random(6, 2);
  const Eigen::MatrixXd original = x;
  Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(6, 1, 6);
  slope::Slope model;
  model.setModifyX(true);
  for (Eigen::VectorXd invalid :
       { Eigen::VectorXd::Ones(5).eval(),
         Eigen::VectorXd::Zero(6).eval(),
         Eigen::VectorXd::Constant(6, -1).eval(),
         Eigen::VectorXd::Constant(6, std::numeric_limits<double>::quiet_NaN())
           .eval(),
         Eigen::VectorXd::Constant(6, std::numeric_limits<double>::infinity())
           .eval() }) {
    REQUIRE_THROWS_AS(
      model.fit(x, y, 0.1, {}, slope::defaultInterruptChecker, invalid),
      std::invalid_argument);
    REQUIRE(x == original);
  }
}

TEST_CASE("Weighted fits preserve uniform weights and ignore zero rows",
          "[weights]")
{
  Eigen::MatrixXd x(8, 2);
  x << -3, 2, -2, -1, -1, 0, 0, 2, 1, -1, 2, 0, 3, 1, 100, -100;
  Eigen::MatrixXd y(8, 2);
  y << -2, 1, -1, 3, 0, -1, 1, 2, 2, 0, 4, -2, 3, 1, 1000, -1000;
  Eigen::VectorXd weights = Eigen::VectorXd::Ones(8);
  slope::Slope model;
  model.setTol(1e-9);
  const auto legacy = model.fit(x, y, 0.1);
  const auto uniform =
    model.fit(x, y, 0.1, {}, slope::defaultInterruptChecker, 9 * weights);
  REQUIRE(
    (Eigen::MatrixXd(legacy.getCoefs()) - Eigen::MatrixXd(uniform.getCoefs()))
      .norm() < 1e-10);
  REQUIRE((legacy.getIntercepts() - uniform.getIntercepts()).norm() < 1e-10);
  weights(7) = 0;
  Eigen::MatrixXd trimmed_x = x.topRows(7);
  Eigen::MatrixXd trimmed_y = y.topRows(7);
  const auto weighted =
    model.fit(x, y, 0.1, {}, slope::defaultInterruptChecker, weights);
  const auto trimmed = model.fit(trimmed_x, trimmed_y, 0.1);
  REQUIRE(
    (Eigen::MatrixXd(weighted.getCoefs()) - Eigen::MatrixXd(trimmed.getCoefs()))
      .norm() < 1e-6);
  Eigen::Map<Eigen::MatrixXd> mapped_x(x.data(), x.rows(), x.cols());
  const auto mapped =
    model.fit(mapped_x, y, 0.1, {}, slope::defaultInterruptChecker, weights);
  REQUIRE(
    (Eigen::MatrixXd(weighted.getCoefs()) - Eigen::MatrixXd(mapped.getCoefs()))
      .norm() < 1e-8);
  for (const std::string screening : { "none", "strong" }) {
    model.setScreening(screening);
    auto view = x(Eigen::seq(0, 7), Eigen::all);
    const auto viewed =
      model.fit(view, y, 0.1, {}, slope::defaultInterruptChecker, weights);
    REQUIRE((Eigen::MatrixXd(weighted.getCoefs()) -
             Eigen::MatrixXd(viewed.getCoefs()))
              .norm() < 1e-6);
  }
}

TEST_CASE("Weighted calibration restrictions are explicit", "[weights]")
{
  Eigen::MatrixXd x = Eigen::MatrixXd::Random(40, 2);
  Eigen::MatrixXd y = Eigen::MatrixXd::Random(40, 1);
  Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(40, 1, 2);
  slope::Slope model;
  model.setLambdaType("gaussian");
  REQUIRE_THROWS_AS(
    model.fit(x, y, 0.1, {}, slope::defaultInterruptChecker, weights),
    std::invalid_argument);
  REQUIRE_NOTHROW(model.fit(x,
                            y,
                            0.1,
                            Eigen::ArrayXd::Ones(2),
                            slope::defaultInterruptChecker,
                            weights));
  REQUIRE_NOTHROW(model.fit(x,
                            y,
                            0.1,
                            {},
                            slope::defaultInterruptChecker,
                            Eigen::VectorXd::Constant(40, 2)));
  model.setLambdaType("bh");
  model.setAlphaType("estimate");
  REQUIRE_THROWS_AS(
    model.path(x, y, {}, {}, slope::defaultInterruptChecker, weights),
    std::invalid_argument);
  REQUIRE_THROWS_AS(
    model.estimateAlpha(x, y, slope::defaultInterruptChecker, weights),
    std::invalid_argument);
}

TEST_CASE("Weight normalization handles extreme scales and single-row support",
          "[weights]")
{
  Eigen::VectorXd weights(3);
  weights << 1e308, 1e308, 0;
  const auto normalized =
    slope::detail::normalizeObservationWeights(weights, 3);
  REQUIRE(normalized.allFinite());
  REQUIRE_THAT(normalized.sum(), Catch::Matchers::WithinAbs(3, 1e-14));
  REQUIRE(normalized(2) == 0);
  REQUIRE_THROWS_AS(slope::detail::normalizeObservationWeights({}, 0),
                    std::invalid_argument);

  Eigen::MatrixXd x(3, 1);
  x << -1, 2, 1e200;
  Eigen::VectorXd y(3);
  y << 4, 5, 1e200;
  weights << 0, 2, 0;
  slope::Slope model;
  const auto fit =
    model.fit(x, y, 0.1, {}, slope::defaultInterruptChecker, weights);
  REQUIRE(fit.getCoefs().norm() == 0);
  REQUIRE_THAT(fit.getIntercepts()(0), Catch::Matchers::WithinAbs(5, 1e-12));
}

TEST_CASE("Weighted relaxation solves weighted least squares",
          "[weights][relax]")
{
  Eigen::MatrixXd x(6, 1);
  x << -2, -1, 0, 1, 2, 3;
  Eigen::VectorXd y(6), weights(6);
  y << -1, 0, 2, 2, 5, 3;
  weights << 1, 3, 0, 2, 1, 4;
  const double center_x = weights.dot(x.col(0)) / weights.sum();
  const double center_y = weights.dot(y) / weights.sum();
  const double expected_beta =
    (weights.array() * (x.col(0).array() - center_x) * (y.array() - center_y))
      .sum() /
    (weights.array() * (x.col(0).array() - center_x).square()).sum();
  const double expected_intercept = center_y - expected_beta * center_x;
  slope::Slope model;
  model.setTol(1e-10);
  model.setRelaxTol(1e-9);
  const auto fit =
    model.fit(x, y, 0.01, {}, slope::defaultInterruptChecker, weights);
  const auto relaxed = model.relax(fit, x, y, 0.0, {}, {}, weights);
  REQUIRE_THAT(relaxed.getCoefs().coeff(0, 0),
               Catch::Matchers::WithinAbs(expected_beta, 1e-6));
  REQUIRE_THAT(relaxed.getIntercepts()(0),
               Catch::Matchers::WithinAbs(expected_intercept, 1e-6));
  const auto mixed = model.relax(fit, x, y, 0.5, {}, {}, weights);
  REQUIRE_THAT(mixed.getIntercepts()(0),
               Catch::Matchers::WithinAbs(
                 0.5 * (expected_intercept + fit.getIntercepts()(0)), 1e-6));
}
