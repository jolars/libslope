#include "generate_data.hpp"
#include "load_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/constants.h>
#include <slope/logger.h>
#include <slope/losses/quadratic.h>
#include <slope/ols.h>
#include <slope/slope.h>
#include <slope/threads.h>

TEST_CASE("Weighted quadratic loss and dual certificates",
          "[quadratic][weights]")
{
  using namespace Catch::Matchers;
  slope::Quadratic loss;
  Eigen::Vector4d eta, y, weights;
  eta << 2.0, -1.0, 4.0, 1e200;
  y << 1.0, 2.0, 0.0, -1e200;
  weights << 1.0, 2.0, 3.0, 0.0;

  REQUIRE_THAT(loss.loss(eta, y, weights), WithinAbs(67.0 / 12.0, 1e-12));
  Eigen::MatrixXd theta = loss.dualPoint(eta, y, true, weights);
  REQUIRE(theta.allFinite());
  REQUIRE_THAT(weights.dot(theta.col(0)), WithinAbs(0.0, 1e-12));
  REQUIRE(theta(3, 0) == 0.0);
  REQUIRE(std::isfinite(loss.dual(theta, y, weights)));

  Eigen::MatrixXd working_weights, working_response;
  loss.updateWeightsAndWorkingResponse(
    working_weights, working_response, eta, y, weights);
  REQUIRE_THAT(working_weights(0, 0), WithinAbs(2.0 / 3.0, 1e-12));
  REQUIRE(working_weights(3, 0) == 0.0);
  REQUIRE(working_response.allFinite());
  const Eigen::MatrixXd first_weights = working_weights;
  loss.updateWeightsAndWorkingResponse(
    working_weights, working_response, eta, y, weights);
  REQUIRE((working_weights - first_weights).isZero());
}

TEST_CASE("Weighted quadratic OLS certificates use weighted least squares",
          "[quadratic][weights][dual]")
{
  using namespace Catch::Matchers;
  Eigen::MatrixXd x(5, 2), y(5, 2);
  x << 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, -1.0, 2.0, 3.0, -1.0;
  y << 1.0, 2.0, 2.0, -1.0, 0.0, 4.0, 3.0, 1.0, -2.0, 0.0;
  Eigen::VectorXd weights(5);
  weights << 1.0, 3.0, 2.0, 0.0, 4.0;
  const Eigen::VectorXd centers = Eigen::VectorXd::Constant(2, 0.5);
  const Eigen::VectorXd scales = Eigen::VectorXd::Constant(2, 2.0);
  Eigen::SparseMatrix<double> sparse_x = x.sparseView();

  for (const bool intercept : { false, true }) {
    const Eigen::MatrixXd theta = slope::detail::quadraticOlsDualPoint(
      x, y, centers, scales, slope::JitNormalization::Both, intercept, weights);
    const Eigen::MatrixXd sparse_theta =
      slope::detail::quadraticOlsDualPoint(sparse_x,
                                           y,
                                           centers,
                                           scales,
                                           slope::JitNormalization::Both,
                                           intercept,
                                           weights);
    REQUIRE((theta - sparse_theta).norm() < 1e-10);
    Eigen::MatrixXd effective = x.rowwise() - centers.transpose();
    effective.array().rowwise() /= scales.transpose().array();
    Eigen::MatrixXd weighted_theta = theta.array().colwise() * weights.array();
    REQUIRE((effective.transpose() * weighted_theta).norm() < 1e-10);
    if (intercept) {
      REQUIRE(weighted_theta.colwise().sum().norm() < 1e-10);
    }
    REQUIRE(theta.row(3).isZero());
  }
}

TEST_CASE("Weighted dual dispatch preserves legacy loss overrides",
          "[quadratic][weights][dual]")
{
  class LegacyQuadratic : public slope::Quadratic
  {
  public:
    Eigen::MatrixXd dualPoint(const Eigen::MatrixXd& eta,
                              const Eigen::MatrixXd&,
                              bool) override
    {
      return Eigen::MatrixXd::Constant(eta.rows(), eta.cols(), 42.0);
    }

    double deviance(const Eigen::MatrixXd&, const Eigen::MatrixXd&) override
    {
      return 17.0;
    }
  };

  LegacyQuadratic legacy;
  slope::Loss& loss = legacy;
  const Eigen::Vector2d values = Eigen::Vector2d::Zero();
  const Eigen::MatrixXd theta =
    loss.dualPoint(values, values, true, Eigen::Vector2d::Constant(3.0));
  REQUIRE((theta.array() == 42.0).all());
  REQUIRE(loss.deviance(values, values, Eigen::Vector2d::Ones()) == 17.0);
}

TEST_CASE("Weighted OLS certificates mask ignored rows before scaling",
          "[quadratic][weights][dual]")
{
  Eigen::MatrixXd x(4, 1), y(4, 1);
  x << 1e-150, 2e-150, 4e-150, 1e200;
  y << 1.0, -2.0, 3.0, 1e200;
  const Eigen::Vector4d weights(1.0, 2.0, 3.0, 0.0);
  const Eigen::VectorXd centers = Eigen::VectorXd::Constant(1, 2e-150);
  const Eigen::VectorXd scales = Eigen::VectorXd::Constant(1, 1e-150);
  Eigen::SparseMatrix<double> sparse_x = x.sparseView();
  Eigen::MatrixXd safe_x = x;
  safe_x.row(3).setZero();

  for (const bool intercept : { false, true }) {
    for (const auto normalization :
         { slope::JitNormalization::Both, slope::JitNormalization::Scale }) {
      const Eigen::MatrixXd expected = slope::detail::quadraticOlsDualPoint(
        safe_x, y, centers, scales, normalization, intercept, weights);
      const Eigen::MatrixXd dense = slope::detail::quadraticOlsDualPoint(
        x, y, centers, scales, normalization, intercept, weights);
      const Eigen::MatrixXd sparse = slope::detail::quadraticOlsDualPoint(
        sparse_x, y, centers, scales, normalization, intercept, weights);
      REQUIRE(dense.allFinite());
      REQUIRE(sparse.allFinite());
      REQUIRE(dense.row(3).isZero());
      REQUIRE(sparse.row(3).isZero());
      REQUIRE((dense - expected).norm() < 1e-12);
      REQUIRE((sparse - expected).norm() < 1e-12);
    }
  }
}

TEST_CASE("Guassian, simple design", "[quadratic]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x(3, 2);
  Eigen::Vector2d beta;
  Eigen::VectorXd y(3);

  // clang-format off
  x << 1.1, 2.3,
       0.2, 1.5,
       0.5, 0.2;
  // clang-format on
  beta << 1, 2;

  y = x * beta;

  double alpha = 0;
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Ones(2);

  slope::Slope model;
  slope::SlopeFit fit;

  model.setIntercept(false);
  model.setNormalization("none");
  model.setDiagnostics(true);

  fit = model.fit(x, y, alpha, lambda);

  Eigen::VectorXd coef = fit.getCoefs();
  Eigen::VectorXd coef_scaled = fit.getCoefs(false);
  auto dual_gaps = fit.getGaps();
  auto primals = fit.getPrimals();

  REQUIRE(coef.rows() == coef_scaled.rows());
  REQUIRE(coef.cols() == coef_scaled.cols());

  REQUIRE_THAT(coef, VectorApproxEqual(beta, 1e-4));
  REQUIRE_THAT(coef, VectorApproxEqual(coef_scaled, 1e-4));

  double gap = dual_gaps.back();
  double primal = primals.back();

  REQUIRE(gap <= (primal + 1e-10) * 1e-4);
}

TEST_CASE("Quadratic, X is identity", "[quadratic]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x = Eigen::MatrixXd::Identity(4, 4);
  Eigen::VectorXd y(4);
  y << 8, 6, 4, 2;

  double alpha = 1.0;
  Eigen::ArrayXd lambda(4);
  lambda << 1, 0.75, 0.5, 0.25;

  slope::Slope model;
  model.setIntercept(false);
  model.setNormalization("none");
  model.setDiagnostics(true);
  model.setTol(1e-5);
  auto fit = model.fit(x, y, alpha, lambda);

  double gap = fit.getGaps().back();
  double primal = fit.getPrimals().back();

  Eigen::VectorXd coefs = fit.getCoefs();

  std::array<double, 4> expected = { 4.0, 3.0, 2.0, 1.0 };

  REQUIRE_THAT(coefs, VectorApproxEqual(expected, 1e-3));
  REQUIRE(gap < primal * 1e-4);
}

TEST_CASE("Quadratic, automatic lambda and alpha", "[quadratic]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x(3, 2);
  // clang-format off
  x << 1, 2,
       0, 1,
       1, 0;
  // clang-format on
  Eigen::Vector3d y;
  y << -1, 6, 2;

  slope::Slope model;

  REQUIRE_NOTHROW(model.path(x, y));
}

TEST_CASE("Quadratic, various models", "[quadratic]")
{
  using namespace Catch::Matchers;

  const int n = 10;
  const int p = 3;

  Eigen::MatrixXd x(n, p);
  Eigen::VectorXd beta(p);

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

  // Fixed coefficients beta
  beta << 1, -1, 0.2;

  // Compute linear predictor
  Eigen::VectorXd y = x * beta;

  Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(3);

  double alpha = 0.05;

  lambda << 3.0, 2.0, 2.0;

  Eigen::Vector3d coef_target;

  slope::Slope model;

  model.setTol(1e-8);
  model.setLoss("quadratic");
  model.setDiagnostics(true);

  SECTION("No intercept, no standardization")
  {
    model.setNormalization("none");
    model.setIntercept(false);

    coef_target << 0.6864545, -0.6864545, 0.0000000;

    model.setSolver("fista");
    auto fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();

    model.setSolver("hybrid");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_hybrid = fit.getCoefs();

    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-4));

    auto dual_gaps = fit.getGaps();

    REQUIRE(dual_gaps.back() >= -GAP_TOLERANCE);
    REQUIRE(dual_gaps.back() <= 1e-4);
  }

  SECTION("No intercept, with standardization")
  {
    model.setNormalization("standardization");
    model.setIntercept(false);
    // model.setModifyX(true);

    coef_target << 0.700657772, -0.730587233, 0.008997323;

    model.setSolver("pgd");
    auto fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();

    model.setSolver("hybrid");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_hybrid = fit.getCoefs();

    REQUIRE_THAT(coefs_hybrid, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
  }

  SECTION("With intercept, with standardization")
  {
    model.setNormalization("standardization");
    model.setIntercept(true);

    coef_target << 0.700657772, -0.730587234, 0.008997323;
    std::vector<double> intercept_target = { 0.040584733 };

    model.setSolver("hybrid");
    auto fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs = fit.getCoefs();
    Eigen::VectorXd intercept = fit.getIntercepts();

    model.setSolver("pgd");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();
    Eigen::VectorXd intercept_pgd = fit.getIntercepts();

    REQUIRE_THAT(intercept, VectorApproxEqual(intercept_target, 1e-3));
    REQUIRE_THAT(intercept_pgd, VectorApproxEqual(intercept_target, 1e-3));

    REQUIRE_THAT(coefs, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));
  }

  SECTION("With intercept, no standardization")
  {
    model.setNormalization("none");
    model.setIntercept(true);
    model.setMaxIterations(1e5);

    coef_target << 0.68614138, -0.68614138, 0.00000000;

    model.setSolver("hybrid");
    auto fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs = fit.getCoefs();
    double intercept = fit.getIntercepts()[0];

    model.setSolver("pgd");
    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coefs_pgd = fit.getCoefs();
    double intercept_pgd = fit.getIntercepts()[0];

    REQUIRE_THAT(intercept, WithinAbs(0.04148455, 1e-3));
    REQUIRE_THAT(intercept_pgd, WithinAbs(0.04148455, 1e-3));

    REQUIRE_THAT(coefs, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE_THAT(coefs_pgd, VectorApproxEqual(coef_target, 1e-4));

    auto time = fit.getTime();

    REQUIRE_THAT(time, VectorMonotonic(true, false));

    auto passes = fit.getPasses();
    REQUIRE(passes > 0);
    REQUIRE(passes < 1e5);
  }

  SECTION("mtcars data")
  {
    Eigen::MatrixXd x(32, 1);
    Eigen::MatrixXd y(32, 1);

    x << 21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4,
      17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3,
      19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4;
    y << 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
      0, 1, 0, 1, 0, 0, 0, 1;

    slope::Slope model;

    model.setPathLength(1);

    REQUIRE_NOTHROW(model.path(x, y));
  }
}

TEST_CASE("Gaussian parallel", "[quadratic]")
{
  using namespace Catch::Matchers;

  auto data = generateData(1000, 20);

  slope::Slope model;

  slope::Threads::set(1);
  auto fit_par = model.fit(data.x, data.y);

  slope::Threads::set(2);
  auto fit_seq = model.fit(data.x, data.y);

  Eigen::VectorXd coefs_par = fit_par.getCoefs();
  Eigen::VectorXd coefs_seq = fit_seq.getCoefs();

  REQUIRE_THAT(coefs_par, VectorApproxEqual(coefs_seq));
}

TEST_CASE("Gaussian predictions", "[gaussian][predict]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x(3, 2);
  Eigen::VectorXd beta(2);
  Eigen::VectorXd eta(3);

  // clang-format off
  x << 1.1, 2.3,
       0.2, 1.5,
       0.5, 0.2;
  // clang-format on
  beta << 1, 2;

  eta = x * beta;

  slope::Quadratic loss;

  auto pred = loss.predict(eta);

  REQUIRE_THAT(pred.reshaped(), VectorApproxEqual(eta));
}

TEST_CASE("Cluster updating", "[quadratic][clusters]")
{
  using namespace Catch::Matchers;

  auto data = generateData(100, 2000);

  slope::Slope model;

  model.setSolver("hybrid");

  model.setUpdateClusters(true);
  auto fit_updates = model.fit(data.x, data.y);

  model.setUpdateClusters(false);
  auto fit_wo_updates = model.fit(data.x, data.y);

  Eigen::VectorXd coefs_updates = fit_updates.getCoefs();
  Eigen::VectorXd coefs_wo_updates = fit_wo_updates.getCoefs();

  REQUIRE_THAT(coefs_updates, VectorApproxEqual(coefs_wo_updates));
}

TEST_CASE("Low regularization", "[quadratic]")
{
  using namespace Catch::Matchers;

  Eigen::MatrixXd x(3, 2);
  Eigen::Vector2d beta;
  Eigen::VectorXd y(3);

  // clang-format off
  x << 1.1, 2.3,
       0.2, 1.5,
       0.5, 0.2;
  // clang-format on
  beta << 1, 2;

  y = x * beta;

  double alpha = 1e-20;
  Eigen::ArrayXd lambda = Eigen::ArrayXd::Ones(2);

  slope::Slope model;
  slope::SlopeFit fit;

  model.setIntercept(false);
  model.setNormalization("none");
  model.setDiagnostics(true);

  fit = model.fit(x, y, alpha, lambda);

  Eigen::VectorXd coef = fit.getCoefs();
  Eigen::VectorXd coef_scaled = fit.getCoefs(false);
  REQUIRE_THAT(coef, VectorApproxEqual(beta, 1e-4));
  REQUIRE_THAT(coef, VectorApproxEqual(coef_scaled, 1e-4));

  auto primals = fit.getPrimals();
  double gap = fit.getGaps().back();

  REQUIRE(gap <= 1e-10);
}

TEST_CASE("Near-zero regularization on noisy data", "[quadratic][realdata]")
{
  auto [x, y] = loadData("tests/data/abalone.csv");
  auto [ols_intercept, ols_coefs] = slope::detail::fitOls(x, y);

  constexpr double tol = 1e-4;

  for (const std::string solver : { "hybrid", "pgd", "fista" }) {
    for (const double alpha : { 0.0, 1e-14, 1e-12 }) {
      for (const bool diagnostics : { false, true }) {
        CAPTURE(solver, alpha, diagnostics);

        slope::WarningLogger::clearWarnings();

        slope::Slope model;
        model.setSolver(solver);
        model.setDiagnostics(diagnostics);
        model.setMaxIterations(30000);
        model.setTol(tol);

        auto fit = model.fit(x, y, alpha);
        Eigen::VectorXd coefs = fit.getCoefs();
        const double intercept = fit.getIntercepts()(0);

        REQUIRE_FALSE(slope::WarningLogger::hasWarnings());
        REQUIRE_THAT(coefs, VectorApproxEqual(ols_coefs, 1e-3));
        REQUIRE_THAT(intercept,
                     Catch::Matchers::WithinAbs(ols_intercept, 1e-3));

        if (diagnostics) {
          const auto primals = fit.getPrimals();
          const auto gaps = fit.getGaps();
          REQUIRE(gaps.back() >= -1e-8);
          if (alpha > 0.0) {
            REQUIRE(gaps.back() <=
                    (std::abs(primals.back()) + slope::constants::EPSILON) *
                      tol);
          }
        }
      }
    }
  }
}

TEST_CASE("Quadratic OLS dual certificate", "[quadratic][dual]")
{
  Eigen::MatrixXd x(5, 3);
  Eigen::MatrixXd y(5, 2);

  // clang-format off
  x << 1, 2, 2,
       2, 1, 4,
       3, 0, 6,
       4, 1, 8,
       5, 2, 10;
  y << 1,  2,
       0,  1,
       2, -1,
       4,  3,
       3,  5;
  // clang-format on

  Eigen::VectorXd centers = x.colwise().mean();
  Eigen::MatrixXd x_effective = x.rowwise() - centers.transpose();
  Eigen::VectorXd scales =
    ((x_effective.array().square().colwise().sum()) / (x.rows() - 1)).sqrt();
  x_effective.array().rowwise() /= scales.transpose().array();

  Eigen::MatrixXd expected_theta(y.rows(), y.cols());
  for (Eigen::Index k = 0; k < y.cols(); ++k) {
    auto [intercept, coefs] =
      slope::detail::fitOls(x_effective, Eigen::VectorXd(y.col(k)), false);
    expected_theta.col(k) = x_effective * coefs - y.col(k);
  }

  const Eigen::MatrixXd theta = slope::detail::quadraticOlsDualPoint(
    x, y, centers, scales, slope::JitNormalization::Both, false);

  REQUIRE((theta - expected_theta).norm() <= 1e-12);

  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(x.cols() * y.cols());
  slope::updateGradient(gradient,
                        x,
                        theta,
                        centers,
                        scales,
                        Eigen::VectorXd::Ones(x.rows()),
                        slope::JitNormalization::Both);

  const Eigen::ArrayXd lambda =
    Eigen::ArrayXd::Constant(gradient.size(), 1e-20);
  slope::SortedL1Norm norm;
  const double dual_norm = norm.dualNorm(gradient, lambda);
  REQUIRE(dual_norm > 1.0);

  std::unique_ptr<slope::Loss> loss = std::make_unique<slope::Quadratic>();
  const double safe_dual =
    slope::computeDualFromPoint(Eigen::VectorXd::Zero(gradient.size()),
                                theta,
                                loss,
                                norm,
                                lambda,
                                x,
                                y,
                                centers,
                                scales,
                                slope::JitNormalization::Both);
  const double expected_dual =
    loss->dual(theta / dual_norm, y, Eigen::VectorXd::Ones(y.rows()));

  REQUIRE_THAT(safe_dual, Catch::Matchers::WithinAbs(expected_dual, 1e-12));

  Eigen::MatrixXd infeasible_theta = theta;
  infeasible_theta(0, 0) += 1e-6;
  const double zero_penalty_dual =
    slope::computeDualFromPoint(Eigen::VectorXd::Zero(gradient.size()),
                                infeasible_theta,
                                loss,
                                norm,
                                Eigen::ArrayXd::Zero(gradient.size()),
                                x,
                                y,
                                centers,
                                scales,
                                slope::JitNormalization::Both);

  REQUIRE(zero_penalty_dual == 0.0);
}

TEST_CASE("Lasso", "[quadratic][lasso]")
{
  using namespace Catch::Matchers;

  auto [x, y] = loadData("tests/data/diabetes.csv");

  slope::Slope model;

  model.setSolver("hybrid");
  model.setLambdaType("lasso");

  auto fit = model.path(x, y);
}
