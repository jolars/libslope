#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/clusters.h>
#include <slope/losses/quadratic.h>
#include <slope/ols.h>
#include <slope/slope.h>

TEST_CASE("Relaxation reuses consecutive signed patterns", "[relax][reuse]")
{
  using Catch::Matchers::WithinAbs;

  const bool sparse = GENERATE(false, true);
  const bool prepared_response = GENERATE(false, true);
  const double gamma = GENERATE(0.0, 0.4, 1.0);
  CAPTURE(sparse, prepared_response, gamma);

  Eigen::MatrixXd x = Eigen::MatrixXd::Identity(4, 4);
  Eigen::SparseMatrix<double> sparse_x = x.sparseView();
  Eigen::VectorXd y(4);
  y << 4, -2, 1, 3;
  Eigen::VectorXd weights(4);
  weights << 1, 2, 0, 3;

  slope::Slope model;
  model.setNormalization("none");
  model.setIntercept(false);
  model.setDiagnostics(true);
  model.setRelaxTol(1e-10);

  auto make_fit = [](const Eigen::VectorXd& beta, double alpha) {
    return slope::SlopeFit{ Eigen::VectorXd::Zero(1),
                            beta.sparseView(),
                            {},
                            alpha,
                            Eigen::ArrayXd::Constant(4, alpha),
                            0,
                            0,
                            {},
                            {},
                            {},
                            0,
                            "none",
                            "none",
                            false,
                            Eigen::VectorXd::Zero(4),
                            Eigen::VectorXd::Ones(4) };
  };

  std::vector<Eigen::VectorXd> betas;
  betas.emplace_back(Eigen::VectorXd::Zero(4));
  betas.emplace_back(Eigen::VectorXd::Zero(4));
  betas.emplace_back((Eigen::Vector4d() << 1, -1, 0, 0).finished());
  betas.emplace_back((Eigen::Vector4d() << 2, -2, 0, 0).finished());
  betas.emplace_back((Eigen::Vector4d() << 2, 2, 0, 0).finished());
  betas.emplace_back((Eigen::Vector4d() << 1, 1, 0, 0).finished());
  betas.emplace_back((Eigen::Vector4d() << 2, 1, 0, 0).finished());
  betas.emplace_back((Eigen::Vector4d() << 3, 1, 0, 0).finished());
  betas.emplace_back((Eigen::Vector4d() << 3, 1, 0, 1).finished());
  betas.emplace_back((Eigen::Vector4d() << 2, 0.5, 0, 0.5).finished());
  betas.emplace_back((Eigen::Vector4d() << 2, 2, 0, 0.5).finished());
  betas.emplace_back((Eigen::Vector4d() << 1, 1, 0, 0.2).finished());
  betas.emplace_back(Eigen::VectorXd::Zero(4));

  std::vector<slope::SlopeFit> fits;
  for (size_t i = 0; i < betas.size(); ++i) {
    fits.emplace_back(make_fit(betas[i], 1.0 / (i + 1)));
  }
  const slope::SlopePath path(fits);

  auto check = [&](auto& features) {
    const auto relaxed = prepared_response
                           ? slope::detail::SlopeAccess::relax(
                               model, path, features, y, gamma, weights)
                           : model.relax(path, features, y, gamma, weights);
    REQUIRE(relaxed.size() == path.size());
    for (size_t i = 0; i < path.size(); ++i) {
      CAPTURE(i);
      const auto expected =
        model.relax(path(i), features, y, gamma, {}, {}, weights);
      REQUIRE((Eigen::MatrixXd(relaxed(i).getCoefs(false)) -
               Eigen::MatrixXd(expected.getCoefs(false)))
                .norm() < 1e-8);
      REQUIRE((relaxed(i).getIntercepts(false) - expected.getIntercepts(false))
                .norm() < 1e-8);
      REQUIRE_THAT(relaxed(i).getDeviance(),
                   WithinAbs(expected.getDeviance(), 1e-8));
      REQUIRE(relaxed(i).getAlpha() == path(i).getAlpha());
      REQUIRE((relaxed(i).getLambda() == path(i).getLambda()).all());
    }
    if (gamma < 1) {
      for (size_t i : { 1, 3, 5, 7, 9, 11 }) {
        CAPTURE(i);
        REQUIRE(relaxed(i).getPasses() == 0);
        REQUIRE(relaxed(i).getPrimals().empty());
        REQUIRE(relaxed(i).getDuals().empty());
        REQUIRE(relaxed(i).getTime().empty());
      }
      REQUIRE(relaxed(2).getPasses() > 0);
      REQUIRE(relaxed(4).getPasses() > 0);
      REQUIRE(relaxed(6).getPasses() > 0);
      REQUIRE(relaxed(8).getPasses() > 0);
      REQUIRE(relaxed(10).getPasses() > 0);
    }
  };
  if (sparse) {
    check(sparse_x);
  } else {
    check(x);
  }
}

TEST_CASE("Relaxed GLM paths reuse unpenalized fits", "[relax][reuse][weights]")
{
  const std::string loss_type =
    GENERATE("quadratic", "logistic", "poisson", "multinomial");
  const double gamma = GENERATE(0.0, 0.4);
  CAPTURE(loss_type, gamma);

  auto data = generateData(48, 3, loss_type, 3, 1, 1, 91);
  Eigen::SparseMatrix<double> x = data.x.sparseView();
  const Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(48, 1, 3);
  slope::Slope model;
  model.setLoss(loss_type);
  model.setReturnClusters(false);
  model.setRelaxTol(1e-9);
  const auto fit =
    model.fit(x, data.y, 0.08, {}, slope::defaultInterruptChecker, weights);
  const slope::SlopePath path(std::vector<slope::SlopeFit>{ fit, fit });

  const Eigen::MatrixXd y =
    slope::setupLoss(loss_type)->preprocessResponse(data.y);
  const auto relaxed =
    slope::detail::SlopeAccess::relax(model, path, x, y, gamma, weights);
  const auto expected = model.relax(fit, x, data.y, gamma, {}, {}, weights);

  for (size_t i = 0; i < path.size(); ++i) {
    REQUIRE((Eigen::MatrixXd(relaxed(i).getCoefs()) -
             Eigen::MatrixXd(expected.getCoefs()))
              .norm() < 1e-7);
    REQUIRE((relaxed(i).getIntercepts() - expected.getIntercepts()).norm() <
            1e-7);
    REQUIRE_THAT(relaxed(i).getDeviance(),
                 Catch::Matchers::WithinAbs(expected.getDeviance(), 1e-8));
  }
  REQUIRE(relaxed(1).getPasses() == 0);
  REQUIRE(Eigen::MatrixXd(x).isApprox(data.x));
}

TEST_CASE("Relaxed quadratic fits", "[relax][quadratic]")
{
  using Catch::Matchers::WithinRel;

  slope::Slope model;
  slope::SlopeFit fit;

  int n = 3;
  int p = 2;

  Eigen::MatrixXd x(n, p);
  Eigen::Vector2d beta;
  Eigen::VectorXd y(n);

  // clang-format off
  x << 1.1, 2.3,
       0.2, 1.5,
       0.5, 0.2;
  // clang-format on
  beta << 1, 2;

  y = x * beta;

  Eigen::ArrayXd lambda = Eigen::ArrayXd::Ones(p);

  SECTION("Relaxed fit is OLS on full set")
  {
    double alpha = 1e-2;

    fit = model.fit(x, y, alpha, lambda);

    Eigen::VectorXd coef0 = fit.getCoefs();

    // Assert that the coefficients are non-zero and different
    REQUIRE(coef0[0] > 0);
    REQUIRE(coef0[1] > 0);
    REQUIRE(coef0[1] != coef0[0]);

    Eigen::VectorXd bb(p);
    // bb << 0.37416793521666464, 1.7307684486543986;
    bb << 1, 2;
    double bb0 = -0.0000020240555609711093;

    Eigen::VectorXd eta = (x * bb).array() + bb0;
    Eigen::VectorXd grad = x.transpose() * (eta - y) / n;

    REQUIRE_THAT(grad(0), Catch::Matchers::WithinAbs(0.0, 1e-4));

    auto relaxed_fit = model.relax(fit, x, y);
    auto relaxed_fit2 = model.relax(fit, x, y, 0.5);

    Eigen::VectorXd coef = relaxed_fit.getCoefs();
    Eigen::VectorXd coef2 = relaxed_fit2.getCoefs();

    std::vector<double> coef_target = { 1, 2 };

    REQUIRE_THAT(coef, VectorApproxEqual(coef_target, 1e-4));
    REQUIRE(coef[0] > coef2[0]);
  }

  SECTION("Second predictor selected")
  {
    double alpha = 0.7;

    fit = model.fit(x, y, alpha, lambda);

    auto relaxed_fit = model.relax(fit, x, y);

    auto [beta0_ols, beta_ols] =
      slope::detail::fitOls(slope::subsetCols(x, { 1 }), y);

    Eigen::VectorXd coef = relaxed_fit.getCoefs();
    std::vector<double> coef_target = { 0, 2.1 };

    REQUIRE_THAT(coef[1], WithinRel(beta_ols[0], 1e-4));
  }

  SECTION("Scaling")
  {
    double alpha = 0.7;

    fit = model.fit(x, y, alpha, lambda);
    Eigen::VectorXd coef0 = fit.getCoefs();

    double gamma = 0.25;

    auto relaxed_fit = model.relax(fit, x, y, gamma);
    Eigen::VectorXd coef = relaxed_fit.getCoefs();

    auto [beta0_ols, beta_ols] =
      slope::detail::fitOls(slope::subsetCols(x, { 1 }), y);

    Eigen::VectorXd full_coefs = model.relax(fit, x, y, 0).getCoefs();

    double coef_target = (1 - gamma) * beta_ols[0] + gamma * coef0[1];

    REQUIRE_THAT(full_coefs[1], WithinRel(beta_ols[0], 1e-4));

    REQUIRE(coef[0] == 0);
    REQUIRE_THAT(coef[1], WithinRel(coef_target, 1e-4));
  }

  SECTION("Clustered solution")
  {
    double alpha = 0.12;

    int n = 4;
    int p = 3;

    Eigen::MatrixXd x(n, p);
    Eigen::VectorXd beta(p);

    // model.setNormalization("none");

    // clang-format off
    x << 1.1, 0.3, 0.2,
         0.2, 0.9, 1.1,
         0.2, 2.5, 0.5,
         0.5, 0.0, 0.2;
    // clang-format on

    beta << 2.05, 0, 2;

    Eigen::VectorXd y = x * beta;

    fit = model.fit(x, y, alpha);
    Eigen::VectorXd coef_reg = fit.getCoefs();

    // Check that there is the expected cluster
    REQUIRE(coef_reg[0] > 0);
    REQUIRE(coef_reg[0] == coef_reg[2]);

    double gamma = 0;

    auto relaxed_fit = model.relax(fit, x, y, gamma);

    Eigen::SparseMatrix<double> U =
      slope::patternMatrix(fit.getCoefs(false)).cast<double>();

    REQUIRE(U.rows() == p);
    REQUIRE(U.cols() == 2);

    Eigen::MatrixXd x_collapsed = x * U;

    auto [beta0_ols, beta_ols] = slope::detail::fitOls(x_collapsed, y);

    Eigen::VectorXd coef = relaxed_fit.getCoefs();

    REQUIRE_THAT(coef[0], WithinRel(beta_ols[0], 1e-4));
    REQUIRE_THAT(coef[2], WithinRel(beta_ols[0], 1e-4));
  }
}

TEST_CASE("Orthogonal relaxed path", "[relax]")
{
  slope::Slope model;
  int n = 3;
  int p = 3;

  Eigen::MatrixXd x(n, p);
  Eigen::VectorXd beta(p);

  model.setNormalization("none");
  model.setScreening("none");
  model.setSolver("pgd");
  model.setIntercept(false);

  // clang-format off
  x << 1, 0, 0,
       0, 1, 0,
       0, 0, 1;
  // clang-format on
  beta << 1, 0, 1;

  Eigen::VectorXd y = x * beta;

  auto path = model.path(x, y);

  Eigen::VectorXd first_nonzero = path(2).getCoefs();

  REQUIRE(first_nonzero(0) != 0);
  REQUIRE(first_nonzero(1) == 0);
  REQUIRE(first_nonzero(2) != 0);

  model.setRelaxTol(1e-6);
  model.setRelaxMaxInnerIterations(1e3);

  auto relaxed_path = model.relax(path, x, y, 0.0);

  Eigen::VectorXd coefs = relaxed_path(20).getCoefs();
  Eigen::VectorXd coefs_early = relaxed_path(3).getCoefs();

  REQUIRE(coefs(0) == beta(0));
  REQUIRE(coefs(2) == beta(2));
  REQUIRE(coefs_early(0) == coefs(0));
}

TEST_CASE("Relaxed path", "[relax]")
{
  using Catch::Matchers::WithinAbs;

  slope::Slope model;
  slope::SlopeFit fit;

  model.setPathLength(20);

  model.setRelaxTol(1e-8);
  model.setRelaxMaxInnerIterations(1e3);

  auto data = generateData(100, 2, "quadratic", 1, 1, 1);

  model.setNormalization("none");

  auto path = model.path(data.x, data.y);

  auto relaxed_path = model.relax(path, data.x, data.y, 0.0);

  Eigen::VectorXd coefs1 = relaxed_path(6).getCoefs();
  Eigen::VectorXd coefs2 = relaxed_path(12).getCoefs();

  REQUIRE(coefs1(0) != 0);
  REQUIRE(coefs1(1) != 0);

  REQUIRE_THAT(coefs1(0), WithinAbs(coefs2(0), 1e-5));
  REQUIRE_THAT(coefs1(1), WithinAbs(coefs2(1), 1e-5));

  REQUIRE(relaxed_path.size() == path.size());
}

TEST_CASE("Weighted GLM relaxation decreases the unpenalized loss",
          "[relax][weights]")
{
  using namespace slope;
  for (const std::string loss_type : { "logistic", "poisson", "multinomial" }) {
    for (bool intercept_only : { false, true }) {
      DYNAMIC_SECTION(loss_type << (intercept_only ? ": intercept only" : ""))
      {
        auto data = generateData(48, 3, loss_type, 3, 1, 1, 91);
        if (loss_type == "poisson") {
          data.y = (0.5 + 0.5 * data.x.col(0).array()).exp().round();
        }
        if (intercept_only) {
          data.x.setZero();
        }
        const Eigen::MatrixXd original_x = data.x;
        Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(48, 1, 3);
        weights(0) = weights(9) = 0;

        Slope model;
        model.setLoss(loss_type);
        model.setTol(1e-8);
        model.setRelaxTol(1e-8);
        model.setDiagnostics(true);
        auto loss = setupLoss(loss_type);
        const Eigen::MatrixXd prepared = loss->preprocessResponse(data.y);
        const Eigen::ArrayXd lambda =
          Eigen::ArrayXd::Ones(data.x.cols() * prepared.cols());
        const auto fit = model.fit(
          data.x, data.y, 0.08, lambda, defaultInterruptChecker, weights);
        const Eigen::MatrixXd initial_eta = fit.predict(data.x, "linear");
        const double initial_loss = loss->loss(initial_eta, prepared, weights);

        const auto relaxed =
          model.relax(fit, data.x, data.y, 0, {}, {}, weights);
        const Eigen::MatrixXd relaxed_eta = relaxed.predict(data.x, "linear");
        const double relaxed_loss = loss->loss(relaxed_eta, prepared, weights);
        CAPTURE(initial_loss, relaxed_loss);
        REQUIRE(data.x.isApprox(original_x));
        REQUIRE(relaxed_eta.allFinite());
        REQUIRE(loss->inverseLink(relaxed_eta).allFinite());
        REQUIRE(std::isfinite(relaxed_loss));
        REQUIRE(relaxed_loss <= initial_loss + 1e-9);

        const auto& primals = relaxed.getPrimals();
        REQUIRE_FALSE(primals.empty());
        for (size_t i = 0; i < primals.size(); ++i) {
          REQUIRE(std::isfinite(primals[i]));
          if (i > 0) {
            CAPTURE(i, primals[i - 1], primals[i]);
            REQUIRE(primals[i] <=
                    primals[i - 1] + 1e-9 * (1 + std::abs(primals[i - 1])));
          }
        }
      }
    }
  }
}
