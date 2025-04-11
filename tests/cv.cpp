#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <slope/cv.h>
#include <slope/slope.h>

TEST_CASE("Cross-validation", "[cv]")
{
  using Catch::Matchers::WithinAbs;

  auto data = generateData(1000, 10);
  Eigen::SparseMatrix<double> x_sparse = data.x.sparseView();

  SECTION("Sparse vs dense")
  {
    slope::Slope model;
    auto res_dense = crossValidate(model, data.x, data.y);
    auto res_sparse = crossValidate(model, x_sparse, data.y);

    auto optim_sparse = res_sparse.best_params;
    auto optim_dense = res_dense.best_params;

    REQUIRE_THAT(optim_sparse["alpha"], WithinAbs(optim_dense["alpha"], 1e-6));

    auto alphas1 = res_dense.results[0].alphas;
    auto alphas2 = res_sparse.results[0].alphas;

    REQUIRE_THAT(alphas1, VectorApproxEqual(alphas2, 1e-6));
  }

  for (const std::string loss :
       { "quadratic", "poisson", "logistic", "multinomial" }) {
    DYNAMIC_SECTION("Loss: " << loss)
    {
      auto data = generateData(1000, 10, loss);
      slope::Slope model;
      model.setLoss(loss);

      auto config = slope::CvConfig();
      // config.n_folds = 5;

      model.setSolver("auto");

      if (loss == "multinomial") {
        config.metric = "auc";
      } else if (loss == "poisson") {
        config.metric = "deviance";
        model.setSolver("fista");
      } else if (loss == "logistic") {
        config.metric = "accuracy";
      }

      auto res = crossValidate(model, data.x, data.y, config);

      auto optim = res.best_params;
      REQUIRE(optim["alpha"] > 0);
    }
  }

  SECTION("Vector param values")
  {
    slope::Slope model;

    auto cv_config = slope::CvConfig();

    cv_config.hyperparams["q"] = { 0.1, 0.2 };

    auto res_dense = crossValidate(model, data.x, data.y, cv_config);

    REQUIRE(res_dense.results.size() == 2);
  }

  SECTION("Multiple params")
  {
    slope::Slope model;

    auto cv_config = slope::CvConfig();

    cv_config.hyperparams["q"] = { 0.1, 0.2 };
    cv_config.hyperparams["gamma"] = { 0.0, 0.5, 1.0 };

    auto res_dense = crossValidate(model, data.x, data.y, cv_config);

    REQUIRE(res_dense.results.size() == 6);
  }
}

TEST_CASE("Repeated cross-validation", "[cv]")
{
  using Catch::Matchers::WithinAbs;

  auto data = generateData(100, 10);
  Eigen::SparseMatrix<double> x_sparse = data.x.sparseView();

  slope::Slope model;

  auto cv_config = slope::CvConfig();

  cv_config.metric = "deviance";
  cv_config.n_folds = 3;
  cv_config.n_repeats = 2;
  cv_config.hyperparams["q"] = { 0.1, 0.2 };
  cv_config.random_seed = 83;

  auto res = crossValidate(model, data.x, data.y, cv_config);

  auto optim = res.best_params;
  auto s0 = res.results[0].mean_scores;

  REQUIRE(res.results.front().score.rows() ==
          cv_config.n_repeats * cv_config.n_folds);
}

TEST_CASE("Cross-validation: user folds", "[cv]")
{
  using Catch::Matchers::WithinAbs;

  slope::Slope model;
  int n = 9;

  auto cv_config = slope::CvConfig();
  auto data = generateData(n, 2, "quadratic", 1, 1);

  std::vector<std::vector<std::vector<int>>> user_folds = {
    { { 0, 2, 4 }, { 1, 5, 8 }, { 7, 6, 3 } },
    { { 2, 0, 3 }, { 6, 5, 1 }, { 7, 1, 8 } }
  };

  cv_config.hyperparams["q"] = { 0.1, 0.2 };
  cv_config.predefined_folds = user_folds;

  auto res = crossValidate(model, data.x, data.y, cv_config);

  REQUIRE(res.results.front().score.rows() == 6);
}

TEST_CASE("Cross-validation: user folds", "[cv][user_folds]")
{
  using Catch::Matchers::WithinAbs;

  slope::Slope model;
  int n = 9;

  auto cv_config = slope::CvConfig();
  auto data = generateData(n, 2, "quadratic", 1, 0.15, 0.2, 1);

  std::vector<std::vector<std::vector<int>>> user_folds = {
    { { 0, 2, 4 }, { 1, 5, 8 }, { 7, 6, 3 } },
    { { 2, 0, 3 }, { 6, 5, 1 }, { 7, 1, 8 } }
  };

  cv_config.hyperparams["q"] = { 0.1, 0.2 };
  cv_config.predefined_folds = user_folds;

  REQUIRE_THROWS_AS(crossValidate(model, data.x, data.y, cv_config),
                    std::runtime_error);
}
