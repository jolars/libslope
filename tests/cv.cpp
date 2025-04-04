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
    REQUIRE_THAT(optim_dense["alpha"], WithinAbs(0.01918399, 1e-3));

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

  SECTION("Multiple params")
  {
    slope::Slope model;

    auto cv_config = slope::CvConfig();

    cv_config.hyperparams["q"] = { 0.1, 0.2 };

    auto res_dense = crossValidate(model, data.x, data.y, cv_config);

    REQUIRE(res_dense.results.size() == 2);
  }

  SECTION("Pick correct best score")
  {
    slope::Slope model;
    auto data = generateData(100, 2);

    auto cv_config = slope::CvConfig();

    cv_config.metric = "deviance";
    cv_config.n_folds = 3;
    cv_config.hyperparams["q"] = { 0.1, 0.2 };
    cv_config.random_seed = 1982;

    auto res = crossValidate(model, data.x, data.y, cv_config);

    auto optim = res.best_params;
    auto s0 = res.results[0].mean_scores;

    auto best_s0 = std::min_element(s0.begin(), s0.end());

    int best_alpha_ind = std::distance(s0.begin(), best_s0);

    REQUIRE(*best_s0 == res.best_score);
    REQUIRE(res.best_params["q"] == 0.1);
    REQUIRE(res.best_params["alpha"] == res.results[0].alphas[best_alpha_ind]);

    auto s1 = res.results[1].mean_scores;
    auto best_s1 = std::min_element(s1.begin(), s1.end());

    REQUIRE(*best_s1 != res.best_score);

    cv_config.metric = "accuracy";

    data = generateData(100, 2, "logistic");
    model.setLoss("logistic");
    res = crossValidate(model, data.x, data.y, cv_config);

    optim = res.best_params;
    s0 = res.results[0].mean_scores;

    best_s0 = std::max_element(s0.begin(), s0.end());

    best_alpha_ind = std::distance(s0.begin(), best_s0);

    REQUIRE(*best_s0 == res.best_score);
    REQUIRE(res.best_params["q"] == 0.1);
    REQUIRE(res.best_params["alpha"] == res.results[0].alphas[best_alpha_ind]);

    s1 = res.results[1].mean_scores;
    best_s1 = std::min_element(s1.begin(), s1.end());

    REQUIRE(*best_s1 != res.best_score);
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
