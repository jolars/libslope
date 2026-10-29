#include "generate_data.hpp"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>
#include <slope/cv.h>
#include <slope/slope.h>

TEST_CASE("Cross-validation keeps internally generated alphas absolute",
          "[cv][alpha_scale]")
{
  auto data = generateData(24, 3, "quadratic", 1, 0.8, 0.5, 129);
  Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(24, 1, 3);
  slope::Slope model;
  model.setPathLength(5);
  model.setTol(1e-8);
  slope::CvConfig config;
  config.n_folds = 3;
  config.hyperparams["gamma"] = { 0, 0.5 };
  for (const bool copy_x : { true, false }) {
    DYNAMIC_SECTION("copy_x = " << copy_x)
    {
      config.copy_x = copy_x;
      const auto absolute =
        crossValidate(model, data.x, data.y, config, weights);
      model.setAlphaScale("relative");
      const auto relative =
        crossValidate(model, data.x, data.y, config, weights);
      REQUIRE(relative.best_params == absolute.best_params);
      REQUIRE(relative.results.size() == absolute.results.size());
      for (int j = 0; j < absolute.results.size(); ++j) {
        REQUIRE_THAT(relative.results[j].alphas,
                     VectorApproxEqual(absolute.results[j].alphas, 1e-12));
        REQUIRE((relative.results[j].score - absolute.results[j].score).norm() <
                1e-8);
      }
    }
  }
}

TEST_CASE("Cross-validation", "[cv]")
{
  using Catch::Matchers::WithinAbs;

  auto data = generateData(100, 2);
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
      auto data = generateData(100, 2, loss);
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

  // Should not throw even with potential zero-variance columns from sparse data
  REQUIRE_NOTHROW(crossValidate(model, data.x, data.y, cv_config));
}

TEST_CASE("Cross-validation: do not copy x", "[cv][user_folds]")
{
  using Catch::Matchers::WithinAbs;

  slope::Slope model;
  // Compare normalization paths only after both fits have converged tightly.
  model.setTol(1e-12);
  int n = 9;

  auto cv_config = slope::CvConfig();
  auto data = generateData(n, 2, "quadratic", 1, 1);

  std::vector<std::vector<std::vector<int>>> user_folds = {
    { { 0, 2, 4 }, { 1, 5, 8 }, { 7, 6, 3 } },
    { { 2, 0, 3 }, { 6, 5, 1 }, { 7, 1, 8 } }
  };

  cv_config.hyperparams["q"] = { 0.1, 0.2 };
  cv_config.predefined_folds = user_folds;

  auto res_copy = crossValidate(model, data.x, data.y, cv_config);

  cv_config.copy_x = false;

  auto res_view = crossValidate(model, data.x, data.y, cv_config);

  REQUIRE_THAT(res_copy.results.front().score(0, 0),
               WithinAbs(res_view.results.front().score(0, 0), 1e-10));
}

TEST_CASE("Best alpha index is within bounds", "[cv][alpha_index]")
{
  // Generate small dataset for testing
  auto data = generateData(50, 5);

  slope::Slope model;

  // Create a basic CV configuration
  auto cv_config = slope::CvConfig();
  cv_config.n_folds = 3;
  cv_config.hyperparams["q"] = { 0.1 };

  // Run cross-validation
  auto res = crossValidate(model, data.x, data.y, cv_config);

  int best_alpha_ind = res.best_alpha_ind;

  // Check that best_ind is within range of results
  REQUIRE(res.best_ind >= 0);
  REQUIRE(res.best_ind < static_cast<int>(res.results.size()));

  // Get the best result
  const auto& best_result = res.results[res.best_ind];

  // Get the alpha values
  const auto& alphas = best_result.alphas;

  // Get the mean scores
  const auto& mean_scores = best_result.mean_scores;

  // Verify sizes match
  REQUIRE(alphas.size() == mean_scores.size());

  // Find best alpha index manually
  int expected_best_idx = 0;
  auto scorer = slope::Score::create(cv_config.metric); // Default is "mse"
  auto comp = scorer->getComparator();

  double best_score = mean_scores(0);
  for (int i = 1; i < mean_scores.size(); ++i) {
    if (comp(best_score, mean_scores(i))) {
      best_score = mean_scores(i);
      expected_best_idx = i;
    }
  }

  // Extract best alpha index from the parameters
  double best_alpha = res.best_params["alpha"];

  // Find the index of this alpha in the alphas array
  int actual_best_idx = -1;
  for (int i = 0; i < alphas.size(); ++i) {
    if (std::abs(alphas(i) - best_alpha) < 1e-10) {
      actual_best_idx = i;
      break;
    }
  }

  REQUIRE(best_alpha_ind >= 0);
  REQUIRE(best_alpha_ind < alphas.size());
  REQUIRE(best_alpha_ind == actual_best_idx);
  REQUIRE(best_alpha_ind == expected_best_idx);

  INFO("Alphas: " << alphas);
  INFO("Mean scores: " << mean_scores);
  INFO("Best alpha index: " << best_alpha_ind);
  INFO("Actual best alpha index: " << actual_best_idx);
  INFO("Best alpha value: " << best_alpha);
}

TEST_CASE("Observation-weighted cross-validation matches manual folds",
          "[cv][weights]")
{
  using Catch::Matchers::WithinAbs;
  using namespace slope;

  auto data = generateData(12, 2, "quadratic", 1, 1, 1, 91);
  data.x.array() += 4;
  Eigen::VectorXd weights(12);
  weights << 1, 2, 0, 4, 2, 1, 0, 3, 1, 5, 2, 1;
  Slope model;
  model.setPathLength(5);
  model.setTol(1e-8);
  model.setRelaxTol(1e-8);

  CvConfig config;
  config.hyperparams["gamma"] = { 0, 0.5 };
  config.predefined_folds = std::vector<std::vector<std::vector<int>>>{
    { { 0, 3, 6, 9 }, { 1, 4, 7, 10 }, { 2, 5, 8, 11 } }
  };
  const Folds folds(*config.predefined_folds);
  auto result = crossValidate(model, data.x, data.y, config, weights);

  for (const auto& grid : result.results) {
    for (int fold = 0; fold < 3; ++fold) {
      const auto train_indices = folds.getTrainingIndices(fold);
      const auto test_indices = folds.getTestIndices(fold);
      auto [x_train, y_train, x_test, y_test] =
        folds.split(data.x, data.y, fold);
      Eigen::VectorXd train_weights = weights(train_indices);
      Eigen::VectorXd test_weights = weights(test_indices);
      Slope fold_model = model;
      auto path = fold_model.path(x_train,
                                  y_train,
                                  grid.alphas,
                                  Eigen::ArrayXd(0),
                                  defaultInterruptChecker,
                                  train_weights);
      if (grid.params.at("gamma") > 0) {
        path = fold_model.relax(
          path, x_train, y_train, grid.params.at("gamma"), train_weights);
      }
      for (int j = 0; j < path.size(); ++j) {
        const Eigen::MatrixXd eta = path(j).predict(x_test, "linear");
        const double expected =
          ((eta - y_test).array().square().colwise() * test_weights.array())
            .sum() /
          test_weights.sum();
        REQUIRE_THAT(grid.score(fold, j), WithinAbs(expected, 1e-7));
      }
    }
    for (int j = 0; j < grid.score.cols(); ++j) {
      const double mean = grid.score.col(j).mean();
      const double standard_error =
        std::sqrt((grid.score.col(j).array() - mean).square().mean() / 3);
      REQUIRE_THAT(grid.mean_scores(j), WithinAbs(mean, 1e-12));
      REQUIRE_THAT(grid.std_errors(j), WithinAbs(standard_error, 1e-12));
    }
  }

  config.copy_x = false;
  auto views = crossValidate(model, data.x, data.y, config, weights);
  Eigen::SparseMatrix<double> x_sparse = data.x.sparseView();
  auto sparse = crossValidate(model, x_sparse, data.y, config, weights);
  auto scaled = crossValidate(model, data.x, data.y, config, 17 * weights);
  for (int i = 0; i < result.results.size(); ++i) {
    const auto& scores = result.results[i].score;
    REQUIRE_THAT((scores - views.results[i].score).norm(), WithinAbs(0, 1e-7));
    REQUIRE_THAT((scores - sparse.results[i].score).norm(), WithinAbs(0, 1e-7));
    REQUIRE_THAT((scores - scaled.results[i].score).norm(), WithinAbs(0, 1e-7));
  }
}

TEST_CASE("Cross-validation validates observation weights", "[cv][weights]")
{
  using namespace slope;
  auto data = generateData(9, 2);
  Slope model;
  model.setPathLength(3);
  CvConfig config;
  config.predefined_folds = std::vector<std::vector<std::vector<int>>>{
    { { 0, 3, 6 }, { 1, 4, 7 }, { 2, 5, 8 } }
  };
  REQUIRE_THROWS_AS(
    crossValidate(model, data.x, data.y, config, Eigen::VectorXd::Ones(8)),
    std::invalid_argument);
  REQUIRE_THROWS_AS(
    crossValidate(model, data.x, data.y, config, Eigen::VectorXd::Zero(9)),
    std::invalid_argument);

  Eigen::VectorXd weights = Eigen::VectorXd::Ones(9);
  weights(0) = weights(3) = weights(6) = 0;
  REQUIRE_THROWS_AS(crossValidate(model, data.x, data.y, config, weights),
                    std::invalid_argument);
  weights.setZero();
  weights(0) = 1;
  REQUIRE_THROWS_AS(crossValidate(model, data.x, data.y, config, weights),
                    std::invalid_argument);

  weights.setOnes();
  auto weighted = crossValidate(model, data.x, data.y, config, weights);
  auto unweighted = crossValidate(model, data.x, data.y, config);
  REQUIRE_THAT((weighted.results[0].score - unweighted.results[0].score).norm(),
               Catch::Matchers::WithinAbs(0, 1e-12));

  CvConfig empty_fold_config;
  empty_fold_config.n_folds = data.x.rows() + 1;
  REQUIRE_THROWS_AS(crossValidate(model, data.x, data.y, empty_fold_config),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
    crossValidate(model, data.x, data.y, empty_fold_config, weights),
    std::invalid_argument);

  model.setLoss("logistic");
  Eigen::VectorXd invalid_response = (data.y.array() > 0).cast<double>();
  invalid_response(0) = std::numeric_limits<double>::infinity();
  REQUIRE_THROWS_AS(
    crossValidate(model, data.x, invalid_response, config, weights),
    std::invalid_argument);
  Eigen::MatrixXd invalid_predictors = data.x;
  invalid_predictors(0, 0) = std::numeric_limits<double>::quiet_NaN();
  REQUIRE_THROWS_AS(
    crossValidate(model, invalid_predictors, data.y, config, weights),
    std::invalid_argument);
}

TEST_CASE("Weighted multinomial cross-validation preserves response encoding",
          "[cv][weights][multinomial]")
{
  using namespace slope;
  auto data = generateData(18, 2, "quadratic", 1, 1, 1, 91);
  Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(18, 1, 3);
  CvConfig config;
  config.n_folds = 3;
  config.metric = "accuracy";
  config.hyperparams["gamma"] = { 0, 0.5 };
  Slope model;
  model.setLoss("multinomial");
  model.setPathLength(3);
  model.setTol(1e-6);

  for (int classes : { 2, 3 }) {
    Eigen::VectorXd labels(18);
    for (int i = 0; i < labels.size(); ++i) {
      labels(i) = i % classes;
    }
    auto loss = setupLoss("multinomial");
    auto result = crossValidate(model, data.x, labels, config, weights);
    Folds folds(18, 3, 1, config.random_seed);
    for (const auto& grid : result.results) {
      for (int fold = 0; fold < 3; ++fold) {
        const auto train_indices = folds.getTrainingIndices(fold);
        const auto test_indices = folds.getTestIndices(fold);
        auto [x_train, y_train, x_test, y_test] =
          folds.split(data.x, labels, fold);
        Eigen::VectorXd train_weights = weights(train_indices);
        Eigen::VectorXd test_weights = weights(test_indices);
        Slope fold_model = model;
        auto path = fold_model.path(x_train,
                                    y_train,
                                    grid.alphas,
                                    Eigen::ArrayXd(0),
                                    defaultInterruptChecker,
                                    train_weights);
        if (grid.params.at("gamma") > 0) {
          path = fold_model.relax(
            path, x_train, y_train, grid.params.at("gamma"), train_weights);
        }
        for (int j = 0; j < path.size(); ++j) {
          const Eigen::MatrixXd eta = path(j).predict(x_test, "linear");
          const Eigen::VectorXd predictions = loss->predict(eta);
          const double expected =
            ((predictions.array() == y_test.col(0).array()).cast<double>() *
             test_weights.array())
              .sum() /
            test_weights.sum();
          REQUIRE_THAT(grid.score(fold, j),
                       Catch::Matchers::WithinAbs(expected, 1e-8));
        }
      }
    }
  }
}

TEST_CASE("Weighted multinomial cross-validation retains missing classes",
          "[cv][weights][multinomial]")
{
  using namespace slope;
  auto data = generateData(12, 2, "quadratic", 1, 1, 1, 91);
  Slope model;
  model.setLoss("multinomial");
  model.setPathLength(3);
  model.setTol(1e-6);
  CvConfig config;
  config.metric = "accuracy";
  config.hyperparams["gamma"] = { 0, 0.5 };
  config.predefined_folds = std::vector<std::vector<std::vector<int>>>{
    { { 0, 1, 2, 3 }, { 4, 5, 6, 7 }, { 8, 9, 10, 11 } }
  };
  Eigen::VectorXd labels(12);
  Eigen::VectorXd weights = Eigen::VectorXd::LinSpaced(12, 1, 3);

  SECTION("The highest class occurs only on a zero-weight row")
  {
    for (int i = 0; i < labels.size(); ++i) {
      labels(i) = i % 2;
    }
    labels(11) = 2;
    weights(11) = 0;
  }

  SECTION("A training fold omits the reference class")
  {
    for (int i = 0; i < labels.size(); ++i) {
      labels(i) = i < 8 ? i % 2 : 2;
    }
  }

  SECTION("A training fold omits an explicit class")
  {
    for (int i = 0; i < labels.size(); ++i) {
      labels(i) = i < 8 ? 1 + i % 2 : 0;
    }
  }

  const auto full_path = model.path(data.x,
                                    labels,
                                    Eigen::ArrayXd(0),
                                    Eigen::ArrayXd(0),
                                    defaultInterruptChecker,
                                    weights);
  REQUIRE(full_path(0).getCoefs().cols() == 2);
  const auto copied = crossValidate(model, data.x, labels, config, weights);
  config.copy_x = false;
  const auto views = crossValidate(model, data.x, labels, config, weights);
  REQUIRE(copied.results.size() == 2);
  for (int i = 0; i < copied.results.size(); ++i) {
    REQUIRE(copied.results[i].score.allFinite());
    REQUIRE((copied.results[i].score.array() >= 0).all());
    REQUIRE((copied.results[i].score.array() <= 1).all());
    REQUIRE_THAT((copied.results[i].score - views.results[i].score).norm(),
                 Catch::Matchers::WithinAbs(0, 1e-8));
  }
}
