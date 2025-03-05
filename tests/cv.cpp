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
    REQUIRE_THAT(optim_dense["alpha"], WithinAbs(0.14853, 1e-3));

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
}
