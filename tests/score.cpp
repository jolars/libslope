#include "slope/losses/setup_loss.h"
#include "test_helpers.hpp"
#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <limits>
#include <slope/losses/logistic.h>
#include <slope/score.h>

TEST_CASE("Mean squared error score", "[score][mse]")
{
  using namespace slope;
  using Catch::Matchers::WithinAbs;

  // Create simple prediction and truth matrices
  Eigen::MatrixXd predictions(4, 1);
  predictions << 1.0, 2.0, 3.0, 4.0;

  Eigen::MatrixXd truth(4, 1);
  truth << 2.0, 2.0, 2.0, 2.0;

  // Expected MSE: ((1-2)² + (2-2)² + (3-2)² + (4-2)²) / 4 = 1.5
  auto mse = Score::create("mse");

  double score = mse->eval(predictions, truth, nullptr);

  REQUIRE_THAT(score, WithinAbs(1.5, 1e-10));

  // Test with perfect predictions
  score = mse->eval(truth, truth, nullptr);
  REQUIRE_THAT(score, WithinAbs(0.0, 1e-10));
}

TEST_CASE("Mean absolute error score", "[score][mae]")
{
  using namespace slope;
  using Catch::Matchers::WithinAbs;

  // Create simple prediction and truth matrices
  Eigen::MatrixXd predictions(4, 1);
  predictions << 1.0, 2.0, 3.0, 4.0;

  Eigen::MatrixXd truth(4, 1);
  truth << 2.0, 2.0, 2.0, 2.0;

  // Expected MAE: (|1-2| + |2-2| + |3-2| + |4-2|) / 4
  //             = (1 + 0 + 1 + 2) / 4
  //             = 4/4
  //             = 1.0
  auto mae = Score::create("mae");

  double score = mae->eval(predictions, truth, nullptr);

  REQUIRE_THAT(score, WithinAbs(1.0, 1e-10));

  // Test with perfect predictions
  score = mae->eval(truth, truth, nullptr);
  REQUIRE_THAT(score, WithinAbs(0.0, 1e-10));
}

TEST_CASE("Accuracy and misclassification", "[score][accuracy]")
{
  using Catch::Matchers::WithinAbs;
  using namespace slope;

  SECTION("Binary")
  {

    // Test binary classification
    Eigen::MatrixXd eta(6, 1);
    eta << 2.0, // -> logistic(2.0) ≈ 0.88 -> class 1
      -2.0,     // -> logistic(-2.0) ≈ 0.12 -> class 0
      1.5,      // -> logistic(1.5) ≈ 0.82 -> class 1
      -1.5,     // -> logistic(-1.5) ≈ 0.18 -> class 0
      2.5,      // -> logistic(0.5) ≈ 0.62 -> class 1
      -0.5;     // -> logistic(-0.5) ≈ 0.38 -> class 0

    Eigen::MatrixXd truth(6, 1);
    truth << 1.0, 0.0, 1.0, 0.0, 0.0, 0.0; // Ground truth labels

    auto accuracy = Score::create("accuracy");
    auto loss = setupLoss("logistic");

    double score = accuracy->eval(eta, truth, loss);
    REQUIRE_THAT(score, WithinAbs(5.0 / 6.0, 1e-10));

    // Test perfect predictions
    Eigen::MatrixXd perfect_pred(6, 1);
    perfect_pred << 1.0, 0.0, 1.0, 0.0, 0.0, 0.0;

    score = accuracy->eval(perfect_pred, truth, loss);
    REQUIRE_THAT(score, WithinAbs(1.0, 1e-10));
  }

  SECTION("Multi-class classification")
  {
    // Test multi-class classification
    // clang-format off
    Eigen::MatrixXd multi_eta(3, 3);
    multi_eta <<  2.0, -1.0, -1.0,   // class 0 has highest linear predictor
                 -1.0,  2.0, -1.0,   // class 1 has highest linear predictor
                 -1.0, -1.0,  2.0;   // class 2 has highest linear predictor
    // clang-format on

    Eigen::VectorXd multi_truth(3);
    multi_truth << 0, 1, 2;

    auto multi_loss = setupLoss("multinomial");
    auto accuracy = Score::create("accuracy");
    auto misclass = Score::create("misclass");
    double score = accuracy->eval(multi_eta, multi_truth, multi_loss);
    double misclass_score = misclass->eval(multi_eta, multi_truth, multi_loss);
    REQUIRE_THAT(score, WithinAbs(1.0, 1e-10)); // All predictions correct
    REQUIRE_THAT(
      score, WithinAbs(1 - misclass_score, 1e-6)); // All predictions correct
  }
}

TEST_CASE("ROC AUC score", "[score][auc]")
{
  using namespace slope;
  using Catch::Matchers::WithinAbs;

  SECTION("Binary classification")
  {
    // Create predictions (probabilities) and ground truth
    Eigen::MatrixXd eta(6, 1);
    eta << 2.0, // -> logistic(2.0) ≈ 0.88
      -2.0,     // -> logistic(-2.0) ≈ 0.12
      1.5,      // -> logistic(1.5) ≈ 0.82
      -1.5,     // -> logistic(-1.5) ≈ 0.18
      0.5,      // -> logistic(0.5) ≈ 0.62
      -0.5;     // -> logistic(-0.5) ≈ 0.38

    Eigen::MatrixXd truth(6, 1);
    truth << 1.0, 0.0, 1.0, 0.0, 1.0, 0.0; // Ground truth labels

    auto auc = Score::create("auc");
    auto loss = setupLoss("logistic");

    double score = auc->eval(eta, truth, loss);
    // Perfect AUC would be 1.0, good model should be > 0.5
    REQUIRE_THAT(score, WithinAbs(1.0, 1e-10)); // Perfect separation

    // One good, one bad prediction = 0.5 AUC
    Eigen::VectorXd eta_half(2);
    Eigen::VectorXd truth_half(2);

    eta_half << 2.0, -2.0;
    truth_half << 1.0, 1.0;

    score = auc->eval(eta_half, truth_half, loss);
    REQUIRE_THAT(score, WithinAbs(0.5, 1e-10)); //

    // Test with more complex case
    Eigen::MatrixXd eta2(6, 1);
    eta2 << 1.0, // -> logistic(1.0) ≈ 0.73
      0.5,       // -> logistic(0.5) ≈ 0.62
      -0.2,      // -> logistic(-0.2) ≈ 0.45
      0.8,       // -> logistic(0.8) ≈ 0.69
      -0.5,      // -> logistic(-0.5) ≈ 0.38
      0.9;       // -> logistic(0.1) ≈ 0.52

    Eigen::MatrixXd truth2(6, 1);
    truth2 << 1.0, 1.0, 0.0, 1.0, 0.0, 0.0;

    score = auc->eval(eta2, truth2, loss);
    // This case should have a good but not perfect AUC
    REQUIRE_THAT(score, WithinAbs(0.77769, 1e-3));
  }

  SECTION("Multi-class classification")
  {
    // Test multi-class AUC (one-vs-rest)
    // clang-format off
    Eigen::MatrixXd multi_eta(5, 3);
    multi_eta << 2.0, -1.0, -1.0,   // class 0
                -1.0,  2.0, -1.0,   // class 1
                -1.0, -1.0,  2.0,   // class 2
                 1.0,  0.5, -0.5,   // class 0
                -0.5,  1.0,  0.5;   // class 1
    // clang-format on

    Eigen::MatrixXd multi_truth(5, 3);
    multi_truth << 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0;

    auto multi_loss = setupLoss("multinomial");
    auto auc = Score::create("auc");
    double score = auc->eval(multi_eta, multi_truth, multi_loss);
    REQUIRE_THAT(score, WithinAbs(1.0, 1e-10)); // Should be perfect AUC

    // One hit, one miss = 0.5 AUC
    // clang-format off
    Eigen::MatrixXd multi_eta_half(2, 3);
    multi_eta_half <<  2.0, -1.0, -1.0, // class 0
                      -0.5,  1.0,  0.5; // class 1
    // clang-format on

    Eigen::MatrixXd multi_truth_half(2, 3);
    multi_truth_half << 1, 0, 0, 1, 0, 0;

    multi_loss = setupLoss("multinomial");
    score = auc->eval(multi_eta_half, multi_truth_half, multi_loss);
    REQUIRE_THAT(score, WithinAbs(0.5, 1e-10)); // Should be perfect AUC
  }
}

TEST_CASE("Deviance score", "[score][deviance]")
{
  using Catch::Matchers::WithinAbs;
  using namespace slope;

  // Test deviance score for poisson regression
  Eigen::MatrixXd eta(5, 1);
  eta << 1.5, // exp(1.5) ≈ 4.48
    0.8,      // exp(0.8) ≈ 2.23
    1.2,      // exp(1.2) ≈ 3.32
    0.5,      // exp(0.5) ≈ 1.65
    2.0;      // exp(2.0) ≈ 7.39

  Eigen::MatrixXd truth(5, 1);
  truth << 5.0, 2.0, 4.0, 1.0, 8.0;

  auto deviance = Score::create("deviance");
  auto loss = setupLoss("poisson");

  double score = deviance->eval(eta, truth, loss);

  // For Poisson deviance: 2 * sum(y * log(y/mu) - (y - mu))
  // where mu = exp(eta)
  // This value can be calculated manually for verification
  double expected_deviance = 0.111727; // Calculated based on the formula

  REQUIRE_THAT(score, WithinAbs(expected_deviance, 1e-4));

  // Perfect predictions case
  Eigen::MatrixXd perfect_eta(5, 1);
  perfect_eta << log(5.0), log(2.0), log(4.0), log(1.0), log(8.0);

  score = deviance->eval(perfect_eta, truth, loss);
  REQUIRE_THAT(score, WithinAbs(0.0, 1e-10));
}

TEST_CASE("Observation-weighted scores", "[score][weights]")
{
  using Catch::Matchers::WithinAbs;
  using namespace slope;

  Eigen::MatrixXd eta(4, 2);
  eta << 1, 4, 2, 5, 3, 6, 4, 7;
  Eigen::MatrixXd y = Eigen::MatrixXd::Constant(4, 2, 2.0);
  Eigen::VectorXd weights(4);
  weights << 1, 2, 0, 3;

  SECTION("Regression metrics preserve their response-column conventions")
  {
    auto mse = Score::create("mse");
    auto mae = Score::create("mae");
    REQUIRE_THAT(mse->eval(eta, y, nullptr, weights),
                 WithinAbs(110.0 / 6.0, 1e-12));
    REQUIRE_THAT(mae->eval(eta, y, nullptr, weights),
                 WithinAbs(30.0 / 12.0, 1e-12));
    REQUIRE_THAT(mse->eval(eta, y, nullptr, 9 * weights),
                 WithinAbs(mse->eval(eta, y, nullptr, weights), 1e-12));
    REQUIRE_THAT(mae->eval(eta, y, nullptr, Eigen::VectorXd::Ones(4)),
                 WithinAbs(mae->eval(eta, y, nullptr), 1e-12));
  }

  SECTION("Classification metrics weight observations")
  {
    auto loss = setupLoss("logistic");
    Eigen::VectorXd predictions(4), labels(4);
    predictions << 2, -2, 2, -2;
    labels << 1, 1, 0, 0;
    auto accuracy = Score::create("accuracy");
    auto misclass = Score::create("misclass");
    REQUIRE_THAT(accuracy->eval(predictions, labels, loss, weights),
                 WithinAbs(4.0 / 6.0, 1e-12));
    REQUIRE_THAT(misclass->eval(predictions, labels, loss, weights),
                 WithinAbs(2.0 / 6.0, 1e-12));
  }

  SECTION("Deviance equals integer row replication")
  {
    auto loss = setupLoss("poisson");
    auto score = Score::create("deviance");
    Eigen::VectorXd predictions(4), counts(4);
    predictions << 0.2, 0.5, 1.0, 1.5;
    counts << 1, 3, 0, 5;
    Eigen::VectorXd repeated_eta(6), repeated_y(6);
    repeated_eta << 0.2, 0.5, 0.5, 1.5, 1.5, 1.5;
    repeated_y << 1, 3, 3, 5, 5, 5;
    REQUIRE_THAT(score->eval(predictions, counts, loss, weights),
                 WithinAbs(score->eval(repeated_eta, repeated_y, loss), 1e-12));
  }

  SECTION("Invalid weights are rejected by every built-in score")
  {
    auto loss = setupLoss("logistic");
    Eigen::VectorXd predictions = Eigen::VectorXd::Zero(4);
    Eigen::VectorXd labels(4);
    labels << 1, 0, 1, 0;
    for (const auto& metric :
         { "mse", "mae", "accuracy", "misclass", "deviance", "auc" }) {
      auto scorer = Score::create(metric);
      REQUIRE_THROWS_AS(
        scorer->eval(predictions, labels, loss, Eigen::VectorXd::Ones(3)),
        std::invalid_argument);
      REQUIRE_THROWS_AS(
        scorer->eval(predictions, labels, loss, Eigen::VectorXd::Zero(4)),
        std::invalid_argument);
      Eigen::VectorXd invalid = weights;
      invalid(0) = -1;
      REQUIRE_THROWS_AS(scorer->eval(predictions, labels, loss, invalid),
                        std::invalid_argument);
      invalid(0) = std::numeric_limits<double>::infinity();
      REQUIRE_THROWS_AS(scorer->eval(predictions, labels, loss, invalid),
                        std::invalid_argument);
    }
  }
}

TEST_CASE("Observation-weighted ROC AUC", "[score][weights][auc]")
{
  using Catch::Matchers::WithinAbs;
  using namespace slope;

  Eigen::VectorXd scores(4), labels(4), weights(4);
  scores << 0.9, 0.5, 0.5, 0.1;
  labels << 1, 1, 0, 0;
  weights << 2, 1, 3, 4;

  REQUIRE_THAT(binaryRocAuc(scores, labels, weights),
               WithinAbs(13.0 / 14.0, 1e-12));
  REQUIRE_THAT(binaryRocAuc(scores, labels, 8 * weights),
               WithinAbs(13.0 / 14.0, 1e-12));
  REQUIRE_THAT(binaryRocAuc(scores, labels, Eigen::VectorXd::Ones(4)),
               WithinAbs(binaryRocAuc(scores, labels), 1e-12));

  weights << 0, 0, 3, 4;
  REQUIRE_THAT(binaryRocAuc(scores, labels, weights), WithinAbs(0.5, 1e-12));

  Eigen::MatrixXd multiclass_scores(4, 2), multiclass_labels(4, 2);
  multiclass_scores << 0.9, 0.1, 0.1, 0.9, 0.8, 0.2, 0.2, 0.8;
  multiclass_labels << 1, 0, 0, 1, 1, 0, 0, 1;
  weights << 0, 0, 2, 3;
  REQUIRE_THAT(rocAuc(multiclass_scores, multiclass_labels, weights),
               WithinAbs(1.0, 1e-12));
}

TEST_CASE("Legacy scores retain their unweighted extension point",
          "[score][weights]")
{
  class LegacyScore : public slope::MinimizeScore
  {
  public:
    double eval(const Eigen::MatrixXd&,
                const Eigen::MatrixXd&,
                const std::unique_ptr<slope::Loss>&) const override
    {
      return 42;
    }
  } legacy;

  const slope::Score& score = legacy;
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(2, 1);
  REQUIRE(score.eval(y, y, nullptr, Eigen::VectorXd::Constant(2, 3)) == 42);
  Eigen::VectorXd weights(2);
  weights << 1, 2;
  REQUIRE_THROWS_AS(score.eval(y, y, nullptr, weights), std::invalid_argument);
}
