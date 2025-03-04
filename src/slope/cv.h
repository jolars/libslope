#include "folds.h"
#include "score.h"
#include "slope.h"
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace slope {

struct GridResult
{
  Eigen::MatrixXd score;                // indexed by (fold, alpha)
  std::map<std::string, double> params; // hyperparams (q, etc.)
  Eigen::ArrayXd alphas;                // the sequence of alphas from the path
  Eigen::ArrayXd
    mean_scores; // averaged over folds for each (param,alpha) combo
  Eigen::ArrayXd std_errors;
};

struct CvResult
{
  std::vector<GridResult> results;

  std::map<std::string, double> best_params;
  double best_score;
  int best_ind;
  int best_alpha_ind;
};

struct CvConfig
{
  int n_folds = 10;
  std::string metric = "mse";
  uint64_t random_seed = 42;
  std::map<std::string, std::vector<double>> hyperparams = { { "q", { 0.1 } } };
  std::optional<std::vector<std::vector<int>>> predefined_folds;
};

// Create parameter grid from map of parameter names to their possible values
inline std::vector<std::map<std::string, double>>
createGrid(const std::map<std::string, std::vector<double>>& param_values)
{
  std::vector<std::map<std::string, double>> grid;

  if (param_values.empty()) {
    return grid;
  }

  // Start with first parameter
  auto it = param_values.begin();
  for (double value : it->second) {
    std::map<std::string, double> point;
    point[it->first] = value;
    grid.push_back(point);
  }

  // Add remaining parameters
  for (++it; it != param_values.end(); ++it) {
    std::vector<std::map<std::string, double>> new_grid;
    for (const auto& existing_point : grid) {
      for (double value : it->second) {
        auto new_point = existing_point;
        new_point[it->first] = value;
        new_grid.push_back(new_point);
      }
    }
    grid = std::move(new_grid);
  }

  return grid;
}

inline void
findBestParameters(CvResult& cv_result, const std::unique_ptr<Score>& scorer)
{
  double best_score = scorer->initValue();
  auto comp = scorer->getComparator();

  for (size_t i = 0; i < cv_result.results.size(); ++i) {
    auto result = cv_result.results[i];
    int best_alpha_ind = whichBest(result.mean_scores, comp);
    double current_score = result.mean_scores(best_alpha_ind);

    assert(result.alphas(best_alpha_ind) > 0);

    if (scorer->isWorse(best_score, current_score)) {
      cv_result.best_ind = i;
      cv_result.best_score = current_score;
      cv_result.best_params = result.params;

      cv_result.best_params["alpha"] = result.alphas(best_alpha_ind);
    }
  }
}

template<typename MatrixType>
CvResult
crossValidate(Slope model,
              MatrixType& x,
              const Eigen::MatrixXd& y_in,
              const CvConfig& config = CvConfig())
{
  CvResult cv_result;

  int n = y_in.rows();

  auto loss = setupLoss(model.getLossType());

  auto y = loss->preprocessResponse(y_in);
  auto scorer = Score::create(config.metric);
  auto grid = createGrid(config.hyperparams);

  Folds folds = config.predefined_folds.has_value()
                  ? Folds(config.predefined_folds.value())
                  : Folds(n, config.n_folds, config.random_seed);

  for (const auto& params : grid) {
    GridResult result;
    result.params = params;
    model.setQ(params.at("q"));

    // model.setModifyX(false);

    auto initial_path = model.path(x, y);
    result.alphas = initial_path.getAlpha();

    assert((result.alphas > 0).all());

    // model.setModifyX(true); // We create copies of x in the loop

    Eigen::MatrixXd scores =
      Eigen::MatrixXd::Zero(config.n_folds, result.alphas.size());

    Eigen::setNbThreads(1);

#ifdef _OPENMP
    omp_set_max_active_levels(1);
#pragma omp parallel for num_threads(Threads::get()) shared(scores)
#endif
    for (int i = 0; i < config.n_folds; ++i) {
      Slope thread_model = model;
      thread_model.setModifyX(true);

      // TODO: Maybe consider not copying at all?
      auto [x_train, y_train, x_test, y_test] = folds.split(x, y, i);
      auto path = thread_model.path(x_train, y_train, result.alphas);

      for (int j = 0; j < result.alphas.size(); ++j) {
        auto eta = path(j).predict(x_test, "linear");
        scores(i, j) = scorer->eval(eta, y_test, loss);
      }
    }

    result.mean_scores = scores.colwise().mean();
    result.std_errors = stdDevs(scores).array() / std::sqrt(config.n_folds);
    result.score = std::move(scores);
    cv_result.results.push_back(result);
  }

#ifdef _OPENMP
  Eigen::setNbThreads(0);
#endif

  findBestParameters(cv_result, scorer);

  return cv_result;
}

} // namespace slope
