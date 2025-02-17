#include "generate_data.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <random>

SimulatedData::SimulatedData(const int n, const int p, const int m)
{
  x = Eigen::MatrixXd::Zero(n, p);
  y = Eigen::VectorXd::Zero(n);
  beta = Eigen::MatrixXd::Zero(p, m);
}

SimulatedData
generateData(int n,
             int p,
             const std::string& type,
             int m,
             double x_sparsity,
             double coef_sparsity,
             unsigned seed)
{
  SimulatedData data(n, p, m);

  std::mt19937 rng(seed);

  std::normal_distribution<double> norm(0.0, 1.0);

  for (int j = 0; j < p; ++j) {
    int n_nonzero = std::floor(x_sparsity * n * p);
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int i = 0; i < n; ++i) {
      if (i < n_nonzero) {
        data.x(indices[i], j) = norm(rng);
      }
    }
  }

  for (int k = 0; k < m; ++k) {
    int n_nonzero = std::floor(coef_sparsity * std::min(n, p * k));
    std::vector<int> indices(p);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int j = 0; j < p; ++j) {
      if (j < n_nonzero) {
        data.beta(indices[j], k) = norm(rng);
      }
    }
  }

  Eigen::MatrixXd eta = data.x * data.beta;

  // 3. Generate response y depending on model type.
  // Pre-calculate linear predictor if applicable.
  if (type == "gaussian") {
    // Gaussian: y = x*beta + noise, with noise ~ N(0,1)
    data.y = Eigen::VectorXd(n);
    for (int i = 0; i < n; ++i)
      data.y(i) = eta(i) + norm(rng); // noise added
  } else if (type == "binomial") {
    // Binomial: logistic regression, probability = logistic(x*beta) and y ~
    // Bernoulli(prob)
    data.y = Eigen::VectorXd(n);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
      double prob = 1.0 / (1.0 + std::exp(-eta(i)));
      data.y(i) = (unif(rng) < prob) ? 1.0 : 0.0;
    }
  } else if (type == "poisson") {
    // Poisson: rate = exp(x*beta) and y ~ Poisson(rate)
    data.y = Eigen::VectorXd(n);
    for (int i = 0; i < n; ++i) {
      double lambda = std::exp(eta(i));
      std::poisson_distribution<int> pois(lambda);
      data.y(i) = static_cast<double>(pois(rng));
    }
  } else if (type == "multinomial") {
    // Multinomial logistic regression.
    // Compute scores = x * beta_multi for each row (a vector of size m)
    // Compute softmax probabilities and then sample one class.
    for (int i = 0; i < n; ++i) {
      Eigen::VectorXd scores = data.x.row(i) * data.beta;
      // Subtract max for numerical stability.
      double maxScore = scores.maxCoeff();
      Eigen::VectorXd expScores = (scores.array() - maxScore).exp();
      double sumExp = expScores.sum();
      Eigen::VectorXd prob = expScores / sumExp;
      // Sample from multinomial (cumulative probability method).
      std::uniform_real_distribution<double> unif(0.0, 1.0);
      double r = unif(rng);
      double cumulative = 0.0;
      int cls = 0;

      for (int j = 0; j < m; ++j) {
        cumulative += prob(j);
        if (r < cumulative) {
          cls = j;
          break;
        }
      }
      data.y(i) = cls;
    }
  } else {
    throw std::invalid_argument("Unknown data type");
  }

  return data;
}
