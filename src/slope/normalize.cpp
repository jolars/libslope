#include "normalize.h"

namespace slope {

Eigen::VectorXd
l2Norms(const Eigen::SparseMatrix<double>& x)
{
  const int p = x.cols();

  Eigen::VectorXd out(p);

  for (int j = 0; j < p; ++j) {
    out(j) = x.col(j).norm();
  }

  return out;
}

Eigen::VectorXd
l2Norms(const Eigen::MatrixXd& x)
{
  return x.colwise().norm();
}

Eigen::VectorXd
means(const Eigen::SparseMatrix<double>& x)
{
  const int n = x.rows();
  const int p = x.cols();

  Eigen::VectorXd out(p);

  for (int j = 0; j < p; ++j) {
    out(j) = x.col(j).sum() / n;
  }

  return out;
}

Eigen::VectorXd
means(const Eigen::MatrixXd& x)
{
  return x.colwise().mean();
}

Eigen::VectorXd
ranges(const Eigen::SparseMatrix<double>& x)
{
  const int p = x.cols();

  Eigen::VectorXd out(p);

  for (int j = 0; j < p; ++j) {
    double x_j_max = 0.0;
    double x_j_min = 0.0;

    for (typename Eigen::SparseMatrix<double>::InnerIterator it(x, j); it;
         ++it) {
      x_j_max = std::max(x_j_max, it.value());
      x_j_min = std::min(x_j_min, it.value());
    }

    out(j) = x_j_max - x_j_min;
  }

  return out;
}

Eigen::VectorXd
ranges(const Eigen::MatrixXd& x)
{
  return x.colwise().maxCoeff() - x.colwise().minCoeff();
}

Eigen::VectorXd
maxAbs(const Eigen::SparseMatrix<double>& x)
{
  const int p = x.cols();

  Eigen::VectorXd out(p);

  for (int j = 0; j < p; ++j) {
    double x_j_maxabs = 0.0;

    for (typename Eigen::SparseMatrix<double>::InnerIterator it(x, j); it;
         ++it) {
      x_j_maxabs = std::max(x_j_maxabs, std::abs(it.value()));
    }

    out(j) = x_j_maxabs;
  }

  return out;
}

Eigen::VectorXd
maxAbs(const Eigen::MatrixXd& x)
{
  return x.cwiseAbs().colwise().maxCoeff();
}

Eigen::VectorXd
mins(const Eigen::SparseMatrix<double>& x)
{
  const int p = x.cols();

  Eigen::VectorXd out(p);

  for (int j = 0; j < p; ++j) {
    double x_j_min = 0.0;

    for (typename Eigen::SparseMatrix<double>::InnerIterator it(x, j); it;
         ++it) {
      x_j_min = std::min(x_j_min, it.value());
    }

    out(j) = x_j_min;
  }

  return out;
}

Eigen::VectorXd
mins(const Eigen::MatrixXd& x)
{
  return x.colwise().minCoeff();
}

bool
normalize(Eigen::MatrixXd& x,
          Eigen::VectorXd& x_centers,
          Eigen::VectorXd& x_scales,
          const std::string& centering_type,
          const std::string& scaling_type,
          const bool modify_x)
{
  const int p = x.cols();

  computeCenters(x_centers, x, centering_type);
  computeScales(x_scales, x, scaling_type);

  bool center = centering_type != "none";
  bool scale = scaling_type != "none";
  bool normalize = center || scale;

  bool normalize_jit = normalize && !modify_x;

  if (modify_x && normalize) {
    for (int j = 0; j < p; ++j) {
      if (center) {
        x.col(j).array() -= x_centers(j);
      }
      if (scale) {
        x.col(j).array() /= x_scales(j);
      }
    }
  }

  return normalize_jit;
}

bool
normalize(Eigen::SparseMatrix<double>& x,
          Eigen::VectorXd& x_centers,
          Eigen::VectorXd& x_scales,
          const std::string& centering_type,
          const std::string& scaling_type,
          const bool)
{
  computeCenters(x_centers, x, centering_type);
  computeScales(x_scales, x, scaling_type);

  // TODO: Actually allow normalization in place for sparse matrices
  // Maybe we need to use separate scale_jit and center_jit, which we probably
  // need for the case when the user supplies only one or the other anyway. bool
  // normalize_jit = type != "none" && !modify_x;
  bool normalize_jit = (centering_type != "none") || (scaling_type != "none");

  // if (modify_x) {
  //   for (int j = 0; j < p; ++j) {
  //     for (Eigen::SparseMatrix<double>::InnerIterator it(x, j); it; ++it) {
  //       it.valueRef() = it.value() / x_scales(j);
  //     }
  //   }
  // }

  return normalize_jit;
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXd>
rescaleCoefficients(Eigen::VectorXd beta0,
                    Eigen::MatrixXd beta,
                    const Eigen::VectorXd& x_centers,
                    const Eigen::VectorXd& x_scales,
                    const bool intercept)
{
  const int p = beta.rows();
  const int m = beta.cols();

  bool centering = x_centers.size() > 0;
  bool scaling = x_scales.size() > 0;

  if (centering || scaling) {
    for (int k = 0; k < m; ++k) {
      double x_bar_beta_sum = 0.0;
      for (int j = 0; j < p; ++j) {
        if (scaling) {
          beta(j, k) /= x_scales(j);
        }
        if (centering) {
          x_bar_beta_sum += x_centers(j) * beta(j, k);
        }
      }

      if (intercept) {
        beta0(k) -= x_bar_beta_sum;
      }
    }
  }

  return { beta0, beta };
}

} // namespace slope
