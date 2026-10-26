#include "fit.h"
#include "helpers.h"
#include <Eigen/SparseCore>
#include <slope/slope.h>

namespace {

void
fit_slope_dense_impl(jlcxx::ArrayRef<double, 2> x_in,
                     jlcxx::ArrayRef<double, 1> y_in,
                     jlcxx::ArrayRef<double, 1> alpha_in,
                     jlcxx::ArrayRef<double, 1> lambda_in,
                     std::int64_t n,
                     std::int64_t p,
                     std::int64_t m,
                     bool fit_intercept,
                     std::string loss_type,
                     std::string lambda_type,
                     std::string centering_type,
                     std::string scaling_type,
                     std::size_t path_length,
                     double tol,
                     std::int64_t max_it,
                     double q,
                     std::int64_t max_clusters,
                     double dev_change_tol,
                     double dev_ratio_tol,
                     double alpha_min_ratio,
                     const std::string& hybrid_cd_type,
                     int random_seed,
                     jlcxx::ArrayRef<double, 1> coef_vals_out,
                     jlcxx::ArrayRef<std::int64_t, 1> coef_rows_out,
                     jlcxx::ArrayRef<std::int64_t, 1> coef_cols_out,
                     jlcxx::ArrayRef<double, 1> intercepts_out,
                     jlcxx::ArrayRef<std::int64_t, 1> nnz_out,
                     jlcxx::ArrayRef<double, 1> alpha_out,
                     jlcxx::ArrayRef<double, 1> lambda_out,
                     const Eigen::VectorXd& observation_weights)
{
  using Eigen::ArrayXd;
  using Eigen::Map;
  using Eigen::MatrixXd;
  using Eigen::SparseMatrix;
  using Eigen::VectorXd;

  slope::Slope model = setupModel(fit_intercept,
                                  loss_type,
                                  lambda_type,
                                  centering_type,
                                  scaling_type,
                                  path_length,
                                  tol,
                                  max_it,
                                  q,
                                  max_clusters,
                                  dev_change_tol,
                                  dev_ratio_tol,
                                  alpha_min_ratio,
                                  hybrid_cd_type,
                                  random_seed);

  Map<MatrixXd> x(x_in.data(), n, p);
  Map<VectorXd> y(y_in.data(), n);
  Map<ArrayXd> alpha_map(alpha_in.data(), alpha_in.size());
  Map<ArrayXd> lambda_map(lambda_in.data(), lambda_in.size());

  ArrayXd alpha = alpha_map;
  ArrayXd lambda = lambda_map;

  auto res = model.path(
    x, y, alpha, lambda, slope::defaultInterruptChecker, observation_weights);

  returnOutput(res,
               coef_vals_out,
               coef_rows_out,
               coef_cols_out,
               intercepts_out,
               nnz_out,
               alpha_out,
               lambda_out);
}

void
fit_slope_sparse_impl(jlcxx::ArrayRef<std::int64_t, 1> x_colptr,
                      jlcxx::ArrayRef<std::int64_t, 1> x_rowval,
                      jlcxx::ArrayRef<double, 1> x_vals,
                      jlcxx::ArrayRef<double, 1> y_in,
                      jlcxx::ArrayRef<double, 1> alpha_in,
                      jlcxx::ArrayRef<double, 1> lambda_in,
                      std::int64_t n,
                      std::int64_t p,
                      std::int64_t m,
                      bool fit_intercept,
                      std::string loss_type,
                      std::string lambda_type,
                      std::string centering_type,
                      std::string scaling_type,
                      std::int64_t path_length,
                      double tol,
                      std::int64_t max_it,
                      double q,
                      std::int64_t max_clusters,
                      double dev_change_tol,
                      double dev_ratio_tol,
                      double alpha_min_ratio,
                      const std::string& hybrid_cd_type,
                      int random_seed,
                      jlcxx::ArrayRef<double, 1> coef_vals_out,
                      jlcxx::ArrayRef<std::int64_t, 1> coef_rows_out,
                      jlcxx::ArrayRef<std::int64_t, 1> coef_cols_out,
                      jlcxx::ArrayRef<double, 1> intercepts_out,
                      jlcxx::ArrayRef<std::int64_t, 1> nnz_out,
                      jlcxx::ArrayRef<double, 1> alpha_out,
                      jlcxx::ArrayRef<double, 1> lambda_out,
                      const Eigen::VectorXd& observation_weights)
{
  using Eigen::ArrayXd;
  using Eigen::Map;
  using Eigen::MatrixXd;
  using Eigen::SparseMatrix;
  using Eigen::VectorXd;

  slope::Slope model = setupModel(fit_intercept,
                                  loss_type,
                                  lambda_type,
                                  centering_type,
                                  scaling_type,
                                  path_length,
                                  tol,
                                  max_it,
                                  q,
                                  max_clusters,
                                  dev_change_tol,
                                  dev_ratio_tol,
                                  alpha_min_ratio,
                                  hybrid_cd_type,
                                  random_seed);

  auto x = mapSparseJuliaArray(x_colptr, x_rowval, x_vals, n, p);

  Map<VectorXd> y(y_in.data(), n);

  Map<ArrayXd> alpha_map(alpha_in.data(), alpha_in.size());
  Map<ArrayXd> lambda_map(lambda_in.data(), lambda_in.size());

  ArrayXd alpha = alpha_map;
  ArrayXd lambda = lambda_map;

  auto res = model.path(
    x, y, alpha, lambda, slope::defaultInterruptChecker, observation_weights);

  returnOutput(res,
               coef_vals_out,
               coef_rows_out,
               coef_cols_out,
               intercepts_out,
               nnz_out,
               alpha_out,
               lambda_out);
}

} // namespace

void
fit_slope_dense(jlcxx::ArrayRef<double, 2> x_in,
                jlcxx::ArrayRef<double, 1> y_in,
                jlcxx::ArrayRef<double, 1> alpha_in,
                jlcxx::ArrayRef<double, 1> lambda_in,
                std::int64_t n,
                std::int64_t p,
                std::int64_t m,
                bool fit_intercept,
                std::string loss_type,
                std::string lambda_type,
                std::string centering_type,
                std::string scaling_type,
                std::size_t path_length,
                double tol,
                std::int64_t max_it,
                double q,
                std::int64_t max_clusters,
                double dev_change_tol,
                double dev_ratio_tol,
                double alpha_min_ratio,
                const std::string& hybrid_cd_type,
                int random_seed,
                jlcxx::ArrayRef<double, 1> coef_vals_out,
                jlcxx::ArrayRef<std::int64_t, 1> coef_rows_out,
                jlcxx::ArrayRef<std::int64_t, 1> coef_cols_out,
                jlcxx::ArrayRef<double, 1> intercepts_out,
                jlcxx::ArrayRef<std::int64_t, 1> nnz_out,
                jlcxx::ArrayRef<double, 1> alpha_out,
                jlcxx::ArrayRef<double, 1> lambda_out)
{
  fit_slope_dense_impl(x_in,
                       y_in,
                       alpha_in,
                       lambda_in,
                       n,
                       p,
                       m,
                       fit_intercept,
                       loss_type,
                       lambda_type,
                       centering_type,
                       scaling_type,
                       path_length,
                       tol,
                       max_it,
                       q,
                       max_clusters,
                       dev_change_tol,
                       dev_ratio_tol,
                       alpha_min_ratio,
                       hybrid_cd_type,
                       random_seed,
                       coef_vals_out,
                       coef_rows_out,
                       coef_cols_out,
                       intercepts_out,
                       nnz_out,
                       alpha_out,
                       lambda_out,
                       Eigen::VectorXd());
}

void
fit_slope_dense_weighted(jlcxx::ArrayRef<double, 2> x_in,
                         jlcxx::ArrayRef<double, 1> y_in,
                         jlcxx::ArrayRef<double, 1> alpha_in,
                         jlcxx::ArrayRef<double, 1> lambda_in,
                         std::int64_t n,
                         std::int64_t p,
                         std::int64_t m,
                         bool fit_intercept,
                         std::string loss_type,
                         std::string lambda_type,
                         std::string centering_type,
                         std::string scaling_type,
                         std::size_t path_length,
                         double tol,
                         std::int64_t max_it,
                         double q,
                         std::int64_t max_clusters,
                         double dev_change_tol,
                         double dev_ratio_tol,
                         double alpha_min_ratio,
                         const std::string& hybrid_cd_type,
                         int random_seed,
                         jlcxx::ArrayRef<double, 1> coef_vals_out,
                         jlcxx::ArrayRef<std::int64_t, 1> coef_rows_out,
                         jlcxx::ArrayRef<std::int64_t, 1> coef_cols_out,
                         jlcxx::ArrayRef<double, 1> intercepts_out,
                         jlcxx::ArrayRef<std::int64_t, 1> nnz_out,
                         jlcxx::ArrayRef<double, 1> alpha_out,
                         jlcxx::ArrayRef<double, 1> lambda_out,
                         jlcxx::ArrayRef<double, 1> observation_weights)
{
  Eigen::Map<const Eigen::VectorXd> weights(observation_weights.data(),
                                            observation_weights.size());
  fit_slope_dense_impl(x_in,
                       y_in,
                       alpha_in,
                       lambda_in,
                       n,
                       p,
                       m,
                       fit_intercept,
                       loss_type,
                       lambda_type,
                       centering_type,
                       scaling_type,
                       path_length,
                       tol,
                       max_it,
                       q,
                       max_clusters,
                       dev_change_tol,
                       dev_ratio_tol,
                       alpha_min_ratio,
                       hybrid_cd_type,
                       random_seed,
                       coef_vals_out,
                       coef_rows_out,
                       coef_cols_out,
                       intercepts_out,
                       nnz_out,
                       alpha_out,
                       lambda_out,
                       weights);
}

void
fit_slope_sparse(jlcxx::ArrayRef<std::int64_t, 1> x_colptr,
                 jlcxx::ArrayRef<std::int64_t, 1> x_rowval,
                 jlcxx::ArrayRef<double, 1> x_vals,
                 jlcxx::ArrayRef<double, 1> y_in,
                 jlcxx::ArrayRef<double, 1> alpha_in,
                 jlcxx::ArrayRef<double, 1> lambda_in,
                 std::int64_t n,
                 std::int64_t p,
                 std::int64_t m,
                 bool fit_intercept,
                 std::string loss_type,
                 std::string lambda_type,
                 std::string centering_type,
                 std::string scaling_type,
                 std::int64_t path_length,
                 double tol,
                 std::int64_t max_it,
                 double q,
                 std::int64_t max_clusters,
                 double dev_change_tol,
                 double dev_ratio_tol,
                 double alpha_min_ratio,
                 const std::string& hybrid_cd_type,
                 int random_seed,
                 jlcxx::ArrayRef<double, 1> coef_vals_out,
                 jlcxx::ArrayRef<std::int64_t, 1> coef_rows_out,
                 jlcxx::ArrayRef<std::int64_t, 1> coef_cols_out,
                 jlcxx::ArrayRef<double, 1> intercepts_out,
                 jlcxx::ArrayRef<std::int64_t, 1> nnz_out,
                 jlcxx::ArrayRef<double, 1> alpha_out,
                 jlcxx::ArrayRef<double, 1> lambda_out)
{
  fit_slope_sparse_impl(x_colptr,
                        x_rowval,
                        x_vals,
                        y_in,
                        alpha_in,
                        lambda_in,
                        n,
                        p,
                        m,
                        fit_intercept,
                        loss_type,
                        lambda_type,
                        centering_type,
                        scaling_type,
                        path_length,
                        tol,
                        max_it,
                        q,
                        max_clusters,
                        dev_change_tol,
                        dev_ratio_tol,
                        alpha_min_ratio,
                        hybrid_cd_type,
                        random_seed,
                        coef_vals_out,
                        coef_rows_out,
                        coef_cols_out,
                        intercepts_out,
                        nnz_out,
                        alpha_out,
                        lambda_out,
                        Eigen::VectorXd());
}

void
fit_slope_sparse_weighted(jlcxx::ArrayRef<std::int64_t, 1> x_colptr,
                          jlcxx::ArrayRef<std::int64_t, 1> x_rowval,
                          jlcxx::ArrayRef<double, 1> x_vals,
                          jlcxx::ArrayRef<double, 1> y_in,
                          jlcxx::ArrayRef<double, 1> alpha_in,
                          jlcxx::ArrayRef<double, 1> lambda_in,
                          std::int64_t n,
                          std::int64_t p,
                          std::int64_t m,
                          bool fit_intercept,
                          std::string loss_type,
                          std::string lambda_type,
                          std::string centering_type,
                          std::string scaling_type,
                          std::int64_t path_length,
                          double tol,
                          std::int64_t max_it,
                          double q,
                          std::int64_t max_clusters,
                          double dev_change_tol,
                          double dev_ratio_tol,
                          double alpha_min_ratio,
                          const std::string& hybrid_cd_type,
                          int random_seed,
                          jlcxx::ArrayRef<double, 1> coef_vals_out,
                          jlcxx::ArrayRef<std::int64_t, 1> coef_rows_out,
                          jlcxx::ArrayRef<std::int64_t, 1> coef_cols_out,
                          jlcxx::ArrayRef<double, 1> intercepts_out,
                          jlcxx::ArrayRef<std::int64_t, 1> nnz_out,
                          jlcxx::ArrayRef<double, 1> alpha_out,
                          jlcxx::ArrayRef<double, 1> lambda_out,
                          jlcxx::ArrayRef<double, 1> observation_weights)
{
  Eigen::Map<const Eigen::VectorXd> weights(observation_weights.data(),
                                            observation_weights.size());
  fit_slope_sparse_impl(x_colptr,
                        x_rowval,
                        x_vals,
                        y_in,
                        alpha_in,
                        lambda_in,
                        n,
                        p,
                        m,
                        fit_intercept,
                        loss_type,
                        lambda_type,
                        centering_type,
                        scaling_type,
                        path_length,
                        tol,
                        max_it,
                        q,
                        max_clusters,
                        dev_change_tol,
                        dev_ratio_tol,
                        alpha_min_ratio,
                        hybrid_cd_type,
                        random_seed,
                        coef_vals_out,
                        coef_rows_out,
                        coef_cols_out,
                        intercepts_out,
                        nnz_out,
                        alpha_out,
                        lambda_out,
                        weights);
}
