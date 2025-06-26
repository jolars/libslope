#include "predict.h"

std::tuple<jlcxx::Array<double>, int>
slope_predict(jlcxx::ArrayRef<double, 2> eta_in,
              const int n,
              const int m,
              const std::string& loss_type)
{
  std::unique_ptr<slope::Loss> loss = slope::setupLoss(loss_type);

  Eigen::Map<Eigen::MatrixXd> eta(eta_in.data(), n, m);
  Eigen::MatrixXd pred = loss->predict(eta);

  jlcxx::Array<double> pred_out;

  for (int j = 0; j < pred.cols(); ++j) {
    for (int i = 0; i < pred.rows(); ++i) {
      pred_out.push_back(pred(i, j));
    }
  }

  int n_cols = pred.cols();

  return std::make_tuple(pred_out, n_cols);
}
