#include "predict.h"

std::tuple<jlcxx::Array<double>, int>
slope_predict(const Eigen::MatrixXd& eta, const std::string& loss_type)
{
  std::unique_ptr<slope::Loss> loss = slope::setupLoss(loss_type);
  Eigen::MatrixXd pred = loss->predict(eta);

  jlcxx::Array<double> pred_out;

  for (int i = 0; i < pred.rows(); ++i) {
    for (int j = 0; j < pred.cols(); ++j) {
      pred_out.push_back(pred(i, j));
    }
  }

  int m = pred.cols();

  return std::make_tuple(pred_out, m);
}
