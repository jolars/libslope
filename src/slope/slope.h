#include "parameters.h"
#include "results.h"
#include <Eigen/Sparse>

namespace slope {

Results
slope(const Eigen::MatrixXd& x,
      const Eigen::MatrixXd& y,
      Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(0),
      Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(0),
      const SlopeParameters& params = SlopeParameters());

Results
slope(const Eigen::SparseMatrix<double>& x,
      const Eigen::MatrixXd& y,
      Eigen::ArrayXd alpha = Eigen::ArrayXd::Zero(0),
      Eigen::ArrayXd lambda = Eigen::ArrayXd::Zero(0),
      const SlopeParameters& params = SlopeParameters());

} // namespace slope
