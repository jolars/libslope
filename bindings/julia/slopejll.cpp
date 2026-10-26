#include "cv.h"
#include "fit.h"
#include "jlcxx/jlcxx.hpp"
#include "predict.h"

JLCXX_MODULE
define_julia_module(jlcxx::Module& module)
{
  module.method("fit_slope_dense", &fit_slope_dense);
  module.method("fit_slope_sparse", &fit_slope_sparse);
  module.method("fit_slope_dense_weighted", &fit_slope_dense_weighted);
  module.method("fit_slope_sparse_weighted", &fit_slope_sparse_weighted);
  module.method("cv_slope_dense", &cv_slope_dense);
  module.method("cv_slope_sparse", &cv_slope_sparse);
  module.method("cv_slope_dense_weighted", &cv_slope_dense_weighted);
  module.method("cv_slope_sparse_weighted", &cv_slope_sparse_weighted);
  module.method("slope_predict", &slope_predict);
}
