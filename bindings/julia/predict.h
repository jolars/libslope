#pragma once

#include "jlcxx/array.hpp"
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/tuple.hpp"
#include <Eigen/Core>
#include <slope/losses/setup_loss.h>

std::tuple<jlcxx::Array<double>, int>
slope_predict(jlcxx::ArrayRef<double, 2> eta_in,
              const int n,
              const int m,
              const std::string& loss_type);
