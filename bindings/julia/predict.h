#pragma once

#include "jlcxx/array.hpp"
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/tuple.hpp"
#include <Eigen/Core>
#include <slope/losses/setup_loss.h>

std::tuple<jlcxx::Array<double>, int>
slope_predict(const Eigen::MatrixXd& eta, const std::string& loss_type);
