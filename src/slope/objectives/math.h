#pragma once

#include <cmath>

namespace slope {

template<typename T>
T
sigmoid(const T& x)
{
  return 1.0 / (1.0 + std::exp(-x));
}

template<typename T>
T
clamp(const T& x, const T& lo, const T& hi)
{
  return x < lo ? lo : x > hi ? hi : x;
}

} // namespace slope
