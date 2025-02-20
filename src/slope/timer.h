#pragma once

#include <chrono>

namespace slope {

class Timer
{
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

public:
  void start();
  double elapsed() const;
};

} // namespace slope
