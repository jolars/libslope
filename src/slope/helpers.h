#pragma once

#include <Eigen/Core>
#include <iostream>
#include <string>

// a template that iterates over a container called x
template<typename T>
void
printContents(const T& x, const std::string what = "")
{
  if (what != "") {
    std::cout << what << ": ";
  }

  int n = x.size();
  for (int i = 0; i < n; ++i) {
    std::cout << x[i] << " ";
  }

  std::cout << std::endl;
}

std::string
indent(const int level);
