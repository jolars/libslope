#pragma once

#include <Eigen/Core>
#include <Rcpp.h>

// a template that iterates over a container called x
// a prints each item using Rcpp::Rcout.
template<typename T>
void
printContents(const T& x, const std::string what = "")
{
  if (what != "") {
    Rcpp::Rcout << what << ": ";
  }

  int n = x.size();
  for (int i = 0; i < n; ++i) {
    Rcpp::Rcout << x[i] << " ";
  }

  Rcpp::Rcout << std::endl;
}
