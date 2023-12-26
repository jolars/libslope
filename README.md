# LIBSLOPE

[![CI](https://github.com/jolars/libslope/actions/workflows/ci.yaml/badge.svg)](https://github.com/jolars/libslope/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/gh/jolars/libslope/graph/badge.svg?token=y0mJN9eqYr)](https://codecov.io/gh/jolars/libslope)

This is a C++ library for Sorted L-One Penalized Estimation (SLOPE). Its main purpose is to serve as a backend for R and Python packages, but it can also be used as a standalone library.

## Dependencies

libslope has the following dependencies:

- A C++17 compiler
- CMake 3.15 or later
- Eigen 3.3 or later

In addition, to build the documentation you need

- Doxygen
- Sphinx
- Breathe

And to build and run the tests, you need Catch2.
