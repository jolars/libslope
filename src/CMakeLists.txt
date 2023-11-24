add_library(
  libslope
  slope/clusters.cpp
  slope/qnorm.cpp
  slope/regularization_sequence.cpp
  slope/setup_objective.cpp
  slope/slope_threshold.cpp
  slope/sorted_l1_norm.cpp
  slope/standardize.cpp
  slope/objectives/binomial.cpp
  slope/objectives/gaussian.cpp)

target_include_directories(libslope PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
