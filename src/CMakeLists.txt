add_library(
  slope
  slope/clusters.cpp
  slope/helpers.cpp
  slope/math.cpp
  slope/objectives/binomial.cpp
  slope/objectives/gaussian.cpp
  slope/objectives/multinomial.cpp
  slope/objectives/poisson.cpp
  slope/objectives/setup_objective.cpp
  slope/qnorm.cpp
  slope/regularization_sequence.cpp
  slope/slope.cpp
  slope/solvers/pgd.cpp
  slope/solvers/hybrid.cpp
  slope/solvers/setup_solver.cpp
  slope/solvers/slope_threshold.cpp
  slope/sorted_l1_norm.cpp
  slope/standardize.cpp
  slope/utils.cpp
)

target_include_directories(slope PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS slope DESTINATION lib)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/slope/
  DESTINATION include/slope
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
)
