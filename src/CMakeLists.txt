add_library(
  slope
  slope/clusters.cpp
  slope/kkt_check.cpp
  slope/math.cpp
  slope/normalize.cpp
  slope/losses/binomial.cpp
  slope/losses/gaussian.cpp
  slope/losses/multinomial.cpp
  slope/losses/poisson.cpp
  slope/losses/setup_loss.cpp
  slope/qnorm.cpp
  slope/regularization_sequence.cpp
  slope/screening.cpp
  slope/slope.cpp
  slope/solvers/hybrid.cpp
  slope/solvers/pgd.cpp
  slope/solvers/setup_solver.cpp
  slope/solvers/slope_threshold.cpp
  slope/sorted_l1_norm.cpp
  slope/timer.cpp
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
