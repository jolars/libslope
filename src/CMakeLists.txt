add_library(
  slope
  slope/clusters.cpp
  slope/cv.cpp
  slope/folds.cpp
  slope/kkt_check.cpp
  slope/logger.cpp
  slope/losses/loss.cpp
  slope/losses/logistic.cpp
  slope/losses/multinomial.cpp
  slope/losses/poisson.cpp
  slope/losses/quadratic.cpp
  slope/losses/setup_loss.cpp
  slope/math.cpp
  slope/normalize.cpp
  slope/qnorm.cpp
  slope/regularization_sequence.cpp
  slope/score.cpp
  slope/screening.cpp
  slope/slope.cpp
  slope/solvers/hybrid.cpp
  slope/solvers/hybrid_cd.cpp
  slope/solvers/pgd.cpp
  slope/solvers/setup_solver.cpp
  slope/solvers/slope_threshold.cpp
  slope/sorted_l1_norm.cpp
  slope/timer.cpp
  slope/utils.cpp
)

target_include_directories(
  slope
  PUBLIC 
    ${CMAKE_SOURCE_DIR}/include  # Public headers
    ${EIGEN3_INCLUDE_DIR}
  PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}  # For implementation files to access remaining internal headers
)

target_link_libraries(slope PUBLIC Eigen3::Eigen)

install(TARGETS slope DESTINATION lib)

# Install public headers only
install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/include/
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
)
