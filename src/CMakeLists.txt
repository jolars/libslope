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

target_compile_features(slope PUBLIC cxx_std_17)

target_include_directories(
  slope
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(slope PUBLIC Eigen3::Eigen)

if(OpenMP_FOUND)
  target_link_libraries(slope PUBLIC OpenMP::OpenMP_CXX)
endif()

if(DAP_DEBUG)
  target_compile_options(slope PRIVATE -O0 -Wno-cpp)
endif()
