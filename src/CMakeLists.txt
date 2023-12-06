add_library(
  slope
  slope/clusters.cpp
  slope/qnorm.cpp
  slope/slope.cpp
  slope/helpers.cpp
  slope/regularization_sequence.cpp
  slope/slope_threshold.cpp
  slope/sorted_l1_norm.cpp
  slope/standardize.cpp
  slope/objectives.cpp)

target_include_directories(slope PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS slope DESTINATION lib)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/slope/
  DESTINATION include/slope
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp")
