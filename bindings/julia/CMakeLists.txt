set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")

add_library(
  slopejll
  SHARED
  slopejll.cpp
  cv.cpp
  fit.cpp
  helpers.cpp
  predict.cpp
)

target_include_directories(
  slopejll
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(
  slopejll
  PUBLIC slope JlCxx::cxxwrap_julia Eigen3::Eigen ${SLOPE_LIBRARY}
)

set_target_properties(
  slopejll
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# Force C++17 to override JlCxx's C++20 requirement
target_compile_options(slopejll PRIVATE -std=c++17)

install(TARGETS slopejll LIBRARY)
