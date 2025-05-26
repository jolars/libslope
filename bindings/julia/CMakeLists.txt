set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")

add_library(slopejll SHARED slopejll.cpp cv.cpp fit.cpp helpers.cpp)

target_link_libraries(
  slopejll
  PUBLIC slope JlCxx::cxxwrap_julia Eigen3::Eigen ${SLOPE_LIBRARY}
)

install(TARGETS slopejll LIBRARY)
