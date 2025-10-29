cmake_minimum_required(VERSION 3.15)

file(STRINGS "version.txt" project_version)

project(slope LANGUAGES CXX VERSION ${project_version})

option(BUILD_DOCS "Build documentation" OFF)
option(ENABLE_COVERAGE "Generate coverage report" OFF)
option(DAP_DEBUG "Interactive debugging" OFF)
option(BUILD_JULIA_BINDINGS "Build Julia bindings" OFF)

include(CTest)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(DAP_DEBUG)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wno-cpp")
endif()

set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

find_package(OpenMP)

if(OpenMP_FOUND)
  message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_subdirectory(src)

if(BUILD_TESTING)
  find_package(Catch2 3 REQUIRED)

  add_executable(
    tests
    tests/alpha_est.cpp
    tests/assertions.cpp
    tests/benchmarks.cpp
    tests/clusters.cpp
    tests/cv.cpp
    tests/generate_data.cpp
    tests/hybrid.cpp
    tests/lambda_sequence.cpp
    tests/load_data.cpp
    tests/logger.cpp
    tests/logistic.cpp
    tests/map.cpp
    tests/math.cpp
    tests/multinomial.cpp
    tests/normalization.cpp
    tests/path.cpp
    tests/poisson.cpp
    tests/predictions.cpp
    tests/prox.cpp
    tests/qnorm.cpp
    tests/quadratic.cpp
    tests/real_data.cpp
    tests/relax.cpp
    tests/score.cpp
    tests/screening.cpp
    tests/sparse.cpp
    tests/thresholding.cpp
    tests/utils.cpp
    tests/views.cpp
  )
  target_link_libraries(
    tests
    PRIVATE slope Catch2::Catch2WithMain Eigen3::Eigen
  )
  target_include_directories(tests PUBLIC tests/ "${PROJECT_BINARY_DIR}")

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(tests)

  if(ENABLE_COVERAGE)
    find_package(codecov)
    add_coverage(tests)
    add_coverage(slope)
    coverage_evaluate(tests-gcov)
  endif()
endif()

if(BUILD_JULIA_BINDINGS)
  add_subdirectory(bindings/julia)
endif()

if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)

  string(
    CONCAT
    AWESOME_CSS_URL
    "https://github.com/jothepro/"
    "doxygen-awesome-css/archive/refs/heads/main.zip"
  )

  FetchContent_Declare(
    doxygen-awesome-css
    URL ${AWESOME_CSS_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(doxygen-awesome-css)
  FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR AWESOME_CSS_DIR)

  # Generate the Doxyfile
  set(DOXYGEN_IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/assets")
  set(
    DOXYGEN_HTML_EXTRA_STYLESHEET
    "${AWESOME_CSS_DIR}/doxygen-awesome.css"
    "${AWESOME_CSS_DIR}/doxygen-awesome-sidebar-only.css"
  )
  set(DOXYGEN_DISABLE_INDEX NO)
  set(DOXYGEN_EXCLUDE_SYMBOLS slope::detail)
  set(DOXYGEN_FULL_SIDEBAR NO)
  set(DOXYGEN_GENERATE_TREEVIEW YES)
  set(DOXYGEN_HTML_COLORSTYLE LIGHT)
  set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
  set(DOXYGEN_PROJECT_LOGO "${CMAKE_CURRENT_SOURCE_DIR}/assets/slope-logo.png")
  set(DOXYGEN_USE_MATHJAX YES)
  set(DOXYGEN_SOURCE_BROWSER YES)

  doxygen_add_docs(
    docs
    docs/convergence.md
    docs/dependencies.md
    docs/getting_started.md
    docs/mainpage.md
    include/
    ALL
    COMMENT "Generate package documentation"
  )
endif()

install(FILES LICENSE DESTINATION share/licenses/${PROJECT_NAME})
