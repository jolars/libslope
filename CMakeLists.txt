cmake_minimum_required(VERSION 3.15)

file(STRINGS "version.txt" project_version)

project(slope LANGUAGES CXX VERSION ${project_version})

option(BUILD_DOCS "Build documentation" OFF)
option(ENABLE_COVERAGE "Generate coverage report" OFF)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wno-cpp")

set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

find_package(OpenMP)

if(OPENMP_FOUND)
  message(STATUS "OpenMP found")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)

if(BUILD_TESTING)
  find_package(Catch2 3 REQUIRED)

  add_executable(
    tests
    tests/assertions.cpp
    tests/binomial.cpp
    tests/binomial.cpp
    tests/gaussian.cpp
    tests/lambda_sequence.cpp
    tests/path.cpp
    tests/poisson.cpp
    tests/printing.cpp
    tests/prox.cpp
    tests/qnorm.cpp
    tests/sparse.cpp
    tests/standardize.cpp
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

if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs)
  set(DOXYGEN_USE_MATHJAX YES)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  doxygen_add_docs(
    docs
    ${PROJECT_SOURCE_DIR}
    docs/
    README.md
    ALL
    COMMENT "Generate package documentation"
  )
endif()
