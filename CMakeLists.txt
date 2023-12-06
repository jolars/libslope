cmake_minimum_required(VERSION 3.15)

file(STRINGS "version.txt" project_version)

project(
  slope
  LANGUAGES CXX
  VERSION ${project_version})

option(BUILD_DOCS "Build documentation" OFF)

include(FetchContent)
include(CTest)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)

if(BUILD_TESTING)
  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.4.0)
    FetchContent_MakeAvailable(Catch2)
  endif()

  add_executable(
    tests tests/gaussian.cpp tests/prox.cpp tests/qnorm.cpp
          tests/lambda_sequence.cpp tests/path.cpp tests/standardize.cpp)
  target_link_libraries(tests PRIVATE slope Catch2::Catch2WithMain
                                      Eigen3::Eigen)
  target_include_directories(tests PUBLIC tests/ "${PROJECT_BINARY_DIR}")

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(tests)
endif()

if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  doxygen_add_docs(docs ${PROJECT_SOURCE_DIR} docs/ ALL
                   COMMENT "Generate package documentation")
endif()
