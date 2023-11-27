cmake_minimum_required(VERSION 3.15)

project(
  slope
  LANGUAGES CXX
  VERSION 0.1.0)

option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)

include(FetchContent)
include(CTest)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include_directories(${CMAKE_SOURCE_DIR}/external)

if(BUILD_TESTING)
  add_subdirectory(src)

  find_package(Catch2 3 QUIET)
  if(NOT Catch2_FOUND)
    FetchContent_Declare(
      Catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG v3.4.0)
    FetchContent_MakeAvailable(Catch2)
  endif()

  add_executable(tests tests/test.cpp)
  target_link_libraries(tests PRIVATE libslope Catch2::Catch2WithMain)
  target_include_directories(tests PUBLIC tests/ "${PROJECT_BINARY_DIR}")

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(tests)
endif()

add_subdirectory(bindings)

