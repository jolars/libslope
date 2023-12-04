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

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(PROJECT_SRC_DIR ${CMAKE_SOURCE_DIR}/src)

include_directories(${CMAKE_SOURCE_DIR}/external)

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

  add_executable(tests tests/gaussian.cpp tests/prox.cpp tests/qnorm.cpp
                       tests/lambda_sequence.cpp)
  target_link_libraries(tests PRIVATE libslope Catch2::Catch2WithMain)
  target_include_directories(tests PUBLIC tests/ "${PROJECT_BINARY_DIR}")

  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(tests)
endif()

if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  set(DOXYGEN_IN ${CMAKE_SOURCE_DIR}/docs/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_SOURCE_DIR}/docs/Doxyfile)
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  add_custom_target(
    docs ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    COMMENT "Generating Doxygen XML"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/docs/
    VERBATIM)
  add_custom_target(
    docs_sphinx ALL
    COMMAND make html
    COMMENT "Generating sphinx HTML"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/docs/
    VERBATIM)
endif()
