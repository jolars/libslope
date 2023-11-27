find_package(Python3 COMPONENTS Interpreter Development)

include(CMakePrintHelpers)

set(PY_PKG_NAME sortedl1)
set(PY_PKG_VERSION ${CMAKE_PROJECT_VERSION})
string(TIMESTAMP PYTHON_PACKAGE_DATE "%Y-%m-%d")

# Copy all R package files to the build directory
set(PY_PKG_IN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${PY_PKG_NAME})
set(PY_PKG_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/${PY_PKG_NAME})
set(CPP_SRC_DIR "${CMAKE_SOURCE_DIR}/src/${CMAKE_PROJECT_NAME}")

# Create folder and copy files that won't change at configuration time
file(MAKE_DIRECTORY ${PY_PKG_OUT_DIR}/${PY_PKG_NAME} ${PY_PKG_OUT_DIR}/src ${PY_PKG_OUT_DIR}/external)
file(MAKE_DIRECTORY ${PY_PKG_OUT_DIR}/src/${CMAKE_PROJECT_NAME})
file(COPY ${PY_PKG_IN_DIR}/src/main.cpp DESTINATION ${PY_PKG_OUT_DIR}/src)
file(COPY ${PY_PKG_IN_DIR}/pyproject.toml DESTINATION ${PY_PKG_OUT_DIR})

# Copy files to build directory
add_custom_target(py_copy ALL COMMENT "Copy source files to build directory")

# Copy R files
file(
  GLOB_RECURSE PY_FILES
  RELATIVE ${PY_PKG_IN_DIR}
  CONFIGURE_DEPENDS ${PY_PKG_IN_DIR}/${PY_PKG_NAME}/*.py)

file(
  GLOB_RECURSE CPP_SRC_HEADERS
  RELATIVE ${CPP_SRC_DIR}
  CONFIGURE_DEPENDS ${CPP_SRC_DIR}/*.h)

file(
  GLOB_RECURSE CPP_SRC_SOURCES
  RELATIVE ${CPP_SRC_DIR}
  CONFIGURE_DEPENDS ${CPP_SRC_DIR}/*.cpp)

file(
  GLOB_RECURSE RCPP_SOURCES
  RELATIVE ${PY_PKG_IN_DIR}/src
  CONFIGURE_DEPENDS ${PY_PKG_IN_DIR}/*.cpp)

set(CPP_SRC_FILES ${CPP_SRC_HEADERS} ${CPP_SRC_SOURCES})

# Configure variable files
configure_file(${PY_PKG_IN_DIR}/setup.py.in ${PY_PKG_OUT_DIR}/setup.py
               @ONLY)

# Copy all files over to the build directory
foreach(file ${PY_FILES})
  add_custom_command(
    TARGET py_copy
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${PY_PKG_IN_DIR}/${file}
            ${PY_PKG_OUT_DIR}/${PY_PKG_NAME})
endforeach()

foreach(file ${CPP_SRC_FILES})
  add_custom_command(
    TARGET py_copy
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CPP_SRC_DIR}/${file}
            ${PY_PKG_OUT_DIR}/src/${CMAKE_PROJECT_NAME})
endforeach()

# add_custom_target(py_build ALL COMMENT "Add target for python package build")

# add_custom_command(
#   TARGET py_build
#   POST_BUILD
#   COMMAND ${PYTHON_EXECUTABLE} ARGS "-m" "pip" "install" "."
#   WORKING_DIRECTORY "${PY_PKG_OUT_DIR}"
#   COMMENT "Build python package.")
#
# add_custom_command(
#   TARGET py_build
#   POST_BUILD
#   COMMAND ${RSCRIPT_EXECUTABLE} ARGS "-e" "'roxygen2::roxygenize(\".\")'"
#   WORKING_DIRECTORY "${PY_PKG_OUT_DIR}"
#   COMMENT "Document the R package code using roxygen2.")
#
# add_custom_command(
#   TARGET py_build
#   POST_BUILD
#   COMMAND ${R_EXECUTABLE} CMD build ${PY_PKG_NAME}
#   WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
#   COMMENT "Build the R package.")

install(
  CODE "execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install sortedl1
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
