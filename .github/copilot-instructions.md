# Copilot Instructions for libslope

## Repository Overview

**libslope** is a C++ library for Sorted L-One Penalized Estimation (SLOPE) serving as a backend for R/Python packages. Medium-sized CMake project with ~40 source files, ~30 test files, 80 unit tests.

**Stack:** C++17, CMake 3.15+, Catch2 v3, Doxygen, Eigen 3.4+, OpenMP (optional)

## Build Instructions

### Prerequisites

**Ubuntu:** `sudo apt-get install -y build-essential libeigen3-dev catch2`
**macOS:** `brew install eigen catch2`
**Windows:** `choco install mingw && vcpkg install eigen3 catch2 --triplet x64-mingw-dynamic`

### Standard Build Process (ALWAYS follow in order)

1. **Configure:** `cmake -B build -S . -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug` (~2-5s, creates build/)
2. **Build:** `cmake --build build --parallel 4` (~60-120s, produces build/src/libslope.a and build/tests)
3. **Test:** `ctest --test-dir build --output-on-failure` (~180s for 80 tests, --output-on-failure is REQUIRED)

### Build Variants

**Documentation:** `cmake -B build -S . -DBUILD_DOCS=ON -DBUILD_TESTING=OFF && cmake --build build` (requires doxygen/graphviz, output: build/docs/html/)
**Coverage:** `-DENABLE_COVERAGE=ON` (requires lcov/gcovr, used by CI)
**Release:** `-DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF` (optimized)
**Julia Bindings:** `-DBUILD_JULIA_BINDINGS=ON` (optional)

**Task Runner (Optional):** Project has Taskfile.yml but `go-task` is NOT installed by default. Use CMake commands directly.

### Critical Build Notes

**ALWAYS configure before building.** Clean: `rm -rf build/*` then reconfigure. Incremental builds work. Tests take 180s minimum. Use `--parallel 4+` for speed.

## Repository Structure

### Source Code Layout

```
src/slope/           # Main library source code
  slope.h            # Main API header
  slope.cpp          # Main implementation
  losses/            # Loss function implementations
    loss.h           # Base loss class
    logistic.cpp     # Logistic regression
    multinomial.cpp  # Multinomial regression
    poisson.cpp      # Poisson regression
    quadratic.cpp    # Gaussian/quadratic loss
  solvers/           # Optimization solvers
    hybrid.cpp       # Hybrid FISTA solver
    hybrid_cd.cpp    # Coordinate descent solver
    pgd.cpp          # Proximal gradient descent
  [other core files: cv.cpp, normalize.cpp, screening.cpp, etc.]

tests/               # Test files (one per component)
  [80 test files matching src/ structure]

bindings/julia/      # Julia language bindings (optional)

docs/                # Documentation markdown files
  mainpage.md        # Doxygen main page
  dependencies.md    # Dependency documentation
  getting_started.md # Usage examples
  convergence.md     # Algorithm details

cmake/               # CMake helper modules
  FindGcov.cmake     # Coverage tool finder
  FindLcov.cmake     # Lcov finder
  Findcodecov.cmake  # Codecov integration
```

### Configuration Files

**CMakeLists.txt** (root, main build), **src/CMakeLists.txt** (library target), **package.json** (semantic-release), **version.txt** (auto-updated by CI), **.gersemirc** (CMake format: 2 spaces)

## Testing

**All tests:** `ctest --test-dir build --output-on-failure`
**Specific:** `./build/tests "test name"`
**Benchmarks:** `./build/tests [!benchmark] --benchmark-samples 20`

80 tests using Catch2 v3, mirror source structure. Slow tests: "Abalone dataset" (~7s), "Gaps on screened path" (~18s). Test data in `tests/data/`, helpers in `tests/generate_data.cpp` and `tests/load_data.cpp`.

## CI/CD Pipeline

**ci.yaml** (runs on push/PR):
1. **build-and-test:** Ubuntu/macOS/Windows matrix → install deps → cmake configure/build/install → ctest
2. **code-coverage:** Ubuntu only, `-DENABLE_COVERAGE=ON`, uploads to Codecov (requires lcov/gcovr)
3. **release:** semantic-release updates version.txt/CHANGELOG.md (requires Node.js)

**docs.yaml:** Manual/release trigger → builds Doxygen → deploys to GitHub Pages

**Replicate CI locally:** Install deps → `cmake -B build -S . -DBUILD_TESTING=ON` → `cmake --build build` → `ctest --test-dir build --output-on-failure`

## Making Code Changes

**Workflow:** Explore first (find/grep) → minimal changes → incremental build (`cmake --build build --parallel 4`, no reconfigure unless CMakeLists.txt changes) → test frequently (`ctest --test-dir build --output-on-failure`)

**Common patterns:**
- New source: Edit `src/CMakeLists.txt` add_library()
- New test: Edit root `CMakeLists.txt` add_executable(tests)
- Loss functions: `src/slope/losses/`
- Solvers: `src/slope/solvers/`

**Style:** C++17, 2-space indent (CMake), `#pragma once`, Eigen for matrices, OpenMP for parallelization

**Known quirks:** TODOs exist (optimization opportunities), hybrid solver has convergence issues with some multinomial data, warm starts disabled in some solvers

## Dependencies

**Required:** Eigen 3.4+ (header-only), CMake 3.15+, C++17 compiler, Catch2 v3 (NOT v2)
**Optional:** OpenMP (auto-detected, recommended), Doxygen+Graphviz (docs), lcov/gcovr (coverage), Node.js (CI only)
**Note:** All system-installed, no vendoring/submodules. Catch2 v3 required (v2 incompatible).

## Quick Reference

**Clean build:** `rm -rf build/* && cmake -B build -S . -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug && cmake --build build --parallel 4 && ctest --test-dir build --output-on-failure`

**Incremental:** `cmake --build build --parallel 4 && ctest --test-dir build --output-on-failure`

**Times:** Configure 2-5s, full build 60-120s, incremental 5-30s, tests 180s, docs 30-60s

**Artifacts:** `build/src/libslope.a` (library), `build/tests` (executable), `build/docs/html/` (docs), `build/compile_commands.json` (IDE database)

## Final Notes

**Trust these validated instructions.** Search only if incomplete, error not documented, or need algorithm details.

**When in doubt:** Clean (`rm -rf build/*`) → Configure → Build → Test (see Quick Reference)

**Avoid:** Building before configuring, timeout <180s, forgetting `--output-on-failure`, no `--parallel`, Catch2 v2 (need v3)
