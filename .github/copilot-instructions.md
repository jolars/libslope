# Agent Instructions for libslope

## Repository Overview

**libslope** is a C++ library for Sorted L-One Penalized Estimation (SLOPE) serving as a backend for R/Python packages. Medium-sized CMake project with 26 source files, 32 test files, 174 test cases across 87 test executables.

**Stack:** C++17, CMake 3.15+, Catch2 v3, Doxygen, Eigen 3.4+, OpenMP (optional)

## Build Instructions

### Standard Build Process (ALWAYS follow in order)

1. **Configure:** `cmake -B build -S . -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug` (~2-5s, creates build/)
2. **Build:** `cmake --build build --parallel 4` (~60-120s, produces build/src/libslope.a and build/tests)
3. **Test:** `ctest --test-dir build --output-on-failure` (~2s for 87 tests, --output-on-failure is REQUIRED)

### Build Variants

**Documentation:** `cmake -B build -S . -DBUILD_DOCS=ON -DBUILD_TESTING=OFF && cmake --build build` (requires doxygen/graphviz, output: build/docs/html/)
**Coverage:** `-DENABLE_COVERAGE=ON` (requires lcov/gcovr, used by CI)
**Release:** `-DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF` (optimized)
**Julia Bindings:** `-DBUILD_JULIA_BINDINGS=ON` (optional)

**Task Runner (Optional):** Project has Taskfile.yml.

### Critical Build Notes

**ALWAYS configure before building.** Clean: `rm -rf build/*` then reconfigure. Incremental builds work. Tests take ~2s. Use `--parallel 4+` for speed.

## Repository Structure

### Source Code Layout

```
include/slope/       # Public API headers
  slope.h            # Main API header
  losses/            # Loss function headers
    loss.h           # Base loss class
    logistic.h       # Logistic regression
    multinomial.h    # Multinomial regression
    poisson.h        # Poisson regression
    quadratic.h      # Gaussian/quadratic loss
    setup_loss.h     # Loss factory
  solvers/           # Optimization solver headers
    solver.h         # Base solver class
    hybrid.h         # Hybrid FISTA solver
    hybrid_cd.h      # Coordinate descent solver
    pgd.h            # Proximal gradient descent
    setup_solver.h   # Solver factory
    slope_threshold.h # Thresholding operations
  [other headers: cv.h, normalize.h, screening.h, score.h, etc.]

src/slope/           # Implementation files
  slope.cpp          # Main implementation
  losses/            # Loss function implementations
    loss.cpp
    logistic.cpp
    multinomial.cpp
    poisson.cpp
    quadratic.cpp
    setup_loss.cpp
  solvers/           # Optimization solver implementations
    hybrid.cpp
    hybrid_cd.cpp
    pgd.cpp
    setup_solver.cpp
    slope_threshold.cpp
  [other implementations: cv.cpp, normalize.cpp, screening.cpp, etc.]

tests/               # Test files (32 files, 87 test executables, 174 test cases)
  alpha_est.cpp
  assertions.cpp
  benchmarks.cpp
  clusters.cpp
  cv.cpp
  generate_data.cpp  # Test data generation
  input_validation.cpp # Input validation tests
  load_data.cpp      # Test data loading
  [24 more test files]
  data/              # Test datasets
  *.hpp              # Test helpers

bindings/julia/      # Julia language bindings (optional)

docs/                # Documentation markdown files
  mainpage.md        # Doxygen main page
  dependencies.md    # Dependency documentation
  getting_started.md # Usage examples

cmake/               # CMake helper modules
  FindGcov.cmake     # Coverage tool finder
  FindLcov.cmake     # Lcov finder
  Findcodecov.cmake  # Codecov integration
```

### Configuration Files

**CMakeLists.txt** (root, main build), **src/CMakeLists.txt** (library target), **.releaserc.json** (semantic-release config), **version.txt** (auto-updated by CI), **.gersemirc** (CMake format: 2 spaces)

## Testing

**All tests:** `ctest --test-dir build --output-on-failure`
**Specific:** `./build/tests "test name"`
**Benchmarks:** `./build/tests [!benchmark] --benchmark-samples 20`

87 test executables with 174 test cases using Catch2 v3, mirror source structure. Slow tests: "Abalone dataset" (~7s), "Gaps on screened path" (~18s). Test data in `tests/data/`, helpers in `tests/generate_data.cpp`, `tests/load_data.cpp`, and `tests/test_helpers.hpp`.

## CI/CD Pipeline

**build-and-test.yaml** (runs on push/PR): Ubuntu/macOS/Windows matrix → install deps → cmake configure/build/install → ctest

**codecov.yaml** (runs on push/PR): Ubuntu only, `-DENABLE_COVERAGE=ON`, uploads to Codecov (requires lcov/gcovr)

**release.yml** (manual trigger): Runs build-and-test → semantic-release updates version.txt/CHANGELOG.md

**docs.yaml**: Manual/release trigger → builds Doxygen → deploys to GitHub Pages

**Replicate CI locally:** Install deps → `cmake -B build -S . -DBUILD_TESTING=ON` → `cmake --build build --parallel 4` → `ctest --test-dir build --output-on-failure`

## Making Code Changes

**Workflow:** Explore first (find/grep) → minimal changes → incremental build (`cmake --build build --parallel 4`, no reconfigure unless CMakeLists.txt changes) → test frequently (`ctest --test-dir build --output-on-failure`)

**Common patterns:**

- New source: Add to `src/CMakeLists.txt` add_library()
- New test: Add to root `CMakeLists.txt` add_executable(tests)
- Loss functions: Headers in `include/slope/losses/`, implementations in `src/slope/losses/`
- Solvers: Headers in `include/slope/solvers/`, implementations in `src/slope/solvers/`

**Testing Requirements (CRITICAL):**

- **ALWAYS add unit tests for new features/validation immediately** - don't wait to be asked
- Add test file to `tests/` directory (e.g., `tests/feature_name.cpp`)
- Update root `CMakeLists.txt` to include the new test file in add_executable(tests)
- Use Catch2 v3 macros: `TEST_CASE`, `SECTION`, `REQUIRE`, `REQUIRE_THROWS_AS`, `REQUIRE_NOTHROW`
- Include necessary matchers: `#include <catch2/matchers/catch_matchers_string.hpp>` for string matching
- Test both success and failure cases
- Verify all tests pass before considering feature complete

**Style:** C++17, 2-space indent (CMake), `#pragma once`, Eigen for matrices, OpenMP for parallelization

**Known quirks:** TODOs exist (optimization opportunities), hybrid solver has convergence issues with some multinomial data, warm starts disabled in some solvers

## Commit Message Conventions

**Use [Conventional Commits](https://www.conventionalcommits.org/)** for all commits. Format: `<type>(<scope>): <subject>`

**Types:**
- `feat`: New feature (minor version bump)
- `fix`: Bug fix (patch version bump)
- `docs`: Documentation changes
- `style`: Code style/formatting
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test changes
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Other non-src/test changes

**Breaking changes:** Use `feat!:` or add `BREAKING CHANGE:` in footer (major version bump)

**Examples:** `feat(solvers): add ADMM solver`, `fix(loss): correct Poisson gradient`, `docs: update README`, `test(cv): add edge case tests`

**Note:** semantic-release auto-generates version.txt and CHANGELOG.md from commit messages

## Dependencies

**Required:** Eigen 3.4+ (header-only), CMake 3.15+, C++17 compiler, Catch2 v3 (NOT v2)
**Optional:** OpenMP (auto-detected, recommended), Doxygen+Graphviz (docs), lcov (coverage), Node.js (CI only)
**Note:** All system-installed, no vendoring/submodules. Catch2 v3 required (v2 incompatible).

## Quick Reference

**Clean build:** `rm -rf build/* && cmake -B build -S . -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug && cmake --build build --parallel 4 && ctest --test-dir build --output-on-failure`

**Incremental:** `cmake --build build --parallel 4 && ctest --test-dir build --output-on-failure`

**Times:** Configure 2-5s, full build 60-120s, incremental 5-30s, tests ~2s, docs 30-60s

**Artifacts:** `build/src/libslope.a` (library), `build/tests` (executable), `build/docs/html/` (docs), `build/compile_commands.json` (IDE database)

## Final Notes

**Trust these validated instructions.** Search only if incomplete, error not documented, or need algorithm details.

**When in doubt:** Clean (`rm -rf build/*`) → Configure → Build → Test (see Quick Reference)

**Avoid:** Building before configuring, timeout <5s for tests, forgetting `--output-on-failure`, no `--parallel`, Catch2 v2 (need v3)
