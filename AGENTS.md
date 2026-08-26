# Repository Guidelines

## Project Structure & Module Organization

`include/slope/` contains the public C++ API; keep implementation details in
`src/slope/`. Losses and solvers use matching subdirectories in both trees.
Catch2 tests live in `tests/`, with shared helpers beside them and datasets in
`tests/data/`. Optional Julia bindings are under `bindings/julia/`. CMake
helpers belong in `cmake/`, Doxygen sources in `docs/`, and images or styles in
`assets/`.

## Build, Test, and Development Commands

The Nix/devenv shell supplies CMake, Eigen, Catch2, Clang tooling, and
`go-task`.

- `task configure`: configure a debug build with tests in `build/`.
- `task build`: configure and compile with eight parallel jobs.
- `task test`: build, then run all tests with failure output.
- `./build/tests "Cross-validation"`: run a selected Catch2 test case; tags such
  as `[cv]` also work.
- `task docs`: generate Doxygen output in `build/docs/html/`.
- `task coverage`: configure, build, and test with coverage enabled.

Equivalent direct commands are `cmake -B build -S . -DBUILD_TESTING=ON`,
`cmake --build build`, and `ctest --test-dir build --output-on-failure`.

## Coding Style & Naming Conventions

Use C++17 and format C++ with `clang-format`; `.clang-format` selects Mozilla
style and two-space indentation. Format CMake with `gersemi`. Use `#pragma once`
in headers. Follow existing names: `snake_case.cpp` files and variables,
`PascalCase` types, and `camelCase` methods. Keep public declarations in
`include/` synchronized with implementations in `src/`.

## Testing Guidelines

Use Catch2 v3 and develop changes test-first. Add focused `TEST_CASE`s and
`SECTION`s to the matching `tests/*.cpp` file; create a new snake-case file when
no suitable module exists, and register it in the root `CMakeLists.txt`. Cover
success, validation, and regression paths. Run the focused case during
development and `task test` before submission. CI also measures coverage, but
the repository declares no numeric threshold.

## Commit & Pull Request Guidelines

Use Conventional Commits, matching history: `feat(solvers): add ADMM solver` or
`fix(loss): correct Poisson gradient`. Keep commits focused and use `!` or a
`BREAKING CHANGE:` footer for incompatible API changes. Pull requests should
explain the problem and solution, link relevant issues, include tests, update
public documentation when behavior changes, and confirm the full test suite
passes. Add screenshots only for documentation or asset changes where rendering
matters.
