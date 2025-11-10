# Contributing to libslope

Thank you for your interest in contributing to libslope!

## Getting Started

### Building the Project

1. **Configure:**

   ```bash
   cmake -B build -S . -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Debug
   ```

2. **Build:**

   ```bash
   cmake --build build --parallel 4
   ```

3. **Test:**
   ```bash
   ctest --test-dir build --output-on-failure
   ```

### Dependencies

- C++17 compiler
- CMake 3.15+
- Eigen 3.4+
- Catch2 v3 (for testing)
- Doxygen (optional, for documentation)

## Making Changes

1. Fork the repository and create a new branch
2. Make your changes with clear, focused commits
3. Ensure all tests pass: `ctest --test-dir build --output-on-failure`
4. Update documentation if needed
5. Submit a pull request

## Code Style

- For C++ code, we loosely follow the Mozilla C++ style guide, but please see
  the exisiting code base. Make sure to format with `clang-format` before
  committing. A `.clang-format` file is provided.
- For CMake files, we use gersemi to format. A `.gersemirc` file is provided.
- We use `#pragma once`, rather than include guards, for header files.

## Commit Messages

We use conventional commit messages. Please follow this format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

The footer can include `<type>...` lines for
additional changes.

## Testing

All new features should include tests. Tests are located in the `tests/` directory and use Catch2 v3.
Check existing tests fro examples.

Run specific tests:

```bash
./build/tests "test name"
```

## Questions?

Feel free to open an issue for discussion before starting work on major changes.
