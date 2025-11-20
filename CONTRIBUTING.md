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
  the existing code base. Make sure to format with `clang-format` before
  committing. A `.clang-format` file is provided.
- For CMake files, we use gersemi to format. A `.gersemirc` file is provided.
- We use `#pragma once`, rather than include guards, for header files.

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages. This allows automated version management and changelog generation via semantic-release.

### Format

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: A new feature (triggers minor version bump)
- `fix`: A bug fix (triggers patch version bump)
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, whitespace)
- `refactor`: Code refactoring without changing functionality
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Changes to build system or dependencies
- `ci`: Changes to CI/CD configuration
- `chore`: Other changes that don't modify src or test files

### Breaking Changes

Add `BREAKING CHANGE:` in the footer or append `!` after the type/scope to indicate breaking changes (triggers major version bump):

```
feat!: remove deprecated API
```

or

```
feat: add new parameter

BREAKING CHANGE: The old parameter is no longer supported.
```

### Examples

```
feat(solvers): add ADMM solver implementation
fix(loss): correct gradient calculation for Poisson loss
docs: update installation instructions
test(cv): add tests for cross-validation edge cases
```

## Testing

All new features should include tests. Tests are located in the `tests/` directory and use Catch2 v3.
Check existing tests for examples.

Run specific tests:

```bash
./build/tests "test name"
```

## Questions?

Feel free to open an issue for discussion before starting work on major changes.
