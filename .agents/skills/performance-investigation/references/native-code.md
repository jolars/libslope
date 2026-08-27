# Profiling native libslope code

Use this reference when measuring compiled C++ code in libslope. The devenv
declares `hyperfine`, `perf`, and Valgrind; enter the project shell before using
the commands below.

## Build optimized code with symbols

Keep profiling objects separate from the Debug and Release benchmark trees:

```bash
cmake -B build/profile -S . \
  -DBUILD_DOCS=OFF \
  -DENABLE_COVERAGE=OFF \
  -DBUILD_TESTING=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g -DNDEBUG -fno-omit-frame-pointer"
cmake --build build/profile --target tests --parallel 8
```

This build preserves representative optimization while resolving project source
lines and frame-pointer call stacks. Verify that project symbols appear in the
profile before drawing conclusions. Inlining and Eigen expression templates can
move samples away from the apparent source expression, so inspect callers and
generated instructions when attribution is surprising.

Use the Release tree for elapsed-time benchmarks:

