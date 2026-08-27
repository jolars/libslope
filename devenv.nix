{
  config,
  pkgs,
  ...
}:

{
  packages = with pkgs; [
    git
    bashInteractive
    bzip2
    curl
    go-task
    catch2_3
    clang
    clang-tools
    cmake
    doxygen
    eigen
    gcc
    ghostscript
    gdb
    graphviz
    hyperfine
    lcov
    lldb
    llvmPackages.openmp
    nodejs
    perf
    valgrind
  ];

  languages = {
    cplusplus = {
      enable = true;
    };

    julia = {
      enable = true;
    };
  };

  env.JULIA_PROJECT = "${config.devenv.root}/bindings/julia";

  scripts.julia-bindings-check = {
    description = "Build and smoke-test the Julia bindings";
    exec = ''
      julia --startup-file=no -e 'using Pkg; Pkg.instantiate()'
      jlcxx_prefix="$(
        julia --startup-file=no -e 'using CxxWrap; print(CxxWrap.prefix_path())'
      )"
      cmake \
        -B build/julia \
        -S . \
        -DBUILD_JULIA_BINDINGS=ON \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_PREFIX_PATH="$jlcxx_prefix"
      cmake --build build/julia --target slopejll --parallel 8
      julia \
        --startup-file=no \
        bindings/julia/smoke_test.jl \
        build/julia/lib/libslopejll
    '';
  };

  git-hooks = {
    hooks = {
      clang-format = {
        enable = true;
      };
    };
  };
}
