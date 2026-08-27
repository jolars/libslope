{
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
  };

  git-hooks = {
    hooks = {
      clang-format = {
        enable = true;
      };
    };
  };
}
