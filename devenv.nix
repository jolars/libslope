{
  pkgs,
  ...
}:

{
  packages = with pkgs; [
    git
    bashInteractive
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
    lcov
    lldb
    llvmPackage.openmp
    nodejs
  ];

  languages = {
    cplusplus = {
      enable = true;
    };
  };
}
