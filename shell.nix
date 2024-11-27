{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    cmake
    doxygen
    gcc
    clang
    clang-tools
  ];
  buildInputs = with pkgs; [
    catch2_3
    eigen
  ];
}
