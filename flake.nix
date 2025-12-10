{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };

  outputs =
    { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "libslope";
          version = builtins.readFile ./version.txt;

          src = ./.;

          nativeBuildInputs = with pkgs; [ cmake ];
          buildInputs = with pkgs; [ eigen ];

          cmakeFlags = [
            "-DBUILD_TESTING=OFF"
            "-DCMAKE_BUILD_TYPE=Release"
          ];

          meta = with pkgs.lib; {
            description = "Sorted L-One Penalized Estimation (SLOPE) library";
            license = licenses.gpl3Plus;
            platforms = platforms.unix;
          };
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
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
            llvmPackages.openmp
            nodejs
          ];
        };
      }
    );
}
