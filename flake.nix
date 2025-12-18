{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    naersk.url = "github:nix-community/naersk";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs =
    {
      self,
      flake-utils,
      naersk,
      nixpkgs,
      rust-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = (import nixpkgs) {
          inherit system;
          overlays = [
            (import rust-overlay)
          ];
        };

        toolchain = (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml);
        naersk' = pkgs.callPackage naersk {
          cargo = toolchain;
          rustc = toolchain;
        };

        buildInputs = with pkgs; [
          libxkbcommon
          wayland
          vulkan-loader
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
          xorg.libxcb
          xorg.libXrender
          xorg.libXfixes
          fontconfig
          freetype
          openssl
          libgit2
          alsa-lib
          zlib
          stdenv.cc.cc.lib
        ];

        nativeBuildInputs = with pkgs; [
          toolchain
          pkg-config
          cmake
          perl
          python3
        ];
      in
      rec {
        defaultPackage = packages.gpui-ce;
        packages = {
          gpui-ce = naersk'.buildPackage {
            src = ./.;
            nativeBuildInputs = nativeBuildInputs;
            buildInputs = buildInputs;
          };
          container = pkgs.dockerTools.buildImage {
            name = "gpui-ce";
            config = {
            };
          };
        };

        devShell = pkgs.mkShell {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;
          XDG_SESSION_TYPE = "wayland";
          shellHook = ''
            export WAYLAND_DISPLAY=''${WAYLAND_DISPLAY:-wayland-0}
            export XDG_RUNTIME_DIR=''${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
          '';

          nativeBuildInputs =
            with pkgs;
            [
              nixfmt
              cmake
            ]
            ++ buildInputs
            ++ nativeBuildInputs;
        };
      }
    );
}
