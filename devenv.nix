{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:
let
  # NixOS glibc dynamic loader.
  loader = "${lib.getLib pkgs.glibc}/lib/ld-linux-x86-64.so.2";

  # pip CUDA wheels (tileiras, ptxas, ...) ship generic-Linux ELF binaries whose
  # interpreter is /lib64/ld-linux-x86-64.so.2, which NixOS only stubs out.
  # Rewrite them to the NixOS loader. Idempotent, and cheap enough to re-run on
  # every shell entry so it also repairs binaries that `uv sync` re-installed.
  patchBundledElf = ''
    site_packages="''${UV_PROJECT_ENVIRONMENT:-.venv}"/lib/python*/site-packages
    patched=0
    for bin_dir in $(find $site_packages -type d -name bin 2>/dev/null); do
      for exe in "$bin_dir"/*; do
        [ -f "$exe" ] && [ -x "$exe" ] || continue
        interp=$(patchelf --print-interpreter "$exe" 2>/dev/null) || continue
        if [ -z "$interp" ] || [ "$interp" = "${loader}" ]; then
          continue
        fi
        # Break the hardlink into uv's download cache before editing in place.
        if [ "$(stat -c %h "$exe")" -gt 1 ]; then
          cp -f "$exe" "$exe.nixos-patch" && mv -f "$exe.nixos-patch" "$exe"
        fi
        chmod u+w "$exe" 2>/dev/null || true
        patchelf --set-interpreter "${loader}" "$exe" && patched=$((patched + 1))
      done
    done
    if [ "$patched" -gt 0 ]; then
      echo "devenv: patched $patched bundled ELF binarie(s) for NixOS"
    fi
  '';
in
{
  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    uv.enable = true;
  };

  # zlib is required by manylinux wheels (numpy/pandas) on NixOS.
  # cuda_nvcc provides NixOS-executable ptxas/nvlink and CUDA_PATH for jax;
  # pip nvidia-* wheels provide the CUDA runtime libs for jax, torch and cuTile.
  # patchelf rewrites the ELF interpreter of pip-shipped CUDA binaries.
  packages = [
    pkgs.zlib
    pkgs.cudaPackages.cuda_nvcc
    pkgs.patchelf
    # Slide deck (mesh-quality presentation.pdf is written in Typst)
    pkgs.typst
  ];

  scripts = {
    fix-cuda-binaries.exec = patchBundledElf;
    check-gpu.exec = ''uv run python scripts/check_gpu.py "$@"'';
  };

  # /run/opengl-driver/lib provides libcuda.so — required by CUPTI (pip nvidia wheels),
  # without it jax xla_cuda13 fails with "Unknown CUPTI error 999" and falls back to CPU.
  enterShell = ''
    export LD_LIBRARY_PATH="${lib.getLib pkgs.glibc}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
    export CUDA_PATH="${pkgs.cudaPackages.cuda_nvcc}"
    # Persist XLA compilations so a fresh process skips the CPU-bound compile tax.
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS=10
    # triton (used by torch.compile) probes /sbin/ldconfig for libcuda, which does
    # not exist on NixOS; point it straight at the driver's libcuda.so.
    export TRITON_LIBCUDA_PATH="/run/opengl-driver/lib"
    ${patchBundledElf}
  '';
}
