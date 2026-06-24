#!/usr/bin/env bash
# Build the torch-strumpack binding wheel against an already-built STRUMPACK
# (STRUMPACK_DIR set by the caller), bundle the STRUMPACK + CPU math shared
# libs into the wheel, and stamp a per-backend build tag.
#
# The native-lib bundler is OS-specific:
#   Linux  -> auditwheel  (manylinux platform tag)
#   macOS  -> delocate    (macosx_<ver>_<arch> platform tag)
# delvewheel (Windows) is driven from scripts/build_wheel.ps1, not this script.
#
# Usage: ci_build_wheel.sh <backend-tag>     # e.g. cpu | cuda12x | rocm6x
#
# Env in:
#   STRUMPACK_DIR        CMake package dir of the STRUMPACK install
#   STRUMPACK_PREFIX     install prefix (for runtime lib path; default /opt/strumpack)
#   PYBIN                python bin dir to use (manylinux); empty -> `python` on PATH
#   AUDITWHEEL_EXCLUDE   comma-separated libs to NOT vendor (GPU runtimes)  [Linux]
set -euo pipefail

BACKEND_TAG="${1:?usage: ci_build_wheel.sh <backend-tag>}"
PREFIX="${STRUMPACK_PREFIX:-/opt/strumpack}"
UNAME="$(uname -s)"

if command -v nproc >/dev/null 2>&1; then
  NJOBS="$(nproc)"
elif [ "$UNAME" = "Darwin" ]; then
  NJOBS="$(sysctl -n hw.ncpu)"
else
  NJOBS="4"
fi

if [ -n "${PYBIN:-}" ]; then
  PYEXE="$PYBIN/python"
else
  PYEXE="$(command -v python3 || command -v python)"
fi
echo "==> python: $PYEXE ($($PYEXE --version 2>&1))  os=$UNAME"

"$PYEXE" -m pip install --upgrade pip wheel build
if [ "$UNAME" = "Darwin" ]; then
  "$PYEXE" -m pip install delocate
else
  "$PYEXE" -m pip install auditwheel
fi
# nanobind + torch headers are needed to compile the extension. The cpu
# backend MUST link against CPU torch (the default PyPI torch pulls a CUDA
# build, which would (a) bloat the build env and (b) make the .cpu wheel
# install drag a CUDA torch into a torch+cpu venv). TORCH_INDEX_URL lets the
# caller pin the torch wheel index, e.g. https://download.pytorch.org/whl/cpu
"$PYEXE" -m pip install nanobind numpy
if [ -n "${TORCH_INDEX_URL:-}" ]; then
  "$PYEXE" -m pip install --index-url "$TORCH_INDEX_URL" "torch>=2.1"
else
  "$PYEXE" -m pip install "torch>=2.1"
fi

# Make the freshly built STRUMPACK + its math deps discoverable at link &
# repair time.
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${DYLD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"

# On macOS, also make Homebrew openblas/metis dylibs resolvable for the repair.
if [ "$UNAME" = "Darwin" ]; then
  BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
  OPENBLAS_PREFIX="$(brew --prefix openblas 2>/dev/null || echo "$BREW_PREFIX/opt/openblas")"
  METIS_PREFIX="$(brew --prefix metis 2>/dev/null || echo "$BREW_PREFIX/opt/metis")"
  export DYLD_LIBRARY_PATH="$OPENBLAS_PREFIX/lib:$METIS_PREFIX/lib:$DYLD_LIBRARY_PATH"
  export CMAKE_PREFIX_PATH="$OPENBLAS_PREFIX:$METIS_PREFIX:$CMAKE_PREFIX_PATH"
fi

# --- compile the extension into the package dir via CMake -----------
# On macOS the STRUMPACK config does find_dependency(OpenMP); AppleClang needs
# Homebrew libomp to satisfy it. Surface those flags so find_package(OpenMP)
# succeeds for the C/CXX extension (STRUMPACK itself uses Fortran OpenMP).
EXTRA_CMAKE_ARGS=()
# ROCm: STRUMPACK::strumpack transitively carries the hip::device INTERFACE,
# which adds `-x hip` to the extension's C++ compile. The image-default GNU g++
# rejects that ("language hip not recognized"), so compile the extension with
# the ROCm clang too (matches how build_strumpack.sh built STRUMPACK).
if [ "$BACKEND_TAG" = "rocm6x" ]; then
  : "${ROCM_PATH:=/opt/rocm}"
  HIPCXX="$ROCM_PATH/llvm/bin/amdclang++"
  HIPCC="$ROCM_PATH/llvm/bin/amdclang"
  [ -x "$HIPCXX" ] || HIPCXX="$(command -v hipcc || echo "$ROCM_PATH/bin/hipcc")"
  [ -x "$HIPCC" ]  || HIPCC="$(command -v hipcc || echo "$ROCM_PATH/bin/hipcc")"
  echo "==> ROCm extension C/C++ compiler: CXX=$HIPCXX CC=$HIPCC"
  EXTRA_CMAKE_ARGS+=(
    -DCMAKE_C_COMPILER="$HIPCC"
    -DCMAKE_CXX_COMPILER="$HIPCXX"
  )
fi
if [ "$UNAME" = "Darwin" ]; then
  LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
  if [ -n "$LIBOMP_PREFIX" ]; then
    EXTRA_CMAKE_ARGS+=(
      -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_PREFIX/include"
      -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_PREFIX/include"
      -DOpenMP_C_LIB_NAMES=omp
      -DOpenMP_CXX_LIB_NAMES=omp
      -DOpenMP_omp_LIBRARY="$LIBOMP_PREFIX/lib/libomp.dylib"
    )
  fi
fi

BUILD_DIR="build_ext"
rm -rf "$BUILD_DIR"
cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYEXE" \
  ${EXTRA_CMAKE_ARGS[@]+"${EXTRA_CMAKE_ARGS[@]}"} \
  -DSTRUMPACK_DIR="${STRUMPACK_DIR:?set STRUMPACK_DIR}"
cmake --build "$BUILD_DIR" -j"$NJOBS"
cmake --install "$BUILD_DIR"          # drops _strumpack_ext*.so into torch_strumpack/

echo "==> extension built:"
ls -1 torch_strumpack/*.so

# --- build the platform wheel (BinaryDistribution => platform tag) --
rm -rf dist
"$PYEXE" -m build --wheel --no-isolation

# --- bundle native libs into the wheel ------------------------------
rm -rf dist-repaired dist-final
mkdir -p dist-repaired

if [ "$UNAME" = "Darwin" ]; then
  # delocate copies the STRUMPACK + openblas + metis dylibs into the wheel and
  # rewrites the install names. It also stamps the macosx_<ver>_<arch> platform
  # tag, so the mac cpu wheel won't collide with the linux cpu wheel as a
  # Release asset. DYLD_LIBRARY_PATH (set above) lets it find the dylibs.
  for whl in dist/*.whl; do
    "$PYEXE" -m delocate.cmd.delocate_wheel \
      -v \
      -w dist-repaired \
      "$whl"
  done
else
  EXCLUDE_ARGS=()
  if [ -n "${AUDITWHEEL_EXCLUDE:-}" ]; then
    IFS=',' read -ra _ex <<< "$AUDITWHEEL_EXCLUDE"
    for lib in "${_ex[@]}"; do EXCLUDE_ARGS+=( --exclude "$lib" ); done
  fi

  # manylinux_2_28 inside the pypa image; the ROCm job builds on ubuntu 22.04
  # (glibc 2.35) so it must target manylinux_2_35. Caller may override.
  PLAT="${AUDITWHEEL_PLAT:-manylinux_2_28_x86_64}"

  for whl in dist/*.whl; do
    "$PYEXE" -m auditwheel repair \
      --plat "$PLAT" \
      "${EXCLUDE_ARGS[@]}" \
      -w dist-repaired \
      "$whl"
  done
fi

# --- stamp the per-backend build tag (cpu / cuda12x / rocm6x) -------
"$PYEXE" scripts/tag_wheel.py "$BACKEND_TAG" dist-repaired dist-final

echo "==> final wheels:"
ls -1 dist-final/*.whl
