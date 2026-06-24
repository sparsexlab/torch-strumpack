#!/usr/bin/env bash
# Build the native STRUMPACK sparse direct solver for one backend, into an
# install prefix the wheel build then links against (via STRUMPACK_DIR).
#
# This is the heavy native dependency for torch-strumpack (analogous to how
# torch-amgx vendors+builds AmgX). One recipe, three backends, selected by
# $STRUMPACK_BACKEND:
#
#   cpu  -> OpenBLAS/LAPACK + METIS, no GPU (USE_CUDA=OFF, USE_HIP=OFF)
#   cuda -> + CUDA offload (USE_CUDA=ON, needs a CUDA toolkit on PATH)
#   rocm -> + HIP/ROCm offload (USE_HIP=ON, needs ROCm at $ROCM_PATH)
#
# Works on Linux and macOS (Homebrew gcc/gfortran + openblas + metis). STRUMPACK
# hard-requires a Fortran compiler (project(... LANGUAGES CXX C Fortran) and it
# compiles bundled LAPACK .f sources), so on macOS we must use Homebrew gcc's
# gfortran and tell CMake to use it.
#
# Inputs (env):
#   STRUMPACK_BACKEND   cpu | cuda | rocm           (default: cpu)
#   STRUMPACK_PREFIX    install prefix              (default: /opt/strumpack)
#   STRUMPACK_REF       git ref/tag to build        (default: v8.0.0)
#   CUDA_ARCHS          e.g. "80;89;90"             (cuda only)
#   HIP_ARCHS           e.g. "gfx90a;gfx942;gfx1100" (rocm only)
#   CMAKE_Fortran_COMPILER  override the Fortran compiler (e.g. gfortran-14)
#   STRUMPACK_BLAS_LIB / STRUMPACK_LAPACK_LIB  explicit BLAS/LAPACK libs
#
# Output: a STRUMPACK install at $STRUMPACK_PREFIX whose CMake package dir is
#   $STRUMPACK_PREFIX/lib/cmake/STRUMPACK  (point find_package(STRUMPACK) there).
set -euo pipefail

BACKEND="${STRUMPACK_BACKEND:-cpu}"
PREFIX="${STRUMPACK_PREFIX:-/opt/strumpack}"
REF="${STRUMPACK_REF:-v8.0.0}"
SRC="${STRUMPACK_SRC:-/tmp/STRUMPACK}"

UNAME="$(uname -s)"
# Parallel jobs: nproc on Linux, sysctl on macOS.
if command -v nproc >/dev/null 2>&1; then
  NJOBS="$(nproc)"
elif [ "$UNAME" = "Darwin" ]; then
  NJOBS="$(sysctl -n hw.ncpu)"
else
  NJOBS="4"
fi

echo "==> Building STRUMPACK ref=$REF backend=$BACKEND prefix=$PREFIX (os=$UNAME, jobs=$NJOBS)"

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$REF" https://github.com/pghysels/STRUMPACK.git "$SRC"
fi

CMAKE_ARGS=(
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DSTRUMPACK_USE_MPI=OFF          # sequential build; no MPI on CI runners
  -DSTRUMPACK_USE_OPENMP=ON
  -DTPL_ENABLE_METIS=ON
  -DTPL_ENABLE_SCOTCH=OFF
  -DTPL_ENABLE_PARMETIS=OFF
  -DTPL_ENABLE_PTSCOTCH=OFF
  -DBUILD_SHARED_LIBS=ON
)

# A Fortran compiler is mandatory. On macOS the default `gfortran` may be
# missing while Homebrew installs versioned binaries (gfortran-14); let the
# caller pin it, otherwise auto-detect a Homebrew gfortran.
if [ -n "${CMAKE_Fortran_COMPILER:-}" ]; then
  CMAKE_ARGS+=( -DCMAKE_Fortran_COMPILER="$CMAKE_Fortran_COMPILER" )
elif [ "$UNAME" = "Darwin" ] && ! command -v gfortran >/dev/null 2>&1; then
  GF="$(ls -1 "$(brew --prefix 2>/dev/null)"/bin/gfortran-* 2>/dev/null | sort -V | tail -1 || true)"
  if [ -n "$GF" ]; then
    echo "==> using Homebrew Fortran: $GF"
    CMAKE_ARGS+=( -DCMAKE_Fortran_COMPILER="$GF" )
  fi
fi

# On macOS, point CMake at the Homebrew OpenBLAS + METIS (keg-only / not on the
# default search path). OpenBLAS ships LAPACK too.
if [ "$UNAME" = "Darwin" ]; then
  BREW_PREFIX="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
  OPENBLAS_PREFIX="$(brew --prefix openblas 2>/dev/null || echo "$BREW_PREFIX/opt/openblas")"
  METIS_PREFIX="$(brew --prefix metis 2>/dev/null || echo "$BREW_PREFIX/opt/metis")"
  CMAKE_ARGS+=(
    -DCMAKE_PREFIX_PATH="$OPENBLAS_PREFIX;$METIS_PREFIX;$BREW_PREFIX"
    -DBLAS_LIBRARIES="$OPENBLAS_PREFIX/lib/libopenblas.dylib"
    -DLAPACK_LIBRARIES="$OPENBLAS_PREFIX/lib/libopenblas.dylib"
    -DTPL_METIS_INCLUDE_DIRS="$METIS_PREFIX/include"
    -DTPL_METIS_LIBRARIES="$METIS_PREFIX/lib/libmetis.dylib"
  )
  # Bake the deployment target into STRUMPACK so its dylibs match the wheel.
  if [ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ]; then
    CMAKE_ARGS+=( -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" )
  fi
fi

case "$BACKEND" in
  cpu)
    CMAKE_ARGS+=( -DSTRUMPACK_USE_CUDA=OFF -DSTRUMPACK_USE_HIP=OFF )
    ;;
  cuda)
    CMAKE_ARGS+=(
      -DSTRUMPACK_USE_CUDA=ON
      -DSTRUMPACK_USE_HIP=OFF
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS:-70;80;89;90}"
    )
    ;;
  rocm)
    : "${ROCM_PATH:=/opt/rocm}"
    export ROCM_PATH
    CMAKE_ARGS+=(
      -DSTRUMPACK_USE_CUDA=OFF
      -DSTRUMPACK_USE_HIP=ON
      -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCHS:-gfx90a;gfx942;gfx1100}"
      -DCMAKE_PREFIX_PATH="$ROCM_PATH"
    )
    ;;
  *)
    echo "unknown STRUMPACK_BACKEND='$BACKEND' (want cpu|cuda|rocm)" >&2
    exit 2
    ;;
esac

rm -rf "$SRC/build"
cmake -S "$SRC" -B "$SRC/build" "${CMAKE_ARGS[@]}"
cmake --build "$SRC/build" -j"$NJOBS"
cmake --install "$SRC/build"

echo "==> STRUMPACK installed:"
ls -1 "$PREFIX/lib" 2>/dev/null | grep -i strumpack || \
  ls -1 "$PREFIX/lib64" 2>/dev/null | grep -i strumpack || true
echo "STRUMPACK_DIR=$PREFIX/lib/cmake/STRUMPACK"
