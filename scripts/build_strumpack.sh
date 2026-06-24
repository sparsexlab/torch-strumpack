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
# Inputs (env):
#   STRUMPACK_BACKEND   cpu | cuda | rocm           (default: cpu)
#   STRUMPACK_PREFIX    install prefix              (default: /opt/strumpack)
#   STRUMPACK_REF       git ref/tag to build        (default: v8.0.0)
#   CUDA_ARCHS          e.g. "80;89;90"             (cuda only)
#   HIP_ARCHS           e.g. "gfx90a;gfx942;gfx1100" (rocm only)
#
# Output: a STRUMPACK install at $STRUMPACK_PREFIX whose CMake package dir is
#   $STRUMPACK_PREFIX/lib/cmake/STRUMPACK  (point find_package(STRUMPACK) there).
set -euo pipefail

BACKEND="${STRUMPACK_BACKEND:-cpu}"
PREFIX="${STRUMPACK_PREFIX:-/opt/strumpack}"
REF="${STRUMPACK_REF:-v8.0.0}"
SRC="${STRUMPACK_SRC:-/tmp/STRUMPACK}"

echo "==> Building STRUMPACK ref=$REF backend=$BACKEND prefix=$PREFIX"

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
cmake --build "$SRC/build" -j"$(nproc)"
cmake --install "$SRC/build"

echo "==> STRUMPACK installed:"
ls -1 "$PREFIX/lib" | grep -i strumpack || true
echo "STRUMPACK_DIR=$PREFIX/lib/cmake/STRUMPACK"
