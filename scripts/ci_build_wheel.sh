#!/usr/bin/env bash
# Build the torch-strumpack binding wheel against an already-built STRUMPACK
# (STRUMPACK_DIR set by the caller), bundle the STRUMPACK + CPU math shared
# libs into the wheel with auditwheel, and stamp a per-backend build tag.
#
# Usage: ci_build_wheel.sh <backend-tag>     # e.g. cpu | cuda12x | rocm6x
#
# Env in:
#   STRUMPACK_DIR        CMake package dir of the STRUMPACK install
#   STRUMPACK_PREFIX     install prefix (for LD_LIBRARY_PATH; default /opt/strumpack)
#   PYBIN                python bin dir to use (manylinux); empty -> `python` on PATH
#   AUDITWHEEL_EXCLUDE   comma-separated libs to NOT vendor (GPU runtimes)
set -euo pipefail

BACKEND_TAG="${1:?usage: ci_build_wheel.sh <backend-tag>}"
PREFIX="${STRUMPACK_PREFIX:-/opt/strumpack}"

if [ -n "${PYBIN:-}" ]; then
  PYEXE="$PYBIN/python"
else
  PYEXE="$(command -v python3 || command -v python)"
fi
echo "==> python: $PYEXE ($($PYEXE --version 2>&1))"

"$PYEXE" -m pip install --upgrade pip wheel build auditwheel
# nanobind + torch headers are needed to compile the extension.
"$PYEXE" -m pip install nanobind "torch>=2.1" numpy

# Make the freshly built STRUMPACK + its math deps discoverable at link &
# repair time.
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"

# --- compile the extension into the package dir via CMake -----------
BUILD_DIR="build_ext"
rm -rf "$BUILD_DIR"
cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PYEXE" \
  -DSTRUMPACK_DIR="${STRUMPACK_DIR:?set STRUMPACK_DIR}"
cmake --build "$BUILD_DIR" -j"$(nproc)"
cmake --install "$BUILD_DIR"          # drops _strumpack_ext*.so into torch_strumpack/

echo "==> extension built:"
ls -1 torch_strumpack/*.so

# --- build the platform wheel (BinaryDistribution => platform tag) --
rm -rf dist
"$PYEXE" -m build --wheel --no-isolation

# --- bundle native libs into the wheel ------------------------------
EXCLUDE_ARGS=()
if [ -n "${AUDITWHEEL_EXCLUDE:-}" ]; then
  IFS=',' read -ra _ex <<< "$AUDITWHEEL_EXCLUDE"
  for lib in "${_ex[@]}"; do EXCLUDE_ARGS+=( --exclude "$lib" ); done
fi

# manylinux_2_28 inside the pypa image; the ROCm job builds on ubuntu 22.04
# (glibc 2.35) so it must target manylinux_2_35. Caller may override.
PLAT="${AUDITWHEEL_PLAT:-manylinux_2_28_x86_64}"

rm -rf dist-repaired dist-final
mkdir -p dist-repaired
for whl in dist/*.whl; do
  "$PYEXE" -m auditwheel repair \
    --plat "$PLAT" \
    "${EXCLUDE_ARGS[@]}" \
    -w dist-repaired \
    "$whl"
done

# --- stamp the per-backend build tag (cpu / cuda12x / rocm6x) -------
"$PYEXE" scripts/tag_wheel.py "$BACKEND_TAG" dist-repaired dist-final

echo "==> final wheels:"
ls -1 dist-final/*.whl
