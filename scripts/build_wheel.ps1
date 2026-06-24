# Build the torch-strumpack nanobind wheel on Windows against a prebuilt
# STRUMPACK ($env:STRUMPACK_DIR), bundle the DLLs with delvewheel, and stamp
# the per-backend build tag via scripts/tag_wheel.py.
#
# STATUS: SCAFFOLD / NOT YET WORKING — depends on a working STRUMPACK Windows
# build (see scripts/build_strumpack.ps1). Not run by CI. The Windows job in
# wheels.yml is disabled until the Fortran toolchain story is resolved.
#
# Usage: pwsh scripts/build_wheel.ps1 <backend-tag>     # e.g. cpu
$ErrorActionPreference = "Stop"
param([string]$BackendTag = "cpu")

$Prefix = if ($env:STRUMPACK_PREFIX) { $env:STRUMPACK_PREFIX } else { "$PWD\strumpack-prefix" }
if (-not $env:STRUMPACK_DIR) { throw "set STRUMPACK_DIR" }

python -m pip install --upgrade pip wheel build delvewheel
python -m pip install nanobind numpy
if ($env:TORCH_INDEX_URL) {
  python -m pip install --index-url $env:TORCH_INDEX_URL "torch>=2.1"
} else {
  python -m pip install "torch>=2.1"
}

# compile the extension (_strumpack_ext.pyd) into torch_strumpack/
Remove-Item -Recurse -Force build_ext -ErrorAction SilentlyContinue
cmake -S . -B build_ext -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DPython_EXECUTABLE=(Get-Command python).Source `
  -DSTRUMPACK_DIR=$env:STRUMPACK_DIR
cmake --build build_ext -j
cmake --install build_ext

# platform wheel
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
python -m build --wheel --no-isolation

# bundle the STRUMPACK + openblas + metis DLLs; delvewheel stamps the
# win_amd64 platform tag so the windows wheel is a distinct Release asset.
Remove-Item -Recurse -Force dist-repaired, dist-final -ErrorAction SilentlyContinue
New-Item -ItemType Directory dist-repaired | Out-Null
$libdirs = "$Prefix\bin;$Prefix\lib;$env:VCPKG_ROOT\installed\x64-windows\bin"
Get-ChildItem dist\*.whl | ForEach-Object {
  python -m delvewheel repair --add-path $libdirs -w dist-repaired $_.FullName
}

# per-backend build tag (so cpu/cuda/rocm + OS variants don't collide)
python scripts/tag_wheel.py $BackendTag dist-repaired dist-final
Get-ChildItem dist-final\*.whl
