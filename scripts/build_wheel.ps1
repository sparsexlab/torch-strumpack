# Build the torch-strumpack nanobind wheel on Windows against a prebuilt
# STRUMPACK ($env:STRUMPACK_DIR), bundle the DLLs with delvewheel, and stamp
# the per-backend build tag via scripts/tag_wheel.py.
#
# PROVEN RECIPE (tb16): the extension is compiled with clang-cl (the same
# compiler STRUMPACK was built with) against the conda-forge STRUMPACK build and
# MSVC-built CPU torch. delvewheel then vendors strumpack/openblas/metis/libomp
# while EXCLUDING torch's own DLLs (they ship with torch and must not be
# duplicated). A per-backend build tag (0_cpu) keeps the windows wheel distinct
# from the linux/macOS Release assets.
#
# Inputs (env):
#   STRUMPACK_DIR      CMake package dir of the STRUMPACK install (required)
#   STRUMPACK_PREFIX   install prefix (for DLL bundling; default $PWD\strumpack-prefix)
#   CONDA_PREFIX       conda env prefix (openblas/metis/libomp DLLs live here)
#   LLVM_BIN           dir with clang-cl.exe (default $CONDA_PREFIX\Library\bin)
#   TORCH_INDEX_URL    pin the torch wheel index (e.g. .../whl/cpu)
#
# Usage: pwsh scripts/build_wheel.ps1 <backend-tag>     # e.g. cpu
param([string]$BackendTag = "cpu")
$ErrorActionPreference = "Stop"

$Prefix = if ($env:STRUMPACK_PREFIX) { $env:STRUMPACK_PREFIX } else { "$PWD\strumpack-prefix" }
if (-not $env:STRUMPACK_DIR) { throw "set STRUMPACK_DIR" }
$Conda  = $env:CONDA_PREFIX
if (-not $Conda) { throw "CONDA_PREFIX not set -- activate the conda env first" }
$LlvmBin = if ($env:LLVM_BIN) { $env:LLVM_BIN } else { "$Conda\Library\bin" }
$ClangCl = "$LlvmBin\clang-cl.exe"

python -m pip install --upgrade pip wheel build delvewheel
if ($LASTEXITCODE -ne 0) { throw "pip install build tools failed" }
python -m pip install nanobind numpy
if ($LASTEXITCODE -ne 0) { throw "pip install nanobind/numpy failed" }
if ($env:TORCH_INDEX_URL) {
  python -m pip install --index-url $env:TORCH_INDEX_URL "torch>=2.1"
} else {
  python -m pip install "torch>=2.1"
}
if ($LASTEXITCODE -ne 0) { throw "pip install torch failed" }

# compile the extension (_strumpack_ext.pyd) into torch_strumpack/ with clang-cl
$PyExe = (Get-Command python).Source
Write-Host "==> python: $PyExe"
Write-Host "==> clang-cl: $ClangCl"
# Forward-slash every path handed to cmake -D. On Windows, backslash paths are
# re-parsed as escape sequences by CMake (e.g. \M in C:\Miniconda) -> "Invalid
# character escape". CMake accepts '/' everywhere on Windows.
$PyExeFwd       = $PyExe.Replace('\','/')
$ClangClFwd     = $ClangCl.Replace('\','/')
$CondaFwd       = $Conda.Replace('\','/')
$StrumpackDirFwd = $env:STRUMPACK_DIR.Replace('\','/')
# Build the cmake args as an array and splat. (Backtick-continued inline args
# with a $(...) subexpression mis-parse under pwsh and pass e.g. the literal
# string '$ClangCl' through to CMake -> 'is not a full path'.)
$extArgs = @(
  "-S", ".",
  "-B", "build_ext",
  "-G", "Ninja",
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_C_COMPILER=$ClangClFwd",
  "-DCMAKE_CXX_COMPILER=$ClangClFwd",
  "-DCMAKE_LINKER=link",
  "-DPython_EXECUTABLE=$PyExeFwd",
  "-DSTRUMPACK_DIR=$StrumpackDirFwd",
  "-DCMAKE_PREFIX_PATH=$CondaFwd/Library"
)
Remove-Item -Recurse -Force build_ext -ErrorAction SilentlyContinue
cmake @extArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure (extension) failed" }
cmake --build build_ext -j
if ($LASTEXITCODE -ne 0) { throw "cmake build (extension) failed" }
cmake --install build_ext
if ($LASTEXITCODE -ne 0) { throw "cmake install (extension) failed" }

Write-Host "==> extension built:"
Get-ChildItem torch_strumpack\*.pyd | ForEach-Object { Write-Host "    $($_.Name)" }

# platform wheel (BinaryDistribution => win_amd64 platform tag + bundled .pyd)
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
python -m build --wheel --no-isolation
if ($LASTEXITCODE -ne 0) { throw "python -m build failed" }

# Bundle STRUMPACK + openblas + metis + libomp DLLs. EXCLUDE torch's own DLLs:
# they ship with the user's torch install and must not be duplicated into the
# wheel (doing so re-introduces the OpenMP double-runtime conflict and bloats
# the wheel). delvewheel stamps the win_amd64 platform tag.
$libdirs = "$Prefix\bin;$Prefix\lib;$Conda\Library\bin"
$exclude = @(
  "c10.dll","torch.dll","torch_cpu.dll","torch_python.dll",
  "c10_cuda.dll","torch_cuda.dll","fbgemm.dll","asmjit.dll",
  "uv.dll","libiomp5md.dll","shm.dll"
)
Remove-Item -Recurse -Force dist-repaired, dist-final -ErrorAction SilentlyContinue
New-Item -ItemType Directory dist-repaired | Out-Null
Get-ChildItem dist\*.whl | ForEach-Object {
  python -m delvewheel repair --add-path $libdirs --exclude ($exclude -join ";") -w dist-repaired $_.FullName
  if ($LASTEXITCODE -ne 0) { throw "delvewheel repair failed" }
}

# per-backend build tag (so cpu/cuda/rocm + OS variants don't collide)
python scripts/tag_wheel.py $BackendTag dist-repaired dist-final
if ($LASTEXITCODE -ne 0) { throw "tag_wheel failed" }
Write-Host "==> final wheels:"
Get-ChildItem dist-final\*.whl | ForEach-Object { Write-Host "    $($_.Name)" }
