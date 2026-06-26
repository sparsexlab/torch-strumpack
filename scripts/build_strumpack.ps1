# Build STRUMPACK (CPU, no-MPI, OpenMP) on Windows into $env:STRUMPACK_PREFIX.
#
# PROVEN RECIPE (validated on a real Windows box, tb16): STRUMPACK compiles on
# Windows with the conda-forge LLVM toolchain -- clang-cl for C/C++ and flang
# for Fortran -- linked against MSVC's link.exe + CRT, so the resulting DLL is
# ABI-compatible with MSVC-built PyTorch and the nanobind extension. A clean-env
# solve with the resulting wheel reaches residual 1.69e-16.
#
# This is NOT the old MSVC/MinGW/Intel-ifx story. It needs, from conda-forge:
#   cmake ninja flang flang_win-64 flang-rt_win-64 openblas liblapack metis llvm-openmp
# and an MSVC environment on PATH (for link.exe + the CRT).
#
# Inputs (env):
#   STRUMPACK_PREFIX   install prefix   (default: $PWD\strumpack-prefix)
#   STRUMPACK_REF      git ref/tag      (default: v8.0.0)
#   STRUMPACK_SRC      checkout dir     (default: $env:TEMP\STRUMPACK)
#   CONDA_PREFIX       conda env prefix (BLAS/LAPACK/METIS/flang-rt live here)
#   LLVM_BIN           dir with clang-cl.exe / flang.exe (default: $CONDA_PREFIX\Library\bin)
#
# Output: a STRUMPACK install at $STRUMPACK_PREFIX; its CMake package dir is
#   $STRUMPACK_PREFIX\lib\cmake\STRUMPACK  (point find_package(STRUMPACK) there).
$ErrorActionPreference = "Stop"

$Prefix  = if ($env:STRUMPACK_PREFIX) { $env:STRUMPACK_PREFIX } else { "$PWD\strumpack-prefix" }
$Ref     = if ($env:STRUMPACK_REF)    { $env:STRUMPACK_REF }    else { "v8.0.0" }
$Src     = if ($env:STRUMPACK_SRC)    { $env:STRUMPACK_SRC }    else { "$env:TEMP\STRUMPACK" }
$Conda   = $env:CONDA_PREFIX
if (-not $Conda) { throw "CONDA_PREFIX not set -- activate the conda env first" }
$LlvmBin = if ($env:LLVM_BIN) { $env:LLVM_BIN } else { "$Conda\Library\bin" }

# CMake bakes the values of -DCMAKE_PREFIX_PATH (and friends) verbatim into the
# *installed* strumpack-config.cmake (it re-exports them for transitive TPL
# discovery). A Windows path with backslashes -- e.g. C:\Miniconda\...\Library
# -- is then re-parsed by CMake when the extension build does
# find_package(STRUMPACK), and \M / \L / ... trip "Invalid character escape".
# CMake accepts forward slashes everywhere on Windows, so normalise every path
# we hand to cmake -D to forward slashes. ($CondaFwd is for the -D args;
# $Conda / $LlvmBin keep backslashes for native PowerShell Test-Path / IO.)
$CondaFwd = $Conda.Replace('\','/')

Write-Host "==> Building STRUMPACK ref=$Ref backend=cpu prefix=$Prefix (windows / clang-cl + flang)"
Write-Host "    CONDA_PREFIX=$Conda"
Write-Host "    LLVM_BIN=$LlvmBin"

if (-not (Test-Path "$Src\.git")) {
  git clone --depth 1 --branch $Ref https://github.com/pghysels/STRUMPACK.git $Src
  if ($LASTEXITCODE -ne 0) { throw "git clone STRUMPACK failed" }
}

# --- patch: strict clang-cl libc++/MSVC-STL headers do NOT transitively pull
# <numeric> for std::iota / std::partial_sum the way libstdc++ does. STRUMPACK
# v8.0.0 relies on that transitive include in several TUs, so add an explicit
# #include <numeric> at the top of each affected file. Idempotent.
$numericPatchTargets = @(
  "src\sparse\CSRGraph.cpp",
  "src\sparse\fronts\Front.cpp",
  "src\sparse\ordering\MatrixReordering.cpp",
  "src\clustering\Clustering.hpp",
  "src\dense\DenseMatrix.cpp",
  "src\BLR\BLRMatrix.cpp"
)
foreach ($rel in $numericPatchTargets) {
  $f = Join-Path $Src $rel
  if (-not (Test-Path $f)) {
    Write-Host "    (skip patch, not found: $rel)"
    continue
  }
  $content = Get-Content -Raw $f
  if ($content -match '(?m)^\s*#\s*include\s*<numeric>') {
    Write-Host "    already has <numeric>: $rel"
    continue
  }
  "#include <numeric>`r`n" + $content | Set-Content -NoNewline $f
  Write-Host "    patched (+#include <numeric>): $rel"
}

$ClangCl = "$LlvmBin\clang-cl.exe"
$Flang   = "$LlvmBin\flang.exe"
foreach ($exe in @($ClangCl, $Flang)) {
  if (-not (Test-Path $exe)) { throw "compiler not found: $exe" }
}

# MSVC link.exe (for the CRT + import libs). clang-cl needs the LLVM Windows
# runtime libs on the linker path (clang_rt.*); they live in clang's
# lib\windows. CMAKE_LINKER=link uses MSVC link with that extra LIBPATH.
$ClangLibWin = (Get-ChildItem -Recurse -Filter "clang_rt.builtins-x86_64.lib" "$Conda\Library" -ErrorAction SilentlyContinue |
                Select-Object -First 1)
$ClangLibDir = if ($ClangLibWin) { $ClangLibWin.Directory.FullName } else { "$Conda\Library\lib\clang\windows" }
Write-Host "    clang runtime lib dir: $ClangLibDir"

# forward-slash variants for every path handed to cmake -D (see $CondaFwd note)
$PrefixFwd      = $Prefix.Replace('\','/')
$ClangClFwd     = $ClangCl.Replace('\','/')
$FlangFwd       = $Flang.Replace('\','/')
$ClangLibDirFwd = $ClangLibDir.Replace('\','/')

$cmakeArgs = @(
  "-G", "Ninja",
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_INSTALL_PREFIX=$PrefixFwd",
  "-DCMAKE_C_COMPILER=$ClangClFwd",
  "-DCMAKE_CXX_COMPILER=$ClangClFwd",
  "-DCMAKE_Fortran_COMPILER=$FlangFwd",
  "-DCMAKE_LINKER=link",
  "-DCMAKE_EXE_LINKER_FLAGS=/LIBPATH:`"$ClangLibDirFwd`"",
  "-DCMAKE_SHARED_LINKER_FLAGS=/LIBPATH:`"$ClangLibDirFwd`"",
  "-DSTRUMPACK_USE_MPI=OFF",
  "-DSTRUMPACK_USE_OPENMP=ON",
  "-DTPL_ENABLE_METIS=ON",
  "-DTPL_ENABLE_SCOTCH=OFF",
  "-DTPL_ENABLE_PARMETIS=OFF",
  "-DTPL_ENABLE_PTSCOTCH=OFF",
  "-DSTRUMPACK_USE_CUDA=OFF",
  "-DSTRUMPACK_USE_HIP=OFF",
  "-DBUILD_SHARED_LIBS=ON",
  # STRUMPACK has no __declspec(dllexport) annotations, so on Windows nothing is
  # exported and no import lib (strumpack.lib) is produced -- the nanobind
  # extension would then fail to link. Auto-export all symbols.
  "-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON",
  # point BLAS/LAPACK/METIS at the conda env (forward slashes: these get baked
  # into the installed strumpack-config.cmake and re-parsed by find_package)
  "-DCMAKE_PREFIX_PATH=$CondaFwd/Library",
  "-DTPL_METIS_INCLUDE_DIRS=$CondaFwd/Library/include",
  "-DTPL_METIS_LIBRARIES=$CondaFwd/Library/lib/metis.lib",
  "-DBLAS_LIBRARIES=$CondaFwd/Library/lib/openblas.lib",
  "-DLAPACK_LIBRARIES=$CondaFwd/Library/lib/openblas.lib"
)

Remove-Item -Recurse -Force "$Src\build" -ErrorAction SilentlyContinue
cmake -S $Src -B "$Src\build" @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
cmake --build "$Src\build" -j
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }
cmake --install "$Src\build"
if ($LASTEXITCODE -ne 0) { throw "cmake install failed" }

# STRUMPACK's install() rules were written for ELF/Mach-O and on Windows omit
# the runtime strumpack.dll from the install tree (only the .lib import lib is
# installed). Copy the DLL into the prefix bin/ so the wheel can bundle it and
# the extension can load it at runtime.
$dll = Get-ChildItem -Recurse -Filter "strumpack.dll" "$Src\build" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($dll) {
  New-Item -ItemType Directory -Force "$Prefix\bin" | Out-Null
  Copy-Item $dll.FullName "$Prefix\bin\strumpack.dll" -Force
  Write-Host "    copied $($dll.FullName) -> $Prefix\bin\strumpack.dll"
} else {
  Write-Warning "strumpack.dll not found under $Src\build -- the wheel may fail to load"
}

Write-Host "==> STRUMPACK installed. STRUMPACK_DIR=$Prefix\lib\cmake\STRUMPACK"
Get-ChildItem "$Prefix\lib","$Prefix\bin" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "strumpack" } | ForEach-Object { Write-Host "    $($_.FullName)" }
