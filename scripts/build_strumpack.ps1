# Build STRUMPACK (CPU, no-MPI, OpenMP) on Windows into $env:STRUMPACK_PREFIX.
#
# STATUS: SCAFFOLD / NOT YET WORKING. STRUMPACK is project(... LANGUAGES CXX C
# Fortran) and compiles bundled LAPACK .f sources, so a Fortran compiler is
# mandatory. MSVC has none. The two viable paths are:
#   (a) Intel oneAPI `ifx` Fortran + MSVC C/C++  (ABI-compatible with MSVC torch)
#   (b) a full MinGW-w64 gfortran/g++ toolchain   (then the nanobind extension
#       must ALSO be built with MinGW so it links the MinGW STRUMPACK)
# Both are non-trivial; this file exists so the Windows job in wheels.yml can be
# re-enabled once one of them is wired up. It is intentionally not run by CI.
$ErrorActionPreference = "Stop"

$Backend = if ($env:STRUMPACK_BACKEND) { $env:STRUMPACK_BACKEND } else { "cpu" }
$Prefix  = if ($env:STRUMPACK_PREFIX)  { $env:STRUMPACK_PREFIX }  else { "$PWD\strumpack-prefix" }
$Ref     = if ($env:STRUMPACK_REF)     { $env:STRUMPACK_REF }     else { "v8.0.0" }
$Src     = if ($env:STRUMPACK_SRC)     { $env:STRUMPACK_SRC }     else { "$env:TEMP\STRUMPACK" }

Write-Host "==> Building STRUMPACK ref=$Ref backend=$Backend prefix=$Prefix (windows)"

if (-not (Test-Path "$Src\.git")) {
  git clone --depth 1 --branch $Ref https://github.com/pghysels/STRUMPACK.git $Src
}

# vcpkg provides openblas (BLAS+LAPACK) and metis; expect $env:VCPKG_ROOT set,
# with the vcpkg CMake toolchain file used so find_package(BLAS/METIS) resolves.
$Toolchain = "$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"

$cmakeArgs = @(
  "-G", "Ninja",
  "-DCMAKE_BUILD_TYPE=Release",
  "-DCMAKE_INSTALL_PREFIX=$Prefix",
  "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
  "-DSTRUMPACK_USE_MPI=OFF",
  "-DSTRUMPACK_USE_OPENMP=ON",
  "-DTPL_ENABLE_METIS=ON",
  "-DTPL_ENABLE_SCOTCH=OFF",
  "-DTPL_ENABLE_PARMETIS=OFF",
  "-DTPL_ENABLE_PTSCOTCH=OFF",
  "-DBUILD_SHARED_LIBS=ON",
  "-DSTRUMPACK_USE_CUDA=OFF",
  "-DSTRUMPACK_USE_HIP=OFF"
  # TODO: select the Fortran compiler, e.g. -DCMAKE_Fortran_COMPILER=ifx
  #       (Intel oneAPI) and ensure it is ABI-compatible with the MSVC build.
)

Remove-Item -Recurse -Force "$Src\build" -ErrorAction SilentlyContinue
cmake -S $Src -B "$Src\build" @cmakeArgs
cmake --build "$Src\build" -j
cmake --install "$Src\build"
Write-Host "STRUMPACK_DIR=$Prefix\lib\cmake\STRUMPACK"
