Installation
============

**torch-strumpack is cross-platform: it runs on Linux, Windows, and macOS.**
It ships **prebuilt, self-contained wheels on GitHub Releases** (it is **not**
published to PyPI). Each release carries one wheel per platform/backend; pick
the wheel that matches your operating system and hardware and install it by URL.

Browse every asset on the
`Releases page <https://github.com/sparsexlab/torch-strumpack/releases>`_.

Cross-platform wheel matrix
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 14 24 22 22 18

   * - OS / arch
     - Backend (build tag)
     - Python
     - GPU runtime
     - Wheel status
   * - **Linux** x86_64
     - CPU (``0_cpu``)
     - 3.10 / 3.11 / 3.12 / 3.13
     - none (CPU-only)
     - ✓ available
   * - **Linux** x86_64
     - NVIDIA CUDA 12.x (``0_cuda12x``)
     - 3.10 / 3.11 / 3.12 / 3.13
     - CUDA 12.x driver
     - ✓ available
   * - **Linux** x86_64
     - AMD ROCm 6.x (``0_rocm6x``)
     - 3.10 / 3.11 / 3.12 / 3.13
     - ROCm 6.x driver
     - ✓ available
   * - **Windows** x86_64
     - CPU (``0_cpu``)
     - 3.10 / 3.11 / 3.12 / 3.13
     - none (CPU-only)
     - building via CI
   * - **macOS** arm64
     - CPU (``0_cpu``)
     - 3.10 / 3.11 / 3.12 / 3.13
     - none (CPU-only)
     - ✓ available

The filename build tag (``0_cpu`` / ``0_cuda12x`` / ``0_rocm6x``) marks the
backend. CPU math dependencies (OpenBLAS / METIS / OpenMP runtime) are bundled
inside the wheel; the GPU runtime (CUDA / ROCm) is provided by your driver
install.

Per-platform support
--------------------

**Linux** — first-tier platform. CPU, NVIDIA CUDA 12.x, and AMD ROCm 6.x
wheels are published for CPython 3.10–3.13. STRUMPACK is built from source per
backend (GCC/gfortran + OpenBLAS/LAPACK + METIS, plus CUDA or HIP) and bundled
with ``auditwheel``.

**Windows** (x86_64, CPU) — supported. STRUMPACK builds natively on Windows
with the **clang-cl** C/C++ compiler and the **flang** Fortran compiler
(conda-forge LLVM toolchain), linked against MSVC-built PyTorch. The build is
proven on a real Windows box — a clean-environment solve reaches residual
``1.69e-16``. The prebuilt **Windows CPU wheel is being produced by CI** and
will appear on the Releases page alongside the Linux and macOS wheels; until it
lands you can also build it yourself with the same clang-cl + flang recipe (see
``.github/workflows/wheels.yml`` for the exact toolchain and CMake flags).

**macOS** (arm64, CPU) — supported. The CPU wheel is published for CPython
3.10–3.13 (clang + gfortran + OpenBLAS/METIS, bundled with ``delocate``).

torch ABI caveat — pick the matching wheel
------------------------------------------

The wheel is a compiled PyTorch C++ extension, so it is bound to a specific
**torch build ABI**. Install the wheel whose backend matches the torch you
have installed:

- The ``0_cpu`` wheel is built against **CPU PyTorch**. Install CPU torch first
  (or use a fresh ``torch+cpu`` environment) so ``pip`` does not pull a CUDA
  torch when resolving the ``torch`` dependency.
- The ``0_cuda12x`` wheel expects a **CUDA 12.x** torch build.
- The ``0_rocm6x`` wheel expects a **ROCm 6.x** torch build.

Mismatching the wheel's CUDA / ROCm against your installed torch will fail to
load. Match them.

Install examples
----------------

.. code-block:: bash

   # CPU (no GPU runtime needed), Linux x86_64, CPython 3.12
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install --no-deps \
     https://github.com/sparsexlab/torch-strumpack/releases/download/v0.0.1.dev0/torch_strumpack-0.0.1.dev0-0_cpu-cp312-cp312-manylinux_2_28_x86_64.whl

   # NVIDIA CUDA 12.x, Linux x86_64, CPython 3.12
   pip install --no-deps \
     https://github.com/sparsexlab/torch-strumpack/releases/download/v0.0.1.dev0/torch_strumpack-0.0.1.dev0-0_cuda12x-cp312-cp312-manylinux_2_28_x86_64.whl

   # AMD ROCm 6.x, Linux x86_64, CPython 3.12
   pip install --no-deps \
     https://github.com/sparsexlab/torch-strumpack/releases/download/v0.0.1.dev0/torch_strumpack-0.0.1.dev0-0_rocm6x-cp312-cp312-manylinux_2_35_x86_64.whl

Wheels are built for CPython 3.10 / 3.11 / 3.12 / 3.13 — swap ``cp312`` for
``cp310`` / ``cp311`` / ``cp313`` as needed.

Passing ``--no-deps`` is recommended once your matching torch is already
installed: it keeps ``pip`` from re-resolving (and possibly replacing) torch
when it installs the wheel.

Verify the install
------------------

After installing, confirm the compiled STRUMPACK extension actually loaded:

.. code-block:: python

   import torch_strumpack
   print(torch_strumpack.__version__)
   print(torch_strumpack.is_available())   # True only if the STRUMPACK ext loaded

:func:`torch_strumpack.is_available` returns ``True`` **only** when the real
compiled STRUMPACK extension is present — there is no scipy/other stand-in, so
``True`` always means STRUMPACK. If it returns ``False``, you likely installed
a wheel whose backend ABI does not match your torch; re-check the matrix above.
