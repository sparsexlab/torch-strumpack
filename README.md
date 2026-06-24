# torch-strumpack

Differentiable sparse **direct** solver for PyTorch, portable across **CPU / CUDA / ROCm**,
backed by [STRUMPACK](https://github.com/pghysels/STRUMPACK). Fills the AMD gap that
cuDSS (NVIDIA-only) leaves — a drop-in differentiable `A x = b` that also runs on Radeon / MI GPUs.

## Install

Wheels are published to **GitHub Releases** (not PyPI). Each release carries one
self-contained wheel per backend; pick the one matching your hardware and install
it by URL. CPU math deps (OpenBLAS / METIS / libgomp) are bundled; GPU runtimes
(CUDA / ROCm) are provided by your driver install.

The `0_cpu` wheel is built against **CPU PyTorch**. To keep `pip` from pulling a
CUDA torch when it resolves the `torch` dependency, install CPU torch from the
PyTorch CPU index first (or into a fresh `torch+cpu` venv):

```bash
# CPU (no GPU runtime needed), Linux x86_64, CPython 3.12
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install https://github.com/sparsexlab/torch-strumpack/releases/download/v0.0.1.dev0/torch_strumpack-0.0.1.dev0-0_cpu-cp312-cp312-manylinux_2_28_x86_64.whl

# NVIDIA CUDA 12.x, Linux x86_64, CPython 3.12
pip install https://github.com/sparsexlab/torch-strumpack/releases/download/v0.0.1.dev0/torch_strumpack-0.0.1.dev0-0_cuda12x-cp312-cp312-manylinux_2_28_x86_64.whl

# AMD ROCm 6.x, Linux x86_64, CPython 3.12
pip install https://github.com/sparsexlab/torch-strumpack/releases/download/v0.0.1.dev0/torch_strumpack-0.0.1.dev0-0_rocm6x-cp312-cp312-manylinux_2_35_x86_64.whl
```

Wheels are built for CPython 3.10 / 3.11 / 3.12 / 3.13 — swap `cp312` for
`cp310` / `cp311` / `cp313` as needed. Browse all assets on the
[Releases page](https://github.com/sparsexlab/torch-strumpack/releases). The
filename build tag (`0_cpu` / `0_cuda12x` / `0_rocm6x`) marks the backend.

Usable two ways:

- **Standalone** — `torch_strumpack.spsolve(A, b)`, autograd included.
- **As a torch-sla backend** — torch-sla owns autograd and drives the
  autograd-free primitives (`factor` / `solve` / `solve_transpose`). No
  double-differentiation.

```python
import torch
from torch_strumpack import spsolve

A = ...  # square sparse_csr tensor (or dense), requires_grad on values
b = torch.randn(A.shape[0], dtype=torch.float64, requires_grad=True)

x = spsolve(A, b)      # forward: factor + solve
x.sum().backward()     # backward: one transpose solve, reuses the factorization
```

## Status

**Real STRUMPACK, no fallback.** The solver is the compiled STRUMPACK extension and
nothing else — there is deliberately **no scipy/other stand-in**. If the extension is
missing, every call raises loudly, so you can never mistake a stand-in for STRUMPACK.
(scipy appears only in the test-suite, as a correctness oracle.)

Roadmap:
1. ✅ Architecture — pure primitives + autograd split, gradcheck-verified.
2. ✅ Real STRUMPACK **CPU** build (USE_HIP=OFF) — no ROCm dependency, gradcheck-verified.
3. ✅ Real STRUMPACK **ROCm** build (USE_HIP=ON, gfx1100) — GPU offload confirmed on
   Radeon 780M (gfx1103 via override), machine precision.
4. ✅ Real STRUMPACK **CUDA** build (USE_CUDA=ON, sm_89) — GPU offload confirmed on
   RTX 4070 Ti SUPER, machine precision. (WSL note: prepend `/usr/lib/wsl/lib` to
   `LD_LIBRARY_PATH` so the real libcuda is used, not the distro stub.)
5. ⬜ Multi-arch ROCm wheels: add gfx90a (MI200) / gfx942 (MI300), blind-compiled.
6. ✅ Package wheels — CI builds CPU / CUDA / ROCm wheels (py 3.10–3.12) and
   attaches them to the GitHub Release (see `.github/workflows/wheels.yml` and
   the **Install** section above).

One `CMakeLists.txt` covers all three: it finds whichever of ROCm / CUDA deps
exist (all `QUIET`) and links the STRUMPACK build you point it at.

## Design

```
spsolve / _SpSolve   ← autograd (only when standalone)
        │
   _core.factor / solve / solve_transpose   ← pure numeric primitives (no autograd)
        │
   STRUMPACK ext  (CPU / CUDA / ROCm)        ← swapped in for the SciPy stand-in
```

The `_core` layer is the backend contract. Keeping it autograd-free is what lets the
same code run standalone (autograd on top) or under torch-sla (torch-sla's autograd
calls `_core` directly).
