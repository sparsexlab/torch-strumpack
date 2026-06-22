# torch-strumpack

Differentiable sparse **direct** solver for PyTorch, portable across **CPU / CUDA / ROCm**,
backed by [STRUMPACK](https://github.com/pghysels/STRUMPACK). Fills the AMD gap that
cuDSS (NVIDIA-only) leaves — a drop-in differentiable `A x = b` that also runs on Radeon / MI GPUs.

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
6. ⬜ Package wheels (`torch-strumpack` cpu / `-rocm` / `-cuda`).

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
