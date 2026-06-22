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

**Architecture skeleton.** The numeric core is currently a SciPy SuperLU stand-in so the
API, the primitive/autograd split, and the adjoint are validated (`pytest` + `gradcheck`).
The compiled STRUMPACK extension drops in behind the same `_core` signatures with no change
above it.

Roadmap:
1. ✅ CPU skeleton — pure primitives + autograd, gradcheck-verified.
2. ⬜ Real STRUMPACK (CPU).
3. ⬜ CUDA build (validate on NVIDIA).
4. ⬜ ROCm build — local debug on RDNA3 (Radeon 780M, gfx1103 via override),
   blind multi-arch wheels for MI200 (gfx90a) / MI300 (gfx942).

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
