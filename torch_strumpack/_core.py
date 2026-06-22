"""Pure numeric primitives for sparse direct solve.

This layer has **no autograd** and no knowledge of torch-sla. It is the
backend contract that torch-sla (or any caller) drives:

    fac = factor(crow, col, values, n)
    x   = solve(fac, b)
    y   = solve_transpose(fac, g)

In this skeleton the actual factorization is done by SciPy SuperLU as a
stand-in. The real package will swap ``factor/solve/solve_transpose`` for
a compiled STRUMPACK extension (CPU / CUDA / ROCm) -- the signatures here
are exactly what that extension must expose, so nothing above this layer
changes when the real solver lands.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


class Factorization:
    """Opaque factorization handle. Reused across solve / solve_transpose
    (and, crucially, between an autograd forward and its backward)."""

    __slots__ = ("_lu", "n", "dtype", "device")

    def __init__(self, lu, n, dtype, device):
        self._lu = lu
        self.n = n
        self.dtype = dtype
        self.device = device


def _csr_to_csc(crow, col, values, n):
    crow_np = crow.detach().cpu().numpy()
    col_np = col.detach().cpu().numpy()
    val_np = values.detach().cpu().numpy()
    A = sp.csr_matrix((val_np, col_np, crow_np), shape=(n, n))
    return A.tocsc()


def factor(crow, col, values, n) -> Factorization:
    """Symbolic + numeric factorization of a CSR matrix."""
    A = _csr_to_csc(crow, col, values, n)
    lu = spla.splu(A)
    return Factorization(lu, n, values.dtype, values.device)


def _back(arr, fac) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr)).to(
        device=fac.device, dtype=fac.dtype
    )


def solve(fac: Factorization, b: torch.Tensor) -> torch.Tensor:
    """Solve A x = b. b may be (n,) or (n, k)."""
    b_np = b.detach().cpu().numpy()
    return _back(fac._lu.solve(b_np), fac)


def solve_transpose(fac: Factorization, b: torch.Tensor) -> torch.Tensor:
    """Solve A^T y = b. Reuses the same factorization (no refactor)."""
    b_np = b.detach().cpu().numpy()
    return _back(fac._lu.solve(b_np, trans="T"), fac)
