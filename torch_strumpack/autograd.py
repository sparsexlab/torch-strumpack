"""Thin autograd layer for standalone use.

This is the **only** place that knows about gradients. When torch-strumpack
is used by itself, this layer owns the adjoint of the linear solve. When it
is consumed as a torch-sla backend, torch-sla owns autograd instead and
calls ``_core`` directly -- so this module simply isn't used, and there is
no double-differentiation.

Adjoint of  x = A^{-1} b :
    dL/db = A^{-T} (dL/dx)                      (one transpose solve)
    dL/dA_ij = -(dL/db)_i * x_j   on the sparsity pattern
"""

from __future__ import annotations

import torch

from . import _core


class _SpSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, b, crow, col, n):
        fac = _core.factor(crow, col, values, n)
        x = _core.solve(fac, b)
        ctx.fac = fac
        ctx.n = n
        ctx.b_dim = b.dim()
        ctx.save_for_backward(x, crow, col)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        x, crow, col = ctx.saved_tensors
        # dL/db = A^{-T} grad_x  -- reuse the cached factorization
        grad_b = _core.solve_transpose(ctx.fac, grad_x)

        # expand CSR row pointers to a per-nnz row index
        counts = (crow[1:] - crow[:-1]).to(torch.long)
        rows = torch.repeat_interleave(
            torch.arange(ctx.n, device=col.device), counts
        )
        if ctx.b_dim == 1:
            grad_vals = -(grad_b[rows] * x[col])
        else:  # multiple right-hand sides: sum over the rhs columns
            grad_vals = -(grad_b[rows, :] * x[col, :]).sum(dim=1)

        return grad_vals, grad_b, None, None, None


def spsolve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable sparse direct solve ``A x = b``.

    Parameters
    ----------
    A : torch.Tensor
        Square matrix, sparse CSR (preferred) or dense.
    b : torch.Tensor
        Right-hand side, shape ``(n,)`` or ``(n, k)``.
    """
    if A.layout != torch.sparse_csr:
        A = A.to_sparse_csr()
    return _SpSolve.apply(
        A.values(), b, A.crow_indices(), A.col_indices(), A.shape[0]
    )
