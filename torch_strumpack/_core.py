"""Pure numeric primitives backed by the compiled STRUMPACK extension.

NO autograd, NO scipy, NO silent fallback. This package solves with STRUMPACK
and only STRUMPACK. If the compiled extension is not present, every entry point
raises loudly -- so it is impossible to accidentally run on a stand-in and think
you got STRUMPACK.

The extension (`_strumpack_ext`) is built per platform (cpu / rocm / cuda) and
shipped inside the matching wheel.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from . import _strumpack_ext as _ext
    _EXT_ERR = None
except Exception as e:  # pragma: no cover - import-time probe
    _ext = None
    _EXT_ERR = e


def _require_ext():
    if _ext is None:
        raise RuntimeError(
            "torch-strumpack: the compiled STRUMPACK extension is not available "
            f"({_EXT_ERR!r}).\n"
            "This package solves with STRUMPACK ONLY -- there is deliberately no "
            "scipy/other fallback, so you can never mistake a stand-in for STRUMPACK.\n"
            "Install a torch-strumpack wheel built for your platform "
            "(cpu / rocm / cuda), or build the extension from source (see CMakeLists.txt)."
        )
    return _ext


def backend() -> str:
    """Identifier of the live solver. Always 'strumpack' when usable."""
    return _require_ext().backend()


class Factorization:
    """Opaque STRUMPACK factorization handle (reused across solves, and for the
    transpose solve in a backward pass)."""

    __slots__ = ("_ef", "crow", "col", "values", "n", "dtype", "device", "_ef_t")

    def __init__(self, ef, crow, col, values, n):
        self._ef = ef
        self.crow, self.col, self.values, self.n = crow, col, values, n
        self.dtype, self.device = values.dtype, values.device
        self._ef_t = None  # lazy transpose factorization


def _np_csr(crow, col, values):
    return (
        crow.detach().cpu().numpy().astype(np.int32),
        col.detach().cpu().numpy().astype(np.int32),
        values.detach().cpu().to(torch.float64).numpy(),
    )


def factor(crow, col, values, n) -> Factorization:
    ext = _require_ext()
    indptr, indices, data = _np_csr(crow, col, values)
    return Factorization(ext.factorize(indptr, indices, data, n), crow, col, values, n)


def _back(x, fac):
    return torch.from_numpy(x).to(device=fac.device, dtype=fac.dtype)


def solve(fac: Factorization, b: torch.Tensor) -> torch.Tensor:
    ext = _require_ext()
    return _back(ext.solve(fac._ef, b.detach().cpu().to(torch.float64).numpy()), fac)


def solve_transpose(fac: Factorization, b: torch.Tensor) -> torch.Tensor:
    """Solve A^T y = b. Factors A^T once (lazily) from the stored pattern -- no
    scipy, the transpose is built with torch and factored by STRUMPACK."""
    ext = _require_ext()
    if fac._ef_t is None:
        At = (
            torch.sparse_csr_tensor(fac.crow, fac.col, fac.values, (fac.n, fac.n))
            .to_sparse_coo().t().coalesce().to_sparse_csr()
        )
        indptr, indices, data = _np_csr(At.crow_indices(), At.col_indices(), At.values())
        fac._ef_t = ext.factorize(indptr, indices, data, fac.n)
    return _back(ext.solve(fac._ef_t, b.detach().cpu().to(torch.float64).numpy()), fac)
