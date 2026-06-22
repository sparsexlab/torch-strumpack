import torch

from torch_strumpack import spsolve
from torch_strumpack.autograd import _SpSolve

torch.manual_seed(0)


def _spd(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    return M @ M.t() + n * torch.eye(n, dtype=torch.float64)


def test_forward_matches_dense_solve():
    n = 8
    A = _spd(n)
    b = torch.randn(n, dtype=torch.float64)
    x = spsolve(A.to_sparse_csr(), b)
    x_ref = torch.linalg.solve(A, b)
    assert torch.allclose(x, x_ref, atol=1e-10)


def test_forward_multi_rhs():
    n, k = 7, 4
    A = _spd(n, seed=3)
    B = torch.randn(n, k, dtype=torch.float64)
    X = spsolve(A.to_sparse_csr(), B)
    assert torch.allclose(X, torch.linalg.solve(A, B), atol=1e-10)


def _apply(crow, col, n):
    return lambda v, b: _SpSolve.apply(v, b, crow, col, n)


def test_gradcheck_single_rhs():
    n = 6
    Acsr = _spd(n, seed=1).to_sparse_csr()
    crow, col = Acsr.crow_indices(), Acsr.col_indices()
    vals = Acsr.values().clone().requires_grad_(True)
    b = torch.randn(n, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_apply(crow, col, n), (vals, b))


def test_gradcheck_multi_rhs():
    n, k = 5, 3
    Acsr = _spd(n, seed=2).to_sparse_csr()
    crow, col = Acsr.crow_indices(), Acsr.col_indices()
    vals = Acsr.values().clone().requires_grad_(True)
    b = torch.randn(n, k, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(_apply(crow, col, n), (vals, b))
