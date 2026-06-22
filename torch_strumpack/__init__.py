"""torch-strumpack: differentiable sparse direct solver for PyTorch.

Portable across CPU / CUDA / ROCm. Designed to be used either standalone
(``torch_strumpack.spsolve``) or as a torch-sla backend (which drives the
autograd-free primitives in :mod:`torch_strumpack._core`).

NOTE: this is the architecture skeleton. The numeric core is currently a
SciPy SuperLU stand-in; the compiled STRUMPACK extension drops in behind
the same ``_core`` signatures with no change above it.
"""

from __future__ import annotations

from . import _core
from .autograd import spsolve

__version__ = "0.0.1.dev0"

# Pure primitives that a torch-sla backend adapter should call.
factor = _core.factor
solve = _core.solve
solve_transpose = _core.solve_transpose


def is_available() -> bool:
    """Whether the solver can run in this environment.

    Mirrors ``torch_amgx.is_available()`` so a torch-sla backend adapter can
    gate on it the same way. The real package will probe the compiled
    extension and the active device (CUDA vs ROCm via ``torch.version.hip``).
    """
    try:
        import scipy.sparse.linalg  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = [
    "spsolve",
    "factor",
    "solve",
    "solve_transpose",
    "is_available",
    "__version__",
]
