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
    """Whether the compiled STRUMPACK extension is loadable here.

    True only when the real solver is present -- there is no scipy/other
    stand-in, so a True here always means STRUMPACK.
    """
    return _core._ext is not None


__all__ = [
    "spsolve",
    "factor",
    "solve",
    "solve_transpose",
    "is_available",
    "__version__",
]
