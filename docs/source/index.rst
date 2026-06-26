torch-strumpack
===============

**torch-strumpack** is a differentiable sparse **direct** solver for PyTorch,
backed by `STRUMPACK <https://github.com/pghysels/STRUMPACK>`_ (a sparse
multifrontal direct solver). It is a torch-native ``A x = b`` backend that is
**portable across CPU / CUDA / ROCm** — the direct path that closes the
**AMD / ROCm gap** that NVIDIA-only solvers such as cuDSS leave open. It
supports real and complex matrices and ships full autograd.

Why a portable direct solver
-----------------------------

- **Direct, not iterative.** STRUMPACK is a multifrontal *direct* solver: it
  factorizes ``A`` once and reuses that factorization for every subsequent
  solve (including the transpose solve in the backward pass). No tolerance to
  tune, no convergence to babysit.
- **Portable across vendors.** One ``CMakeLists.txt`` builds the same extension
  for **CPU**, **NVIDIA CUDA**, and **AMD ROCm**. The ROCm path is the headline
  feature: it gives you a differentiable sparse *direct* solve on Radeon / MI
  GPUs, where cuDSS (NVIDIA-only) cannot run.
- **Differentiable.** The adjoint of the linear solve is built in. Mark
  ``requires_grad`` on the matrix values and/or right-hand side and call
  ``.backward()`` — the backward pass reuses the cached factorization.
- **STRUMPACK only, no silent fallback.** The numeric core is the compiled
  STRUMPACK extension and nothing else. If the extension is missing, every
  call raises loudly, so a result can never silently come from a stand-in.

Quickstart
----------

.. code-block:: python

   import torch
   from torch_strumpack import spsolve, is_available

   assert is_available()  # True only when the compiled STRUMPACK ext is present

   # Square sparse CSR matrix with grad on its values
   A = ...                         # torch sparse_csr tensor (or dense)
   b = torch.randn(A.shape[0], dtype=torch.float64, requires_grad=True)

   x = spsolve(A, b)               # forward: factor + solve
   x.sum().backward()              # backward: one transpose solve, reuses the factorization

Use it standalone via :func:`torch_strumpack.spsolve`, or as a **torch-sla**
backend (``spsolve(..., backend='strumpack')``), where torch-sla owns autograd
and drives the autograd-free primitives in :mod:`torch_strumpack._core`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
