Usage
=====

torch-strumpack is usable two ways: **standalone**, where it owns the autograd
of the solve, or **through torch-sla**, where torch-sla owns autograd and
drives the autograd-free primitives.

Standalone: basic solve
------------------------

:func:`torch_strumpack.spsolve` solves ``A x = b`` directly. ``A`` is a square
sparse CSR tensor (a dense tensor is accepted and converted), and ``b`` is the
right-hand side of shape ``(n,)`` or ``(n, k)`` for multiple right-hand sides.

.. code-block:: python

   import torch
   from torch_strumpack import spsolve

   # Build an SPD matrix as a sanity example
   n = 8
   M = torch.randn(n, n, dtype=torch.float64)
   A = M @ M.t() + n * torch.eye(n, dtype=torch.float64)

   b = torch.randn(n, dtype=torch.float64)

   x = spsolve(A.to_sparse_csr(), b)
   # matches a dense reference solve to machine precision
   assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-10)

Multiple right-hand sides work the same way — pass ``b`` of shape ``(n, k)``:

.. code-block:: python

   B = torch.randn(n, 4, dtype=torch.float64)
   X = spsolve(A.to_sparse_csr(), B)        # shape (n, 4)

Standalone: differentiation
---------------------------

Mark ``requires_grad`` on the matrix values and/or the right-hand side, then
call ``.backward()``. The backward pass performs a single transpose solve,
**reusing the factorization** computed in the forward pass.

.. code-block:: python

   import torch
   from torch_strumpack import spsolve

   n = 6
   Acsr = (torch.randn(n, n, dtype=torch.float64)).to_sparse_csr()
   # grad flows through the CSR *values*
   vals = Acsr.values().clone().requires_grad_(True)
   A = torch.sparse_csr_tensor(Acsr.crow_indices(), Acsr.col_indices(), vals, (n, n))

   b = torch.randn(n, dtype=torch.float64, requires_grad=True)

   x = spsolve(A, b)
   x.sum().backward()

   print(vals.grad)   # dL/dA on the sparsity pattern
   print(b.grad)      # dL/db = A^{-T} (dL/dx)

The adjoint implemented is, for ``x = A^{-1} b``:

- ``dL/db = A^{-T} (dL/dx)`` — one transpose solve, and
- ``dL/dA_ij = -(dL/db)_i * x_j`` on the sparsity pattern.

Through torch-sla (portable direct backend)
-------------------------------------------

torch-strumpack is designed to plug into **torch-sla** as its sparse *direct*
backend. In that mode torch-sla owns autograd and calls the autograd-free
primitives (:func:`torch_strumpack.factor`, :func:`torch_strumpack.solve`,
:func:`torch_strumpack.solve_transpose`) directly — so there is **no
double-differentiation**.

.. code-block:: python

   import torch
   from torch_sla import spsolve   # torch-sla's dispatcher

   A = ...   # torch-sla sparse tensor / torch sparse_csr
   b = torch.randn(A.shape[0], dtype=torch.float64)

   # Route the solve through STRUMPACK — the portable direct backend.
   x = spsolve(A, b, backend='strumpack')

This is the recommended way to get a **direct** solve on **ROCm / AMD GPUs**:
STRUMPACK is the portable direct path, so the same ``backend='strumpack'`` call
runs on CPU, CUDA, and ROCm without code changes — covering the AMD case that
NVIDIA-only direct solvers cannot.

Low-level primitives
--------------------

If you are building your own backend adapter, the autograd-free contract is
three functions plus an opaque factorization handle:

.. code-block:: python

   import torch_strumpack as ts

   fac = ts.factor(crow, col, values, n)   # -> Factorization (reusable handle)
   x = ts.solve(fac, b)                     # A x = b
   y = ts.solve_transpose(fac, b)           # A^T y = b  (lazily factors A^T once)

The :class:`~torch_strumpack._core.Factorization` handle is reused across
solves and for the transpose solve in a backward pass.

Availability check
------------------

Because there is no fallback, gate any solve on
:func:`torch_strumpack.is_available` if you want a clear error path:

.. code-block:: python

   import torch_strumpack

   if not torch_strumpack.is_available():
       raise RuntimeError("STRUMPACK extension not loaded — check your wheel/backend ABI")
