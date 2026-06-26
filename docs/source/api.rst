API Reference
=============

Public API
----------

.. automodule:: torch_strumpack
   :members: spsolve, factor, solve, solve_transpose, is_available
   :undoc-members:

.. autodata:: torch_strumpack.__version__
   :no-value:

Autograd
--------

.. automodule:: torch_strumpack.autograd
   :members: spsolve
   :undoc-members:

Numeric core
------------

The autograd-free primitives that a backend adapter (e.g. torch-sla) calls
directly. Keeping this layer autograd-free is what lets the same code run
standalone or under torch-sla without double-differentiation.

.. automodule:: torch_strumpack._core
   :members: factor, solve, solve_transpose, backend, Factorization
   :undoc-members:
   :show-inheritance:
