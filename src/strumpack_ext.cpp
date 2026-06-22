// nanobind binding for STRUMPACK's sequential sparse direct solver.
// v1: numpy host arrays in/out. STRUMPACK offloads the dense frontal
// factorization to the GPU internally (CUDA/HIP) when built with it.
//
// Exposes a reusable Factorization (factor once, solve many) matching the
// torch_strumpack._core contract: factor / solve. Transpose solves are done
// by factoring A^T (the caller hands us the transposed CSR), so no separate
// transpose-solve entry point is needed here.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <memory>
#include <vector>
#include <stdexcept>

#include <StrumpackSparseSolver.hpp>

namespace nb = nanobind;
using strumpack::StrumpackSparseSolver;
using strumpack::ReturnCode;

using I32Arr = nb::ndarray<const int, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using F64Arr = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

struct Factorization {
  std::unique_ptr<StrumpackSparseSolver<double, int>> sp;
  // keep copies alive for STRUMPACK's lifetime
  std::vector<int> indptr, indices;
  std::vector<double> data;
  int n;
};

static const char* rc_str(ReturnCode r) {
  switch (r) {
    case ReturnCode::SUCCESS: return "SUCCESS";
    case ReturnCode::MATRIX_NOT_SET: return "MATRIX_NOT_SET";
    case ReturnCode::REORDERING_ERROR: return "REORDERING_ERROR";
    case ReturnCode::ZERO_PIVOT: return "ZERO_PIVOT";
    case ReturnCode::NO_CONVERGENCE: return "NO_CONVERGENCE";
    default: return "UNKNOWN";
  }
}

// Factor a CSR matrix (indptr size n+1, indices/data size nnz).
static Factorization* factorize(I32Arr indptr, I32Arr indices, F64Arr data, int n) {
  auto f = new Factorization();
  f->n = n;
  f->indptr.assign(indptr.data(), indptr.data() + (n + 1));
  size_t nnz = indices.shape(0);
  f->indices.assign(indices.data(), indices.data() + nnz);
  f->data.assign(data.data(), data.data() + nnz);

  f->sp = std::make_unique<StrumpackSparseSolver<double, int>>(false);  // verbose=false
  f->sp->set_csr_matrix(n, f->indptr.data(), f->indices.data(), f->data.data(), false);
  ReturnCode rc = f->sp->reorder();
  if (rc != ReturnCode::SUCCESS) { delete f; throw std::runtime_error(std::string("reorder: ") + rc_str(rc)); }
  rc = f->sp->factor();
  if (rc != ReturnCode::SUCCESS) { delete f; throw std::runtime_error(std::string("factor: ") + rc_str(rc)); }
  return f;
}

// Solve A x = b. b is (n,) or (n, nrhs) C-contiguous. Returns x same shape.
static nb::ndarray<nb::numpy, double> solve(Factorization* f, F64Arr b) {
  int n = f->n;
  int nrhs = (b.ndim() == 1) ? 1 : (int)b.shape(1);
  if ((int)b.shape(0) != n)
    throw std::runtime_error("b leading dim must equal n");

  double* x = new double[(size_t)n * nrhs];
  nb::capsule owner(x, [](void* p) noexcept { delete[] (double*)p; });

  std::vector<double> bcol(n), xcol(n);
  const double* bp = b.data();
  for (int k = 0; k < nrhs; ++k) {
    for (int i = 0; i < n; ++i) bcol[i] = bp[(size_t)i * nrhs + k];  // gather column (C-order)
    ReturnCode rc = f->sp->solve(bcol.data(), xcol.data());
    if (rc != ReturnCode::SUCCESS) { throw std::runtime_error(std::string("solve: ") + rc_str(rc)); }
    for (int i = 0; i < n; ++i) x[(size_t)i * nrhs + k] = xcol[i];   // scatter back
  }

  if (b.ndim() == 1) {
    size_t shape[1] = {(size_t)n};
    return nb::ndarray<nb::numpy, double>(x, 1, shape, owner);
  } else {
    size_t shape[2] = {(size_t)n, (size_t)nrhs};
    return nb::ndarray<nb::numpy, double>(x, 2, shape, owner);
  }
}

NB_MODULE(_strumpack_ext, m) {
  nb::class_<Factorization>(m, "Factorization");
  m.def("factorize", &factorize, nb::rv_policy::take_ownership);
  m.def("solve", &solve);
  m.def("backend", []() { return std::string("strumpack"); });
}
