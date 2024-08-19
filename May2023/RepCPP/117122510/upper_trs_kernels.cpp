

#include "core/solver/upper_trs_kernels.hpp"


#include <memory>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>


namespace gko {
namespace kernels {
namespace omp {

namespace upper_trs {


void should_perform_transpose(std::shared_ptr<const OmpExecutor> exec,
bool& do_transpose)
{
do_transpose = false;
}


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const OmpExecutor> exec,
const matrix::Csr<ValueType, IndexType>* matrix,
std::shared_ptr<solver::SolveStruct>& solve_struct,
bool unit_diag, const solver::trisolve_algorithm algorithm,
const size_type num_rhs)
{
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
GKO_DECLARE_UPPER_TRS_GENERATE_KERNEL);



template <typename ValueType, typename IndexType>
void solve(std::shared_ptr<const OmpExecutor> exec,
const matrix::Csr<ValueType, IndexType>* matrix,
const solver::SolveStruct* solve_struct, bool unit_diag,
const solver::trisolve_algorithm algorithm,
matrix::Dense<ValueType>* trans_b, matrix::Dense<ValueType>* trans_x,
const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x)
{
auto row_ptrs = matrix->get_const_row_ptrs();
auto col_idxs = matrix->get_const_col_idxs();
auto vals = matrix->get_const_values();

#pragma omp parallel for
for (size_type j = 0; j < b->get_size()[1]; ++j) {
for (size_type inv_row = 0; inv_row < matrix->get_size()[0];
++inv_row) {
auto row = matrix->get_size()[0] - 1 - inv_row;
auto diag = one<ValueType>();
x->at(row, j) = b->at(row, j);
for (auto k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
auto col = col_idxs[k];
if (col > row) {
x->at(row, j) -= vals[k] * x->at(col, j);
}
if (col == row) {
diag = vals[k];
}
}
if (!unit_diag) {
x->at(row, j) /= diag;
}
}
}
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
GKO_DECLARE_UPPER_TRS_SOLVE_KERNEL);


}  
}  
}  
}  