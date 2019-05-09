#include "Algebra.hpp"

#include "cpython_TuckerDingens.hpp"

void gc_MatMult(std::vector<std::vector<double> > &A, std::vector<double> &v, std::vector<double> &result){
    result = TuckerDingens.MatMult(A, v);
}

