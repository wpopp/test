#include <vector>
#include <assert.h>

namespace TuckerDingens{

inline double VecDot(std::vector<double> &v, std::vector<double> &w){
	assert(v.size() == v.size());
	double s = 0;
	for(unsigned i = 0; i < v.size(); ++i){
		s+=v[i]*w[i];
	}
	return s;
}

void MatMult(std::vector<std::vector<double> > &A, std::vector<double> &v, std::vector<double> &result){
	result.assign(v.size(), .0f);
	for(unsigned i = 0; i < A.size(); ++i){
		for(unsigned j = 0; j < v.size(); ++j){
			result[i] += A[i][j]*v[j];
		}
	}
}
