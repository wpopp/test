from libcpp.vector cimport vector

################################################################
############ Code that calls code in cpython_TuckerDingens.cpp #

def MatMult(A, v):
    cdef vector[double] result

    gc_paraTD(A, v, result);

	return result

