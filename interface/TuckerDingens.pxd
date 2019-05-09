from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "cpython_TuckerDingens.hpp":

##############################################################
############ MAIN ALGO #######################################

    unsigned gc_MatMult(vector[vector[double]] &A, vector[double] &v, vector[double] &result)
