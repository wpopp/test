# TuckerDingens

INSTALL

./bootstrap
./configure
make install

REQUIREMENTS
autotools and related tools
cython
python
C++ stuff

python call code in ./python ruft python bibliothek die in ./interface liegt auf die wiederrum c++ code in /src called.

FUNCTIONALITY

python:

TuckerDecomposition.MatMult(A, v)

c++:

MatMult(A, v, result)

