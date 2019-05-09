import TuckerDingens

A = [[float(j) if i is j else 0 for j in range(1, 5)] for i in range(1, 5)]
v = [1,2,3,4]
#Av squares diag(v)
Av = TuckerDingens.MatMult(A, v)
print(str(A) + " * " + str(v) + " = " + str(Av))
