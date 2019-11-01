import numpy as np
def normalizer(x):
    x /= np.linalg.norm(x)
    return x
n1=1.000277
n2=1.33

L= np.array([10,2])
C= np.array([0,10])
N= np.array([0,1])
d= L-C
# normalized so that the dot product represents the cos
d = d/np.sqrt(d@d.T)
print("d: %r"%(d))
eta = n2/n1
#print(eta)
R=-d-2*(N@-d.T)*N


T = -eta * (N@d.T)
tmp = 1 -(eta**2)*(1-(N@d.T)**2)
T = T - np.sqrt(tmp)
T = T*N + eta*d
print(T/np.sqrt(T@T.T))

