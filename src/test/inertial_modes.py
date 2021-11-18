""" This module contains is used to test the inertial modes frequency """
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from models import InertialModes
from utils import single_eig

nr, maxnl, m = 11, 21, 0
model = InertialModes(nr, maxnl, m, inviscid=True)
ops = model.setup_operator()
A, B = model.setup_eigen_problem(ops)

# fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
# ax1.spy(A, marker='.')
# ax2.spy(B, marker='.')

if m > 0:
    # test QG modes eigenvalues
    from math import sqrt
    for k in range(1, (2*(nr-1)+1-m)//2):
        wi = -2/(m+2)*(sqrt(1+m*(m+2)/(k*(2*k+2*m+1))) - 1)*1.0j
        w, _ = single_eig(A, B, target=wi+1e-8, nev=1)
        print((w-wi)/np.abs(wi))

    w, _ = la.eig(A.todense(), B.todense())
    w = w[np.abs(w)!=np.inf]
    freq = np.imag(w)
    freq.sort()
    plt.plot(freq, '.')
else:
    w, _ = la.eig(A.todense(), B.todense())
    w = w[np.abs(w) != np.inf]
    print(w)
    freq = np.imag(w)
    freq.sort()
    plt.plot(freq, '.')

plt.show()


