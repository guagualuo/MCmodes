import time
import scipy.sparse as scsp
import scipy.sparse.linalg as spla
import numpy as np


class Timer(object):
    def __init__(self, process_name=''):
        self.name = process_name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.name} elapsed {time.time() - self.start:.2f} s.")


def single_eig(A, B, target, nev=5, tol=1e-14):
    """ Compute the eigenvector, for given eigenvalue """

    A = scsp.csc_matrix(A)
    B = scsp.csc_matrix(B)
    C = A - target * B
    clu = spla.splu(C)

    def bmx(x):
        return clu.solve(B.dot(x))

    D = spla.LinearOperator(dtype=np.complex128, shape=np.shape(A), matvec=bmx)
    evals, u = spla.eigs(D, k=nev, which='LM', tol=tol)
    return 1.0 / evals + target, u
