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


def parity_idx(nr, maxnl, m):
    """ idx for parities of a scalar """
    a_idx, s_idx = np.array([], dtype=int), np.array([], dtype=int)
    k = 0
    for l in range(m, maxnl):
        if (l + m) % 2 == 0:
            for n in range(0, nr):
                s_idx = np.append(s_idx, np.array([k], dtype=int))
                k += 1
        else:
            for n in range(0, nr):
                a_idx = np.append(a_idx, np.array([k], dtype=int))
                k += 1
    return a_idx, s_idx


def vector_parity_idx(nr, maxnl, m, parity):
    """ idx for parity of a vector field """
    a_idx, s_idx = parity_idx(nr, maxnl, m)
    dim = nr * (maxnl - m)
    if parity == "DP":
        return np.append(s_idx, dim + a_idx)
    else:
        return np.append(a_idx, dim + s_idx)


def restrict_parity(mat, res, row_parity, col_parity):
    """ separate the parity of a matrix """
    assert row_parity in ['DP', 'QP'] and col_parity in ['DP', 'QP']
    nr, maxnl, m = res
    dim = nr * (maxnl - m)
    a_idx, s_idx = parity_idx(nr, maxnl, m)
    rowidx = np.append(s_idx, dim + a_idx) if row_parity == 'DP' else np.append(a_idx, dim + s_idx)
    colidx = np.append(s_idx, dim + a_idx) if col_parity == 'DP' else np.append(a_idx, dim + s_idx)
    return mat[rowidx[:, None], colidx]


def reciprocal(w1, w2):
    """ compute the reciprocal of difference of eigenvalues at two truncations """
    # compute "intermodel separation"
    sigma = np.zeros((w1.shape[0],))
    n = 2
    for k in range(w1.shape[0]):
        sigma[k] = np.sum(np.abs(w1 - w1[k])[0:n]) / n
    # compute the "nearest" difference
    delta = np.zeros((w1.shape[0]-1,))
    for k in range(0, w1.shape[0]-1):
        i = np.argmin(np.abs(w2 - w1[k]))
        delta[k] = np.abs(w2[i] - w1[k]) / sigma[k]
        w2 = np.delete(w2, i)
    # compute and plot reciprocal
    reci = 1.0 / delta
    return reci
