import numpy as np
from numba import njit


def chebyshev_grid(nrg: int, ri: float):
    xg = np.cos((2*np.arange(nrg)+1) / (2*nrg) * np.pi)
    return (xg + 1)/2 + ri


def chebyshev_norm(n: int):
    return 1/np.pi**0.5 if n == 0 else (2/np.pi)**0.5


def chebyshev_weight(nr: int):
    return np.pi / nr


@njit
def chebyshev1(nr, x, coe):
    t = np.full((len(x), nr), fill_value=np.NaN)
    t[:, 0] = coe
    if nr > 1:
        t[:, 1] = coe * x
    if nr > 2:
        for n in range(2, nr):
            t[:, n] = 2 * x * t[:, n-1] - t[:, n-2]
    return t


@njit
def chebyshev2(nr, x, coe):
    u = np.full((len(x), nr), fill_value=np.NaN)
    u[:, 0] = coe
    if nr > 1:
        u[:, 1] = 2 * coe * x
    if nr > 2:
        for n in range(2, nr):
            u[:, n] = 2 * x * u[:, n-1] - u[:, n-2]
    return u


def Dchebyshev1(nr, x, coe):
    dt = np.zeros((len(x), nr))
    dt[:, 1:] = chebyshev2(nr-1, x, coe) @ np.diag(np.arange(1, nr))
    return dt


def D2chebyshev1(nr, x, coe):
    i_1 = np.nonzero(x == -1)[0]
    i1 = np.nonzero(x == 1)[0]
    if len(i_1) + len(i1) > 0:
        raise NotImplementedError("evaluation of second derivative of Chebyshev 1st kind not implemented at end points")
    t = chebyshev1(nr, x, coe)
    u = chebyshev2(nr, x, coe)
    return np.diag(1/(x**2-1)) @ (t @ np.diag(np.arange(nr)+1) - u) @ np.diag(np.arange(nr))


def chebyshev(nr, rg, ri, coe=1.0):
    xg = 2 * (rg - ri) - 1
    norm = np.diag([chebyshev_norm(n) for n in range(nr)])
    return chebyshev1(nr, xg, coe) @ norm


def Dchebyshev(nr, rg, ri, coe=1.0):
    xg = 2 * (rg - ri) - 1
    norm = np.diag([chebyshev_norm(n) for n in range(nr)])
    # return Dchebyshev1(nr, xg, coe) @ norm
    return 2*Dchebyshev1(nr, xg, coe) @ norm


def D2chebyshev(nr, rg, ri, coe=1.0):
    xg = 2 * (rg - ri) - 1
    norm = np.diag([chebyshev_norm(n) for n in range(nr)])
    # return D2chebyshev1(nr, xg, coe) @ norm
    return 4*D2chebyshev1(nr, xg, coe) @ norm


def divrT(nr, rg, ri, coe=1.0):
    return np.diag(1/rg) @ chebyshev(nr, rg, ri, coe)


def divrdiffrT(nr, rg, ri, coe=1.0):
    return divrT(nr, rg, ri, coe) + Dchebyshev(nr, rg, ri, coe)


def diff2rT(nr, rg, ri, coe=1.0):
    return 2*Dchebyshev(nr, rg, ri, coe) + np.diag(rg) @ D2chebyshev(nr, rg, ri, coe)


def laplacianlT(nr, l, rg, ri, coe=1.0):
    res = D2chebyshev(nr, rg, ri, coe)
    res += np.diag(2/rg) @ Dchebyshev(nr, rg, ri, coe)
    res += -l*(l+1) * np.diag(1/rg**2) @ chebyshev(nr, rg, ri, coe)
    return res


def quicc_norm(nr):
    """ weights to convert QuICC basis to normalised basis"""
    return np.array([np.pi**0.5 if n == 0 else (np.pi/2)**0.5*2 for n in range(nr)])


def inv_quicc_norm(nr):
    """ weights to convert normalised basis to QuICC basis """
    return np.array([1/np.pi**0.5 if n == 0 else (2/np.pi)**0.5/2 for n in range(nr)])


if __name__ == "__main__":
    ri = 0.35
    nrg = 30
    nr = 11
    rg = chebyshev_grid(nrg=nrg, ri=ri)

    t = chebyshev(nr, rg, ri)
    weight = np.diag(chebyshev_weight(nrg) * np.ones(nrg))
    tmp = t.T @ weight @ t
    print(np.allclose(tmp, np.identity(nr)))

    ri = 0.0001
    nr = 6
    rg = np.linspace(ri, ri+1, 101)[1:-1]
    t = chebyshev(nr, rg, ri)
    divrdiffrt = divrdiffrT(nr, rg, ri)
    diff2rt = diff2rT(nr, rg, ri)
    divrt = divrT(nr, rg, ri)
