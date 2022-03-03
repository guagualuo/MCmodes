import numpy as np
from numba import njit

import quicc.geometry.worland.worland_basis as wb
from utils import Timer


@njit
def jacobiP(nr, a, b, x, coe):
    w = np.full((x.shape[0], nr), fill_value=np.NaN)
    w[:, 0] = coe
    if nr > 1:
        w[:, 1] = 0.5 * coe * (a-b+(2+a+b)*x)
    if nr > 2:
        for n in range(2, nr):
            an = 2*n*(n+a+b)*(2*n+a+b-2)
            bn = (2*n+a+b-1)*(2*n+a+b)*(2*n+a+b-2)
            cn = (2*n+a+b-1)*(a**2-b**2)
            dn = -2*(n+a-1)*(n+b-1)*(2*n+a+b)
            w[:, n] = ((bn*x+cn)*w[:, n-1] + dn*w[:, n-2]) / an
    return w


@njit
def DjacobiP(nr, a, b, x, coe):
    if nr < 2:
        return np.zeros((x.shape[0], nr))
    else:
        p = jacobiP(nr-1, a+1, b+1, x, coe)
        dp = np.zeros((x.shape[0], nr))
        for n in range(1, nr):
            dp[:, n] = 0.5 * (1.0 + a + b + n) * p[:, n-1]
        return dp


@njit
def D2jacobiP(nr, a, b, x, coe):
    if nr < 3:
        return np.zeros((x.shape[0], nr))
    else:
        p = jacobiP(nr-2, a+2, b+2, x, coe)
        d2p = np.zeros((x.shape[0], nr))
        for n in range(2, nr):
            d2p[:, n] = 0.25*(1+a+b+n)*(2+a+b+n) * p[:, n-2]
        return d2p


def worland(nr, l, rg):
    """W(r)"""
    w = jacobiP(nr, -0.5, l-0.5, 2*rg**2-1, rg**l)
    for n in range(nr):
        w[:, n] *= wb.worland_norm(n, l)
    return w


def divrW(nr, l, rg):
    """W(r)/r"""
    if l >= 1:
        divrw = jacobiP(nr, -0.5, l-0.5, 2*rg**2-1, rg**(l-1))
        for n in range(nr):
            divrw[:, n] *= wb.worland_norm(n, l)
        return divrw
    else:
        return np.zeros((rg.shape[0], nr))


def divrdiffrW(nr, l, rg):
    """1/r D r W(r) """
    a, b = -0.5, l-0.5
    p = jacobiP(nr, a, b, 2*rg**2-1, (l+1)*rg**(l-1))
    dp = DjacobiP(nr, a, b, 2*rg**2-1, 4*rg**(l+1))
    divrdiffrw = p + dp
    for n in range(nr):
        divrdiffrw[:, n] *= wb.worland_norm(n, l)
    return divrdiffrw


def diff2rW(nr, l, rg):
    """ D^2 r W_n^l(r) """
    a, b = -0.5, l-0.5
    diff2w = DjacobiP(nr, a, b, 2*rg**2-1, 4*(2*l+3)*rg**(l+1)) + D2jacobiP(nr, a, b, 2*rg**2-1, 16*rg**(l+3))
    if l > 0:
        diff2w += jacobiP(nr, a, b, 2*rg**2-1, l*(l+1) * rg**(l-1))
    for n in range(nr):
        diff2w[:, n] *= wb.worland_norm(n, l)
    return diff2w


def laplacianlW(nr, l, rg):
    """ 1/r^2 D(r^2 D ) - l(l+1)/r^2 W_n^l(r) """
    a, b = -0.5, l - 0.5
    laplw = D2jacobiP(nr, a, b, 2*rg**2-1, 16.0 * rg**(l+2)) + DjacobiP(nr, a, b, 2*rg**2-1, (8.0*l+12.0) * rg**l)
    for n in range(nr):
        laplw[:, n] *= wb.worland_norm(n, l)
    return laplw


# warmer
jacobiP(2, -0.5, 0.5, np.linspace(0,1,4), np.linspace(-1,1,4))
DjacobiP(2, -0.5, 0.5, np.linspace(0, 1, 4), np.linspace(-1, 1, 4))
D2jacobiP(2, -0.5, 0.5, np.linspace(0, 1, 4), np.linspace(-1, 1, 4))
