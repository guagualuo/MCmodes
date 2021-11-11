import numpy as np
from sympy.physics.wigner import gaunt, wigner_3j
from sympy.parsing.mathematica import mathematica
from sympy import lambdify
from sympy.abc import r
from math import sqrt, pi

from utils import Timer


def _gaunt(l_1, l_2, l_3, m_1, m_2, m_3):
    g = gaunt(l_1, l_2, l_3, m_1, m_2, m_3)
    return g if isinstance(g, int) else (-1)**(m_1+m_2)*float(gaunt(l_1, l_2, l_3, m_1, m_2, m_3).n(17))


def _elsasser(l_1, l_2, l_3, m_1, m_2, m_3):
    e = wigner_3j(l_1+1, l_2+1, l_3+1, 0, 0, 0) * wigner_3j(l_1, l_2, l_3, m_1, m_2, m_3)
    if not isinstance(e, int):
        e = float(e.n(17))
    if e != 0:
        e *= sqrt((2*l_1+1)*(2*l_2+1)*(2*l_3+1)/(4*pi))
        e *= sqrt((l_1+l_2+l_3+2)*(l_1+l_2+l_3+4)/4/(l_1+l_2+l_3+3))
        e *= sqrt((l_1+l_2-l_3+1)*(l_1-l_2+l_3+1)*(-l_1+l_2+l_3+1))
    return -1.0j * (-1)**(m_1+m_2) * e


class SphericalHarmonicMode:
    def __init__(self, comp, l, m, radial_func: str):
        assert comp in ['tor', 'pol']
        self.comp = comp
        self.l = l
        self.m = m
        self.radial_expr = mathematica(radial_func)
        self.radial_func = lambdify(r, self.radial_expr, "numpy")


def gaunt_matrix(maxnl, m, lb, return_matrix=False):
    """
    the K_{alpha, beta, gamma} matrix assumes mb = 0, and ma = m, mg = -m
    :param maxnl: l < maxnl
    :param lb: l for beta mode
    :param m: m for alpha mode, -m for gamma mode
    :return:
    """
    assert m >= 0
    ma, mb, mg = m, 0, -m
    gmat = np.zeros((maxnl-m, maxnl-m))
    gmap = {}
    for i, lg in enumerate(range(m, maxnl)):
        for j, la in enumerate(range(m, maxnl)):
            gmat[i, j] = _gaunt(la, lb, lg, m, mb, -m)
            gmap[(lg, la)] = gmat[i, j]
    if return_matrix:
        return gmat
    else:
        return gmap


def elsasser_matrix(maxnl, m, lb, return_matrix=False):
    """
    the L_{alpha, beta, gamma} matrix assumes mb = 0, and ma = m, mg = -m
    :param maxnl: l < maxnl
    :param lb: l for beta mode
    :param m: m for alpha mode, -m for gamma mode
    :return:
    """
    assert m >= 0
    ma, mb, mg = m, 0, -m
    emat = np.zeros((maxnl - m, maxnl - m), dtype=np.complex128)
    emap = {}
    for i, lg in enumerate(range(m, maxnl)):
        for j, la in enumerate(range(m, maxnl)):
            emat[i, j] = _elsasser(la, lb, lg, m, mb, -m)
            emap[(lg, la)] = emat[i, j]
    if return_matrix:
        return emat
    else:
        return emap


if __name__ == "__main__":
    # l_1, l_2, l_3, m_1, m_2, m_3 = 0, 2, 0, 1, 0, -1
    # print(_gaunt(l_1, l_2, l_3, m_1, m_2, m_3))
    # print(_elsasser(l_1, l_2, l_3, m_1, m_2, m_3))
    with Timer():
        import matplotlib.pyplot as plt
        maxnl, m, lb = 10, 1, 1
        g = gaunt_matrix(maxnl, m, lb, return_matrix=True)
        e = elsasser_matrix(maxnl, m, lb, return_matrix=True)
    fig, (ax1, ax2) = plt.subplots(figsize=(10, 4.5), ncols=2)
    ax1.spy(g)
    ax2.spy(e)
    plt.show()

