import numpy as np
import scipy.sparse as scsp
from sympy.abc import r
from sympy import lambdify, simplify, diff
from sympy.parsing.mathematica import mathematica
from scipy import special
from abc import ABC, abstractmethod

import quicc.geometry.worland.worland_basis as wb


def worland_grid(nr):
    return np.cos(np.pi/2*(np.arange(nr-1,-1,-1)+0.5)/nr)

def worland_weight(nr):
    return np.pi/(2*nr)


def _jacobiP(n, a, b, x):
    """ Compute the value of jacobi polynomial at x """
    return special.eval_jacobi(n, a, b, x)


def _DjacobiP(n, a, b, x):
    """ Compute first derivative of jacobi polynomial at x """
    if n == 0:
        return np.zeros_like(x)
    else:
        return 0.5 * (1.0 + a + b + n) * _jacobiP(n-1, a+1, b+1, x)


def _D2jacobiP(n, a, b, x):
    if n == 0 or n == 1:
        return np.zeros_like(x)
    else:
        return 0.25*(1+a+b+n)*(2+a+b+n)*_jacobiP(n-2, a+2, b+2, x)


def _worland(n, l, r_grid):
    """ worland value at radial grids """
    return wb.worland_norm(n, l) * r_grid**l * special.eval_jacobi(n, -0.5, l - 0.5, 2.0 * r_grid ** 2 - 1.0)


def _divrdiffrW(n, l, r_grid):
    """ 1/r D r W_n^l: return 0 for l <= 0"""
    if l > 0:
        a, b = -0.5, l-0.5
        return wb.worland_norm(n, l) * ((l+1.0) * r_grid**(l-1) * _jacobiP(n, a, b, 2.0*r_grid**2-1.0) +\
                                        4.0*r_grid**(l+1)*_DjacobiP(n, a, b, 2.0*r_grid**2-1.0))
    else:
        return np.zeros_like(r_grid)


def _diff2rW(n, l, r_grid):
    """ D^2 r W_n^l(r) """
    a, b = -0.5, l-0.5
    val = 4*(2*l+3)*r_grid**(l+1)*_DjacobiP(n, a, b, 2.0*r_grid**2-1.0) + 16*r_grid**(l+3)*_D2jacobiP(n, a, b, 2.0*r_grid**2-1.0)
    if l > 0:
        val += l*(l+1) * r_grid**(l-1) * _jacobiP(n, a, b, 2.0*r_grid**2-1.0)
    return wb.worland_norm(n, l) * val


def _laplacianlW(n, l, r):
    """ D^2 + 2/r D - l(l+1)/r^2 """
    a, b = -0.5, l-0.5
    return wb.worland_norm(n, l) * (16.0 * r**(l+2) * _D2jacobiP(n,a,b,2.0*r**2-1.0) +
                                   (8.0*l+12.0) * r**l * _DjacobiP(n,a,b,2.0*r**2-1.0))


def worland(nr, l, r_grid):
    """W(r)"""
    return np.concatenate([_worland(ni, l, r_grid).reshape(-1, 1) for ni in range(nr)], axis=1)


def divrW(nr, l, r_grid):
    """W(r)/r"""
    if l == 0:
        return np.zeros((r_grid.shape[0], nr))
    else:
        return np.concatenate([(_worland(ni, l, r_grid)/r_grid).reshape(-1, 1) for ni in range(nr)], axis=1)


def divrdiffrW(nr, l, r_grid):
    """1/r D r W(r) """
    return np.concatenate([_divrdiffrW(ni, l, r_grid).reshape(-1, 1) for ni in range(nr)], axis=1)


def diff2rW(nr, l, r_grid):
    """ D^2 r W_n^l(r) """
    return np.concatenate([_diff2rW(ni, l, r_grid).reshape(-1, 1) for ni in range(nr)], axis=1)


def laplacianlW(nr, l, r_grid):
    """ 1/r^2 D(r^2 D ) - l(l+1)/r^2 W_n^l(r) """
    return np.concatenate([_laplacianlW(ni, l, r_grid).reshape(-1, 1) for ni in range(nr)], axis=1)


class SymOperatorBase(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        pass

    @abstractmethod
    def apply(self, expr, r_grid):
        pass


class SymDivr(SymOperatorBase):
    def __init__(self):
        super(SymDivr, self).__init__()

    def apply(self, expr, r_grid):
        divrf = simplify(expr / r)
        func = lambdify(r, divrf, "numpy")
        return func(r_grid)


class SymDivr2(SymOperatorBase):
    def __init__(self):
        super(SymDivr2, self).__init__()

    def apply(self, expr, r_grid):
        divr2f = simplify(expr / r / r)
        func = lambdify(r, divr2f, "numpy")
        return func(r_grid)


class SymDivrDiffr(SymOperatorBase):
    def __init__(self):
        super(SymDivrDiffr, self).__init__()

    def apply(self, expr, r_grid):
        divrdiffrf = simplify(diff(r * expr, r) / r)
        func = lambdify(r, divrdiffrf, "numpy")
        return func(r_grid)


class SymDiff(SymOperatorBase):
    def __init__(self):
        super(SymDiff, self).__init__()

    def apply(self, expr, r_grid):
        difff = simplify(diff(expr, r))
        func = lambdify(r, difff, "numpy")
        return func(r_grid)


class SymDiffDivr(SymOperatorBase):
    def __init__(self):
        super(SymDiffDivr, self).__init__()

    def apply(self, expr, r_grid):
        diffdivrf = simplify(diff(expr / r, r))
        func = lambdify(r, diffdivrf, "numpy")
        return func(r_grid)


class SymDivr2Diffr(SymOperatorBase):
    def __init__(self):
        super(SymDivr2Diffr, self).__init__()

    def apply(self, expr, r_grid):
        divrdiffrf = simplify(diff(r * expr, r) / r ** 2)
        func = lambdify(r, divrdiffrf, "numpy")
        return func(r_grid)


class SymLaplacianl(SymOperatorBase):
    def __init__(self, l):
        super(SymLaplacianl, self).__init__()
        self.l = l

    def apply(self, expr, r_grid):
        l = self.l
        laplacianlf = simplify(diff(r ** 2 * diff(expr, r), r) / r ** 2 - l * (l + 1) / r ** 2 * expr)
        func = lambdify(r, laplacianlf, "numpy")
        return func(r_grid)


class SymrDiffDivr2Diffr(SymOperatorBase):
    def __init__(self):
        super(SymrDiffDivr2Diffr, self).__init__()

    def apply(self, expr, r_grid):
        opf = simplify(r*diff(diff(r*expr, r)/r**2, r))
        func = lambdify(r, opf, "numpy")
        return func(r_grid)


class SphericalHarmonicMode:
    """
    Spherical harmonic mode of toroidal/poloidal field: curl(t(r)Ylm r) and curl(curl( s(r) Ylm r ))
    """
    def __init__(self, comp, l, m, radial_func: str):
        assert comp in ['tor', 'pol']
        self.comp = comp
        self.l = l
        self.m = m
        if isinstance(radial_func, str):
            self.radial_expr = mathematica(radial_func)
            self.radial_func = lambdify(r, self.radial_expr, "numpy")
        else:
            self.radial_expr = radial_func
            self.radial_func = lambdify(r, self.radial_expr, "numpy")

    def curl(self):
        if self.comp == "tor":
            return SphericalHarmonicMode("pol", self.l, self.m, radial_func=self.radial_expr)
        else:
            l = self.l
            expr = simplify(diff(r**2*diff(self.radial_expr, r), r)/r**2 - l*(l+1)/r**2*self.radial_expr)
            return SphericalHarmonicMode("tor", self.l, self.m, radial_func=-expr)


if __name__ == "__main__":
    from sympy.parsing.mathematica import mathematica
    r_grid = worland_grid(100)
    expr = mathematica("pi/5 r(1-r^2)")
    # print(expr)
    # print(sym_divrdiffr(expr, r_grid))
    # print(np.allclose(2*np.pi/5*(1-2*r_grid**2), sym_divrdiffr(expr, r_grid)))
    print(_diff2rW(5, 2, r_grid))

