import numpy as np
import scipy.sparse as scsp
from sympy.abc import r
from sympy import lambdify, simplify, diff
from sympy.parsing.mathematica import mathematica
from scipy import special

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
        return np.zeros(x.shape)
    else:
        return 0.5 * (1.0 + a + b + n) * _jacobiP(n-1, a+1, b+1, x)


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


def sym_operators(name, **kwargs):
    if name == 'diff': return sym_diff
    if name == 'divr': return sym_divr
    if name == 'diffdivr': return sym_diffdivr
    if name == 'divrdiffr': return sym_divrdiffr


def sym_divr(expr, r_grid):
    divrf = simplify(expr/r)
    func = lambdify(r, divrf, "numpy")
    return func(r_grid)


def sym_divrdiffr(expr, r_grid):
    divrdiffrf = simplify(diff(r*expr, r)/r)
    func = lambdify(r, divrdiffrf, "numpy")
    return func(r_grid)


def sym_diff(expr, r_grid):
    difff = simplify(diff(expr, r))
    func = lambdify(r, difff, "numpy")
    return func(r_grid)


def sym_diffdivr(expr, r_grid):
    diffdivrf = simplify(diff(expr/r, r))
    func = lambdify(r, diffdivrf, "numpy")
    return func(r_grid)


def sym_laplacianl(expr, r_grid, l):
    laplacianlf = simplify(diff(r**2*diff(expr, r), r)/r**2 - l*(l+1)/r**2*expr)
    func = lambdify(r, laplacianlf, "numpy")
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
    print(_divrdiffrW(2, 1, r_grid))

