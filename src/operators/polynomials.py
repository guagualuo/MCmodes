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


def energy_quadrature(nr):
    """Physical space grids and weights for energy computing. Legendre quadrature"""

    assert nr % 2 == 0, 'we assume even number of grids, so we can use the symmetry'
    grids, weights = special.roots_jacobi(nr, 0.0, 0.0)

    grids = grids[int(nr/2):nr]
    weights = weights[int(nr/2):nr]

    return grids, weights


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

def _divrworland(n, l, r_grid):
    """ worland value at radial grids """
    if l > 0:
        return wb.worland_norm(n, l) * r_grid**(l-1) * special.eval_jacobi(n, -0.5, l - 0.5, 2.0 * r_grid ** 2 - 1.0)
    else:
        return np.zeros_like(r_grid)


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
        return np.concatenate([_divrworland(ni, l, r_grid).reshape(-1, 1) for ni in range(nr)], axis=1)


def divrdiffrW(nr, l, r_grid):
    """1/r D r W(r) """
    return np.concatenate([_divrdiffrW(ni, l, r_grid).reshape(-1, 1) for ni in range(nr)], axis=1)


def diff2rW(nr, l, r_grid):
    """ D^2 r W_n^l(r) """
    return np.concatenate([_diff2rW(ni, l, r_grid).reshape(-1, 1) for ni in range(nr)], axis=1)


def laplacianlW(nr, l, r_grid):
    """ 1/r^2 D(r^2 D ) - l(l+1)/r^2 W_n^l(r) """
    return np.concatenate([_laplacianlW(ni, l, r_grid).reshape(-1, 1) for ni in range(nr)], axis=1)


def energy_weight_tor(l, n):
    """Compute the energy weight matrix at l, given maximum radial mode up to n"""
    rdegree = l + 2 * n
    nr = rdegree + 2 + (rdegree + 2) % 2

    grids, weights = energy_quadrature(nr)
    wmat = np.diag(weights)
    r2mat = np.diag(grids*grids)

    poly = np.zeros((n + 1, grids.shape[0]))
    # worland value
    for i in range(n + 1):
        norm = wb.worland_norm(i, l)
        tmp1 = special.eval_jacobi(i, -0.5, l - 0.5, 2.0*grids**2 - 1.0)
        poly[i, :] = grids**l * tmp1 * norm
    tormat = l*(l+1)*np.linalg.multi_dot([poly, r2mat, wmat, poly.transpose()])
    return tormat


def energy_weight_pol(l, n):
    """Compute the energy weight matrix at l, given maximum radial mode up to n, for a velocity field"""
    rdegree = l + 2 * n
    nr = rdegree + 2 + (rdegree + 2) % 2

    grids, weights = energy_quadrature(nr)
    wmat = np.diag(weights)

    poly = np.zeros((n + 1, grids.shape[0]))
    diff = np.zeros((n + 1, grids.shape[0]))

    # worland value and diff(r * worland)
    for i in range(n + 1):
        norm = wb.worland_norm(i, l)
        tmp1 = special.eval_jacobi(i, -0.5, l - 0.5, 2.0 * grids ** 2 - 1.0)
        poly[i, :] = grids ** l * tmp1 * norm
        if i > 0:
            tmp2 = special.eval_jacobi(i - 1, 0.5, l + 0.5, 2.0 * grids ** 2 - 1.0)
        else:
            tmp2 = 0
        diff[i, :] = (2.0 * (1.0 - 0.5 + l - 0.5 + i) * grids ** (l + 2) * tmp2 + (l + 1) * grids ** l * tmp1) * norm
    polmat = l ** 2 * (l + 1) ** 2 * np.linalg.multi_dot([poly, wmat, poly.transpose()]) + l * (
                l + 1) * np.linalg.multi_dot([diff, wmat, diff.transpose()])
    return polmat


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


def alpha(l, m):
    return np.sqrt(((l - m) * (l + m)) / ((2.0 * l - 1.0) * (2.0 * l + 1.0)) * (l >= m))


def lgClm(l, m):
    return 0.5 * (special.loggamma(l - m + 1) - special.loggamma(l + m + 1) + np.log((2. * l + 1.) / 4. / np.pi))


def Plm(m, lmax, theta):
    """ compute spherical harmonics (excluding the e^{im\phi} term), evaluated on theta grid  """
    cos_theta = np.cos(theta).reshape(-1, 1)

    if lmax == m:
        C = np.exp(-0.5 * special.loggamma(2 * m + 1)) * np.sqrt((2. * m + 1.) / (4. * np.pi))
        val = C * (-1.) ** m * special.factorial2(2 * m - 1) * np.exp(m / 2. * np.log(1. - np.cos(theta) ** 2))
        return val.reshape(-1, 1)
    elif lmax > m:
        vals = []
        C = np.exp(-0.5 * special.loggamma(2 * m + 1)) * np.sqrt((2. * m + 1.) / (4. * np.pi))
        tmp = C * (-1.) ** m * special.factorial2(2 * m - 1) * (1. - np.cos(theta) ** 2) ** (m / 2.)
        vals.append(tmp.reshape(-1, 1))
        for l in range(m, lmax):
            if l > m:
                tmp = (cos_theta * vals[l - m] - alpha(l, m) * vals[l - m - 1]) / alpha(l + 1, m)
            else:
                tmp = cos_theta * vals[l - m] / alpha(l + 1, m)
            vals.append(tmp)
        return np.concatenate(vals, axis=1)


def PlmDivSin(m, lmax, theta):
    """ compute value of C_l^m P_l^m(\cos\theta) / \sin\theta, on theta, for a single m
    Only for m > 0 !"""
    if m > 0:
        plm_m1 = Plm(m - 1, lmax, theta)
        plm_p1 = Plm(m + 1, lmax, theta)

        val = []
        for l in range(m, lmax + 1):
            if l - 1 >= m + 1:
                tmp = -1. / (2. * m) * np.exp(lgClm(l, m) - lgClm(l - 1, m + 1)) * plm_p1[:, l - m - 2] \
                      - (l + m - 1.) * (l + m) / (2. * m) * np.exp(lgClm(l, m) - lgClm(l - 1, m - 1)) * plm_m1[:, l - m]
            else:
                tmp = - (l + m - 1.) * (l + m) / (2. * m) * np.exp(lgClm(l, m) - lgClm(l - 1, m - 1)) * plm_m1[:, l - m]
            val.append(tmp.reshape(-1, 1))
        return np.concatenate(val, axis=1)
    else:
        return np.zeros((theta.shape[0], lmax-m+1))


def DthetaPlm(m, lmax, Plm, PlmDivSin, theta):
    """ compute value of D_theta C_l^m P_l^m(\cos\theta), for a single m
    m = 0 compute in a different way """

    sin_theta = np.sin(theta).reshape(-1, 1)
    # m=0
    if m == 0:
        val = [np.zeros(sin_theta.shape), -0.5 * np.sqrt(3. / np.pi) * sin_theta]
        for l in range(1, lmax):
            tmp = np.exp(lgClm(l + 1, 0) - lgClm(l - 1, 0)) * val[l - 1] - \
                  (2. * l + 1.) * np.exp(lgClm(l + 1, 0) - lgClm(l, 0)) * sin_theta * Plm[:, l].reshape(-1, 1)
            val.append(tmp.reshape(-1, 1))
        DthetaPlm = np.concatenate(val, axis=1)
    # m>0
    else:
        val = []
        for l in range(m, lmax + 1):
            if l > m:
                tmp = -(l + 1.) * alpha(l, m) * PlmDivSin[:, l - m - 1] + l * alpha(l + 1, m) * PlmDivSin[:, l + 1 - m]
            else:
                tmp = l * alpha(l + 1, m) * PlmDivSin[:, l + 1 - m]
            val.append(tmp.reshape(-1, 1))
        DthetaPlm = np.concatenate(val, axis=1)
    return DthetaPlm


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

