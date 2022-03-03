import numpy as np
import scipy.sparse as scsp

from quicc.base import utils
import quicc.geometry.worland.worland_basis as wb
import quicc.geometry.spherical.sphere_radius_boundary_worland as radbc


def i2_nobc(nr, l, bc, coeff = 1.0):
    """Create operator for 2nd integral r^l P_n^{-1/2,l-1/2}(2r^2 -1), using lines at N+1, N+2, no bc line"""

    ns = np.arange(0, nr + 2)
    offsets = np.arange(-2, 3)
    nzrow = 1

    # Generate 2nd subdiagonal
    def d_2(n):
        if l == 0:
            val = wb.worland_norm_row(n, l, -2) / ((l + 2.0 * n - 3.0) * (l + 2.0 * n - 1.0))
            val[0] = wb.worland_norm_row(n[0:1], l, -2) * 4.0 * (l + 1.0) / ((l + 1.0) * (l + 2.0) * (l + 3.0))
        else:
            val = wb.worland_norm_row(n, l, -2) * 4.0 * (l + n - 2.0) * (l + n - 1.0) / (
                        (l + 2.0 * n - 4.0) * (l + 2.0 * n - 3.0) * (l + 2.0 * n - 2.0) * (l + 2.0 * n - 1.0))
        return val

    # Generate 1st subdiagonal
    def d_1(n):
        return -wb.worland_norm_row(n, l, -1) * 8.0 * l * (l + n - 1.0) / (
                    (l + 2.0 * n - 3.0) * (l + 2.0 * n - 2.0) * (l + 2.0 * n - 1.0) * (l + 2.0 * n + 1.0))

    # Generate diagonal
    def d0(n):
        return wb.worland_norm_row(n, l, 0) * 2.0 * (2.0 * l ** 2 - 4.0 * l * n - 4.0 * n ** 2 + 1.0) / (
                    (l + 2.0 * n - 2.0) * (l + 2.0 * n - 1.0) * (l + 2.0 * n + 1.0) * (l + 2.0 * n + 2.0))

    # Generate 1st superdiagonal
    def d1(n):
        return wb.worland_norm_row(n, l, 1) * 2.0 * l * (2.0 * n + 1.0) * (2.0 * l + 2.0 * n + 1.0) / (
                    (l + n) * (l + 2.0 * n - 1.0) * (l + 2.0 * n + 1.0) * (l + 2.0 * n + 2.0) * (l + 2.0 * n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return wb.worland_norm_row(n, l, 2) * (2.0 * n + 1.0) * (2.0 * n + 3.0) * (2.0 * l + 2.0 * n + 1.0) * (
                    2.0 * l + 2.0 * n + 3.0) / (
                           4.0 * (l + n) * (l + n + 1.0) * (l + 2.0 * n + 1.0) * (l + 2.0 * n + 2.0) * (
                               l + 2.0 * n + 3.0) * (l + 2.0 * n + 4.0))

    ds = [d_2, d_1, d0, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap=False)

    mat = coeff * scsp.diags(diags, offsets, format='coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)

    return radbc.constrain(mat, l, bc)


def i2qm_nobc(nr, l, bc, coeff = 1.0):
    """Create operator for 2nd integral of Q r^{l-1} P_n^{-1/2,l-3/2}(2r^2 -1). no tau line."""

    ns = np.arange(0, nr + 2)
    offsets = np.arange(-1, 3)
    nzrow = 1

    # Generate 1st subdiagonal
    def d_1(n):
        return -wb.worland_norm_row(n, l, -1, -1) * 8.0 * (l + n - 2.0) * (l + n - 1.0) / (
                    (l + 2.0 * n - 3.0) * (l + 2.0 * n - 2.0) * (l + 2.0 * n - 1.0))

    # Generate main diagonal
    def d0(n):
        return wb.worland_norm_row(n, l, 0, -1) * 4.0 * (l + n - 1.0) * (2.0 * l - 2.0 * n - 1.0) / (
                    (l + 2.0 * n - 2.0) * (l + 2.0 * n - 1.0) * (l + 2.0 * n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return wb.worland_norm_row(n, l, 1, -1) * 2.0 * (2.0 * n + 1.0) * (4.0 * l + 2.0 * n - 1.0) / (
                    (l + 2.0 * n - 1.0) * (l + 2.0 * n + 1.0) * (l + 2.0 * n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return wb.worland_norm_row(n, l, 2, -1) * (2.0 * n + 1.0) * (2.0 * n + 3.0) * (2.0 * l + 2.0 * n + 1.0) / (
                    (l + n) * (l + 2.0 * n + 1.0) * (l + 2.0 * n + 2.0) * (l + 2.0 * n + 3.0))

    ds = [d_1, d0, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap=False)

    mat = coeff * scsp.diags(diags, offsets, format='coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)
    return radbc.constrain(mat, l, bc)


def i2qp_nobc(nr, l, bc, coeff = 1.0):
    """Create operator for 2nd integral of Q r^{l+1} P_n^{-1/2,l+1/2}(2r^2 -1). No tau line"""
    ns = np.arange(0, nr + 2)
    offsets = np.arange(-2,2)
    nzrow = 1

    # Generate 2nd subdiagonal
    def d_2(n):
        return wb.worland_norm_row(n,l,-2,1)*4.0*(l + n - 1.0)*(2.0*l + 2.0*n - 1.0)/((l + 2.0*n - 3.0)*(l + 2.0*n - 2.0)*(l + 2.0*n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -wb.worland_norm_row(n,l,-1,1)*2.0*(4.0*l**2 + 4.0*l - 4.0*n**2 + 4.0*n + 3.0)/((l + 2.0*n - 2.0)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0))

    # Generate main diagonal
    def d0(n):
        return -wb.worland_norm_row(n,l,0,1)*(2.0*l + 2.0*n + 1.0)*(8.0*l*n + 4.0*n**2 + 4.0*n - 3.0)/((l + n)*(l + 2.0*n - 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -wb.worland_norm_row(n,l,1,1)*(2.0*n + 1.0)**2*(2.0*l + 2.0*n + 1.0)*(2.0*l + 2.0*n + 3.0)/(2.0*(l + n)*(l + n + 1.0)*(l + 2.0*n + 1.0)*(l + 2.0*n + 2.0)*(l + 2.0*n + 3.0))

    ds = [d_2, d_1, d0, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets, has_wrap = False)

    mat = coeff*scsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1)*mat*radbc.restrict_eye(mat.shape[1], 'cr', 1)
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)
    return radbc.constrain(mat, l, bc)
