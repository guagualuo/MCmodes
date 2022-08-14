import numpy as np
import scipy.sparse as spsp

import quicc.base.utils as utils
import quicc.geometry.spherical.shell_radius_boundary as radbc


def i2_nobc(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral T_n(x), without tau lines"""

    ns = np.arange(0, nr+2)
    offsets = np.arange(-2,3,2)
    nzrow = 1

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2/(4.0*n*(n - 1.0))

    # Generate diagonal
    def d0(n):
        return -a**2/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return d_2(n+1.0)

    ds = [d_2, d0, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)

    return radbc.constrain(mat, bc)


def i2r2_nobc(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r^2 T_n(x), without tau lines"""

    ns = np.arange(0, nr+2)
    offsets = np.arange(-4,5)
    nzrow = 1

    # Generate 4th subdiagonal
    def d_4(n):
        return a**4/(16.0*n*(n - 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return (a**3*b)/(4.0*n*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return (a**2*(2.0*b**2*n + a**2 + 2.0*b**2))/(8.0*n*(n - 1.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -(a**3*b)/(4.0*n*(n + 1.0))

    # Generate main diagonal
    def d0(n):
        return -(a**2*(a**2 + 4.0*b**2))/(8.0*(n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return d_1(n - 1.0)

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*(a**2 - 2.0*b**2*n + 2.0*b**2)/(8.0*n*(n - 1.0)*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return d_3(n + 1.0)

    # Generate 4th superdiagonal
    def d4(n):
        return d_4(n + 1.0)

    ds = [d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)

    return radbc.constrain(mat, bc)


def i2r3_nobc(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r^3 T_n(x), without tau lines"""

    ns = np.arange(0, nr+2)
    offsets = np.arange(-5,6)
    nzrow = 1

    # Generate 5th subdiagonal
    def d_5(n):
        return a**5/(32.0*n*(n - 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return 3.0*a**4*b/(16.0*n*(n - 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return a**3*(a**2*n + 3.0*a**2 + 12.0*b**2*n + 12.0*b**2)/(32.0*n*(n - 1.0)*(n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2*b*(3.0*a**2 + 2.0*b**2*n + 2.0*b**2)/(8.0*n*(n - 1.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -a**3*(a**2 + 6.0*b**2)/(16.0*n*(n + 1.0))

    # Generate main diagonal
    def d0(n):
        return -a**2*b*(3.0*a**2 + 4.0*b**2)/(8.0*(n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -a**3*(a**2 + 6.0*b**2)/(16.0*n*(n - 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*b*(3.0*a**2 - 2.0*b**2*n + 2.0*b**2)/(8.0*n*(n - 1.0)*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return a**3*(a**2*n - 3.0*a**2 + 12.0*b**2*n - 12.0*b**2)/(32.0*n*(n - 1.0)*(n + 1.0))

    # Generate 4th superdiagonal
    def d4(n):
        return 3.0*a**4*b/(16.0*n*(n + 1.0))

    # Generate 5th superdiagonal
    def d5(n):
        return a**5/(32.0*n*(n + 1.0))

    ds = [d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)

    return radbc.constrain(mat, bc)


def i2r4lapl(nr, l, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r^3 laplacian_l T_n(x)"""
    ns = np.arange(0, nr)
    offsets = np.arange(-4, 5)
    nzrow = 1

    def d_4(n):
        return -a**4*(l - n + 4)*(l + n - 3)/(16*n*(n - 1))

    def d_3(n):
        return -a**3*b*(l**2 + l - 2*n**2 + 11*n - 15)/(4*n*(n - 1))

    def d_2(n):
        return -a**2*(a**2*l**2 + a**2*l - 2*a**2*n**3 + 6*a**2*n**2 + 2*a**2*n - 12*a**2 + 2*b**2*l**2*n + 2*b**2*l**2 + 2*b**2*l*n + 2*b**2*l - 12*b**2*n**3 + 36*b**2*n**2 - 48*b**2)/(8*n*(n - 1)*(n + 1))

    def d_1(n):
        return a*b*(a**2*l**2 + a**2*l + 6*a**2*n**2 - 3*a**2*n - 15*a**2 + 8*b**2*n**2 - 4*b**2*n - 12*b**2)/(4*n*(n + 1))

    def d0(n):
        return (a**4*l**2 + a**4*l + 3*a**4*n**2 - 9*a**4 + 4*a**2*b**2*l**2 + 4*a**2*b**2*l + 24*a**2*b**2*n**2 - 48*a**2*b**2 + 8*b**4*n**2 - 8*b**4)/(8*(n - 1)*(n + 1))

    def d1(n):
        return a*b*(a**2*l**2 + a**2*l + 6*a**2*n**2 + 3*a**2*n - 15*a**2 + 8*b**2*n**2 + 4*b**2*n - 12*b**2)/(4*n*(n - 1))

    def d2(n):
        return a**2*(a**2*l**2 + a**2*l + 2*a**2*n**3 + 6*a**2*n**2 - 2*a**2*n - 12*a**2 - 2*b**2*l**2*n + 2*b**2*l**2 - 2*b**2*l*n + 2*b**2*l + 12*b**2*n**3 + 36*b**2*n**2 - 48*b**2)/(8*n*(n - 1)*(n + 1))

    def d3(n):
        return -a**3*b*(l**2 + l - 2*n**2 - 11*n - 15)/(4*n*(n + 1))

    def d4(n):
        return -a**4*(l - n - 3)*(l + n + 4)/(16*n*(n + 1))

    ds = [d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff * spsp.diags(diags, offsets, format='coo')
    return radbc.constrain(mat, bc)


def i2r3d1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r^3 D_r T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-4,5)
    nzrow = 1

    def d_4(n):
        return a**4*(n - 4)/(16*n*(n - 1))

    # Generate 3rd subdiagonal
    def d_3(n):
        return 3*a**3*b*(n - 3)/(8*n*(n - 1))

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2*(n - 2)*(a**2*n + 2*a**2 + 6*b**2*n + 6*b**2)/(8*n*(n - 1)*(n + 1))

    # Generate 1st subdiagonal
    def d_1(n):
        return a*b*(3*a**2*n + 9*a**2 + 4*b**2*n + 4*b**2)/(8*n*(n + 1))

    # Generate main diagonal
    def d0(n):
        return 3*a**2*(a**2 + 4*b**2)/(8*(n - 1)*(n + 1))

    # Generate 1st superdiagonal
    def d1(n):
        return -a*b*(3*a**2*n - 9*a**2 + 4*b**2*n - 4*b**2)/(8*n*(n - 1))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*(n + 2)*(a**2*n - 2*a**2 + 6*b**2*n - 6*b**2)/(8*n*(n - 1)*(n + 1))

    # Generate 3rd superdiagonal
    def d3(n):
        return -3*a**3*b*(n + 3)/(8*n*(n + 1))

    def d4(n):
        return -a**4*(n + 4)/(16*n*(n + 1))

    ds = [d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)


def i2r3d1_nobc(nr, a, b, bc, coeff=1.0):
    """Create operator for 2nd integral of r^3 D_r T_n(x), without tau lines"""

    ns = np.arange(0, nr+2)
    offsets = np.arange(-4, 5)
    nzrow = 1

    def d_4(n):
        return a ** 4 * (n - 4) / (16 * n * (n - 1))

    # Generate 3rd subdiagonal
    def d_3(n):
        return 3 * a ** 3 * b * (n - 3) / (8 * n * (n - 1))

    # Generate 2nd subdiagonal
    def d_2(n):
        return a ** 2 * (n - 2) * (a ** 2 * n + 2 * a ** 2 + 6 * b ** 2 * n + 6 * b ** 2) / (8 * n * (n - 1) * (n + 1))

    # Generate 1st subdiagonal
    def d_1(n):
        return a * b * (3 * a ** 2 * n + 9 * a ** 2 + 4 * b ** 2 * n + 4 * b ** 2) / (8 * n * (n + 1))

    # Generate main diagonal
    def d0(n):
        return 3 * a ** 2 * (a ** 2 + 4 * b ** 2) / (8 * (n - 1) * (n + 1))

    # Generate 1st superdiagonal
    def d1(n):
        return -a * b * (3 * a ** 2 * n - 9 * a ** 2 + 4 * b ** 2 * n - 4 * b ** 2) / (8 * n * (n - 1))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a ** 2 * (n + 2) * (a ** 2 * n - 2 * a ** 2 + 6 * b ** 2 * n - 6 * b ** 2) / (8 * n * (n - 1) * (n + 1))

    # Generate 3rd superdiagonal
    def d3(n):
        return -3 * a ** 3 * b * (n + 3) / (8 * n * (n + 1))

    def d4(n):
        return -a ** 4 * (n + 4) / (16 * n * (n + 1))

    ds = [d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff * spsp.diags(diags, offsets, format='coo')
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)
    mat = radbc.restrict_eye(mat.shape[0], 'rt', 1) * mat * radbc.restrict_eye(mat.shape[1], 'cr', 1)
    return radbc.constrain(mat, bc)


def i2r4d1(nr, a, b, bc, coeff=1.0):
    """Create operator for 2nd integral of r^3 D_r T_n(x), without tau lines"""

    ns = np.arange(0, nr)
    offsets = np.arange(-5, 6)
    nzrow = 1

    def d_5(n):
        return a**5*(n - 5)/(32*n*(n - 1))

    def d_4(n):
        return a**4*b*(n - 4)/(4*n*(n - 1))

    def d_3(n):
        return a**3*(n - 3)*(3*a**2*n + 5*a**2 + 24*b**2*n + 24*b**2)/(32*n*(n - 1)*(n + 1))

    def d_2(n):
        return a**2*b*(n - 2)*(a**2*n + 2*a**2 + 2*b**2*n + 2*b**2)/(2*n*(n - 1)*(n + 1))

    def d_1(n):
        return a*(a**4*n + 5*a**4 + 12*a**2*b**2*n + 36*a**2*b**2 + 8*b**4*n + 8*b**4)/(16*n*(n + 1))

    def d0(n):
        return a**2*b*(3*a**2 + 4*b**2)/(2*(n - 1)*(n + 1))

    def d1(n):
        return -a*(a**4*n - 5*a**4 + 12*a**2*b**2*n - 36*a**2*b**2 + 8*b**4*n - 8*b**4)/(16*n*(n - 1))

    def d2(n):
        return -a**2*b*(n + 2)*(a**2*n - 2*a**2 + 2*b**2*n - 2*b**2)/(2*n*(n - 1)*(n + 1))

    def d3(n):
        return -a**3*(n + 3)*(3*a**2*n - 5*a**2 + 24*b**2*n - 24*b**2)/(32*n*(n - 1)*(n + 1))

    def d4(n):
        return -a**4*b*(n + 4)/(4*n*(n + 1))

    def d5(n):
        return -a**5*(n + 5)/(32*n*(n + 1))

    ds = [d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff * spsp.diags(diags, offsets, format='coo')
    return radbc.constrain(mat, bc)
