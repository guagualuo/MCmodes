import matplotlib.pyplot as plt
import numpy as np
from typing import List
import scipy.sparse as scsp

from operators.worland_operator import WorlandTransform
from operators.polynomials import SphericalHarmonicMode
from utils import Timer
import quicc.geometry.spherical.sphere_worland as geo
from quicc.geometry.spherical.sphere_boundary_worland import no_bc


def induction(transform: WorlandTransform, beta_modes: List[SphericalHarmonicMode]):
    """ Induction term curl (u x B_0), in which B_0 is the background field.
        [ r.curl2(t_a x B_0), r.curl2(s_a x B_0)
          r.curl1(t_a x B0), r.curl1(s_a x B_0)]"""
    tt = sum([transform.curl2tt(mode) + transform.curl2ts(mode) for mode in beta_modes])
    ts = sum([transform.curl2st(mode) + transform.curl2ss(mode) for mode in beta_modes])
    st = sum([transform.curl1tt(mode) + transform.curl1ts(mode) for mode in beta_modes])
    ss = sum([transform.curl1st(mode) + transform.curl1ss(mode) for mode in beta_modes])
    return scsp.bmat([[tt, ts], [st, ss]], format='csc')


def induction_quasi_inverse(nr, maxnl, m):
    return scsp.block_diag((geo.i2(nr, maxnl, m, no_bc(), l_zero_fix='zero'),
                            geo.i2(nr, maxnl, m, no_bc(), l_zero_fix='zero')))


def induction_mass(nr, maxnl, m):
    return scsp.block_diag((geo.i2(nr, maxnl, m, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                            geo.i2(nr, maxnl, m, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero')))


def induction_diffusion(nr, maxnl, m, bc=True):
    """ Build the dissipation matrix for the magnetic field, insulating boundary condition """
    if bc:
        return scsp.block_diag((geo.i2lapl(nr, maxnl, m, bc={0: 10}, with_sh_coeff='laplh', l_zero_fix='set'),
                                geo.i2lapl(nr, maxnl, m, bc={0: 13}, with_sh_coeff='laplh', l_zero_fix='set')))
    else:
        return scsp.block_diag((geo.i2lapl(nr, maxnl, m, bc=no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                                geo.i2lapl(nr, maxnl, m, bc=no_bc(), with_sh_coeff='laplh', l_zero_fix='zero')))


def lorentz1(transform: WorlandTransform, modes: List[SphericalHarmonicMode]):
    """ Lorentz term (curl B_0) x b, in which B_0 is the background field.
            [ r.curl1(J0 x b), r.curl1(J0 x b)
              r.curl2(J0 x b), r.curl2(J0 x b)]"""
    curl_modes = [mode.curl for mode in modes]
    tt = sum([transform.curl1tt(mode) + transform.curl1ts(mode) for mode in curl_modes])
    ts = sum([transform.curl1st(mode) + transform.curl1ss(mode) for mode in curl_modes])
    st = sum([transform.curl2tt(mode) + transform.curl2ts(mode) for mode in curl_modes])
    ss = sum([transform.curl2st(mode) + transform.curl2ss(mode) for mode in curl_modes])
    return -scsp.bmat([[tt, ts], [st, ss]], format='csc')


def lorentz2(transform: WorlandTransform, modes: List[SphericalHarmonicMode]):
    """ Lorentz term (curl b) x B_0, in which B_0 is the background field.
            [ r.curl1(j x b), r.curl1(j x b)
              r.curl2(j x b), r.curl2(j x b)]"""
    tt = sum([transform.curl1curltt(mode) + transform.curl1curlts(mode) for mode in modes])
    ts = sum([transform.curl1curlst(mode) + transform.curl1curlss(mode) for mode in modes])
    st = sum([transform.curl2curltt(mode) + transform.curl2curlts(mode) for mode in modes])
    ss = sum([transform.curl2curlst(mode) + transform.curl2curlss(mode) for mode in modes])
    return scsp.bmat([[tt, ts], [st, ss]], format='csc')


def lorentz(transform: WorlandTransform, modes: List[SphericalHarmonicMode]):
    return lorentz1(transform, modes) + lorentz2(transform, modes)


if __name__ == "__main__":
    nr, maxnl, m = 11, 11, 1
    n_grid = 120
    with Timer("init op"):
        transform = WorlandTransform(nr, maxnl, m, n_grid)
    beta_mode = SphericalHarmonicMode("tor", 1, 0, "2 Sqrt[pi/3] r")
    with Timer("build induction"):
        ind_op = induction(transform, [beta_mode])
        ind_op[np.abs(ind_op) < 1e-12] = 0
    plt.spy(ind_op).set_marker('.')
    plt.show()
