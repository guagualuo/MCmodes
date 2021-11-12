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
