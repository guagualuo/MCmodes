""" This script compute eigenvalues for the ideal MC problem """
import matplotlib.pyplot as plt
import numpy as np
import os

from models import IdealMagnetoCoriolis, MagnetoCoriolis
from operators.polynomials import SphericalHarmonicMode
from utils import *
from scipy.linalg import eig


def compute_spectrum(A, B, name):
    w = full_spectrum(A, B)
    path = "../data/idealMC/toroidal_qp/eigs"
    os.makedirs(path, exist_ok=True)
    np.save(f"{path}/{name}", w)
    return w


Le = 1e-4
elsasser = 1
for res in [(20, 20, 1),
            (31, 31, 1)]:
    nr, maxnl, m = res
    n_grid = 60
    with Timer("build operator"):
        field_modes = [SphericalHarmonicMode("pol", 1, 0, "1/5 Sqrt[pi/3] r(5-3r^2)")]
        # field_modes = [SphericalHarmonicMode("pol", 2, 0, "Sqrt[5]/(8 Sqrt[2]) r^2(7-5r^2)")]
        # field_modes = [SphericalHarmonicMode("tor", 1, 0, "3 Sqrt[pi] r(1-r^2)")]
        model = IdealMagnetoCoriolis(nr, maxnl, m, n_grid)
        op_dp, op_qp = model.setup_operator(field_modes=field_modes, setup_eigen=True, lehnert=Le,
                                            parity=True, u_parity='opposite')
        # model = MagnetoCoriolis(nr, maxnl, m, n_grid, galerkin=True, boundary_condition=True, ideal=False)
        # op_dp, op_qp = model.setup_operator(field_modes=field_modes, setup_eigen=True, elsasser=1, magnetic_ekman=Le**2,
        #                                     parity=True, u_parity='same')
    name = f"feigs_{nr - 1}_{maxnl-1}_{m}.npy"
    clu = spla.splu(op_qp[0])
    compute_spectrum(op_qp[0], op_qp[1], name)
