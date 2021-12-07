""" This script compute eigenvalues for m=0 perturbations, to search for torsional modes """
import matplotlib.pyplot as plt
import numpy as np
import os

from models import MagnetoCoriolis
from operators.polynomials import SphericalHarmonicMode
from utils import *
from scipy.linalg import eig


def compute_spectrum(A, B, name):
    w = full_spectrum(A, B)
    path = "../data/torsional_modes/poloidal_dp/eigs"
    os.makedirs(path, exist_ok=True)
    np.save(f"{path}/{name}", w)
    return w


mag_ideal = True
if mag_ideal:
    case = "ideal"
else:
    case = 'dissipative'

Eeta = 1e-6
elsasser = 1
for res in [(21, 21, 0),
            (31, 31, 0)]:
    nr, maxnl, m = res
    n_grid = 60
    with Timer("build operator"):
        field_modes = [SphericalHarmonicMode("pol", 1, 0, "1/5 Sqrt[pi/3] r(5-3r^2)")]
        # field_modes = [SphericalHarmonicMode("pol", 2, 0, "Sqrt[5]/(8 Sqrt[2]) r^2(7-5r^2)")]
        model = MagnetoCoriolis(nr, maxnl, m, n_grid, inviscid=True, mag_ideal=mag_ideal)
        op_dp, op_qp = model.setup_operator(field_modes=field_modes, setup_eigen=True, magnetic_ekman=Eeta, elsasser=elsasser,
                                            parity=True, u_parity='opposite')
    name = f"feigs_E{-int(np.log10(Eeta))}_{nr - 1}_{maxnl-1}_{case}.npy"
    compute_spectrum(op_dp[0], op_dp[1], name)
