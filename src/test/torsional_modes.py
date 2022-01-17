""" This script compute eigenvalues for m=0 perturbations, to search for torsional modes """
import matplotlib.pyplot as plt
import numpy as np
import os

from models import TorsionalOscillation, IdealTorsionalOscillation
from operators.polynomials import SphericalHarmonicMode
from utils import *


def compute_spectrum(A, B, config, name):
    w = full_spectrum(A, B)
    path = f"../data/torsional_modes/{config}/eigs"
    os.makedirs(path, exist_ok=True)
    np.save(f"{path}/{name}", w)
    return w


# resolutions = [(16, 31), (21, 41)]
resolutions = [(41, 81)]

ideal = False

if 0:
    field_modes = [SphericalHarmonicMode("pol", 1, 0, "Sqrt[7/46]/2 r(5-3r^2)")]

    if ideal:
        config = 'poloidal_dp_ideal'
        Le = 1e-3
        for res in resolutions:
            nr, maxnl = res
            model = IdealTorsionalOscillation(nr, maxnl)
            op_dp, op_qp = model.setup_operator(field_modes=field_modes, setup_eigen=True, lehnert=Le,
                                                parity=True, u_parity='opposite')
            name = f"feigs_Le{-np.log10(Le):.2f}_{nr - 1}_{maxnl - 1}.npy"
            compute_spectrum(op_dp[0], op_dp[1], config, name)
    else:
        config = 'poloidal_dp'
        for le in [-2, -2.5, -3, -3.5, -4]:
            Le = 10**le
            print(Le)
            Lu = 2/Le
            for res in resolutions:
                nr, maxnl = res
                model = TorsionalOscillation(nr, maxnl, inviscid=True)
                op_dp, op_qp = model.setup_operator(field_modes=field_modes, setup_eigen=True,
                                                    lehnert=Le, lundquist=Lu,
                                                    parity=True, u_parity='opposite')
                name = f"feigs_Le{-np.log10(Le):.2f}_{nr - 1}_{maxnl - 1}.npy"
                compute_spectrum(op_dp[0], op_dp[1], config, name)


if 1:
    field_modes = [SphericalHarmonicMode("pol", 2, 0, "(5 Sqrt[3/182])/14 * 7/40 r^2 (157 - 296 r^2 + 143 r^4)")]

    if ideal:
        config = 'poloidal_qp_ideal'
        Le = 1e-3
        for res in resolutions:
            nr, maxnl = res
            model = IdealTorsionalOscillation(nr, maxnl)
            op_dp, op_qp = model.setup_operator(field_modes=field_modes, setup_eigen=True, lehnert=Le,
                                                parity=True, u_parity='same')
            name = f"feigs_Le{-np.log10(Le):.2f}_{nr - 1}_{maxnl - 1}.npy"
            compute_spectrum(op_dp[0], op_dp[1], config, name)
    else:
        config = 'poloidal_qp'
        for le in [-2, -2.5, -3, -3.5, -4]:
            Le = 10**le
            print(Le)
            Lu = 2/Le
            for res in resolutions:
                nr, maxnl = res
                model = TorsionalOscillation(nr, maxnl, inviscid=True, galerkin=False)
                op_dp, op_qp = model.setup_operator(field_modes=field_modes, setup_eigen=True,
                                                    lehnert=Le, lundquist=Lu,
                                                    parity=True, u_parity='same')
                name = f"feigs_Le{-np.log10(Le):.2f}_{nr - 1}_{maxnl - 1}.npy"
                compute_spectrum(op_qp[0], op_qp[1], config, name)

if 0:
    field_modes = [SphericalHarmonicMode("pol", 1, 0, "5 Sqrt[21/13982] r(5-3r^2)"),
                   SphericalHarmonicMode("pol", 2, 0, "5/20 * Sqrt[21/13982] r^2(7-5r^2)")]
    if ideal:
        config = 'poloidal_mixed_ideal'
        Le = 1e-3
        for res in resolutions:
            nr, maxnl = res
            model = IdealTorsionalOscillation(nr, maxnl)
            A, B = model.setup_operator(field_modes=field_modes, setup_eigen=True, lehnert=Le)
            name = f"feigs_Le{-np.log10(Le):.2f}_{nr - 1}_{maxnl - 1}.npy"
            compute_spectrum(A, B, config, name)
    else:
        config = 'poloidal_mixed'
        for le in [-2, -2.5, -3, -3.5, -4]:
            Le = 10**le
            print(Le)
            Lu = 2/Le
            for res in resolutions:
                nr, maxnl = res
                model = TorsionalOscillation(nr, maxnl, inviscid=True)
                A, B = model.setup_operator(field_modes=field_modes, setup_eigen=True, lehnert=Le, lundquist=Lu)
                name = f"feigs_Le{-np.log10(Le):.2f}_{nr - 1}_{maxnl - 1}.npy"
                compute_spectrum(A, B, config, name)

