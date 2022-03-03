import numpy as np
import os

from models import MagnetoCoriolis
from operators.polynomials import SphericalHarmonicMode
from utils import *


def compute_spectrum(A, B, parity, name):
    w = full_spectrum(A, B)
    path = f"../data/Malkus/{parity}/eigs"
    os.makedirs(path, exist_ok=True)
    np.save(f"{path}/{name}", w)
    return w


for res in [(21, 23, 3),
            (31, 33, 3),
            (41, 43, 3)]:
    print(res)
    nr, maxnl, m = res
    field_modes = [SphericalHarmonicMode("tor", 1, 0, "2 Sqrt[pi/3] r")]
    model = MagnetoCoriolis(nr, maxnl, m, inviscid=True)
    op_dp, op_qp = model.setup_operator(field_modes, setup_eigen=True, magnetic_ekman=0, elsasser=1,
                                        parity=True, u_parity='same')
    compute_spectrum(op_qp[0], op_qp[1], 'QP', f'feigs_{nr-1}_{maxnl-m}_{m}_no_inertial.npy')
