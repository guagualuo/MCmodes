""" This module contains is used to test a kinematic dynamo problem
    Compares with Table 1 of Li et al 2010 """
from operators.physical_operators import *
from operators.polynomials import SphericalHarmonicMode
from utils import *

""" negative here because the convention of induction term is curl(A x B), B being the fixed field """
t10 = SphericalHarmonicMode("tor", 1, 0, "-Sqrt[4 pi / 3] Sin[pi r]")
s10 = SphericalHarmonicMode("pol", 1, 0, "-17/100*Sqrt[4 pi / 3] Sin[pi r]")
s20 = SphericalHarmonicMode("pol", 2, 0, "-13/100*Sqrt[4 pi / 5] r Sin[pi r]")

nr, maxnl, m = 41, 41, 1
n_grid = 100
with Timer("init op"):
    transform = WorlandTransform(nr, maxnl, m, n_grid)

beta_modes = [t10, s10]
Rm = 160
with Timer("build op"):
    ind_op = induction(transform, beta_modes)
    quasi_inverse = induction_quasi_inverse(nr, maxnl, m)
    ind_op = quasi_inverse @ ind_op
    mag_diff = mag_diffusion(nr, maxnl, m)
    mass = induction_mass(nr, maxnl, m)

A = Rm*ind_op + mag_diff
B = mass
w, _ = single_eig(A, B, target=0.313-34.84j)
print(w[0])
