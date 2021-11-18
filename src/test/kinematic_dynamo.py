""" This module contains is used to test a kinematic dynamo problem
    Compares with Table 1 of Li et al 2010 """
from models import KinematicDynamo
from operators.polynomials import SphericalHarmonicMode
from utils import *

t10 = SphericalHarmonicMode("tor", 1, 0, "Sqrt[4 pi / 3] Sin[pi r]")
s10 = SphericalHarmonicMode("pol", 1, 0, "17/100*Sqrt[4 pi / 3] Sin[pi r]")
s20 = SphericalHarmonicMode("pol", 2, 0, "13/100*Sqrt[4 pi / 5] r Sin[pi r]")
LJt10 = SphericalHarmonicMode("tor", 1, 0, "8.107929179422066 * r(1 - r^2)")
LJs20 = SphericalHarmonicMode("pol", 2, 0, "1.193271237996972 * r^2 (1 - r^2)^2")

""" D.J. t1s1 """
nr, maxnl, m = 41, 41, 1
n_grid = 100
model = KinematicDynamo(nr, maxnl, m, n_grid=n_grid)
with Timer("build op"):
    A, B = model.setup_operator(flow_modes=[t10, s10], setup_eigen=True, Rm=160)

w, _ = single_eig(A, B, target=0.313-34.84j)
print(w[0])

""" Modified D.J. t1s2 """
nr, maxnl, m = 41, 41, 0
n_grid = 100
model = KinematicDynamo(nr, maxnl, m, n_grid=n_grid)
with Timer("build op"):
    operators = model.setup_operator(flow_modes=[LJt10, LJs20], setup_eigen=False)

Rm = 100
A = Rm*operators['induction'] + operators['diffusion']
B = operators['mass']
w, _ = single_eig(A, B, target=-6.9)
print(w[0])
