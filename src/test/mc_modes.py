""" This module contains benchmarks of MC modes """
from models import MagnetoCoriolis
from operators.polynomials import SphericalHarmonicMode
from utils import *


def test_eig(A, B, targets):
    for target  in targets:
        w, _ = single_eig(A, B, target=target, nev=1)
        print(f"target {target}, eig: {w[0]}")
        
# Eeta = 0
Eeta = 1e-10
elsasser = 1

""" Poloidal field m = 1 """
nr, maxnl, m = 41, 41, 1
n_grid = 100
field_modes = [SphericalHarmonicMode("pol", 1, 0, "1/5 Sqrt[pi/3] r(5-3r^2)")]
model = MagnetoCoriolis(nr, maxnl, m, n_grid, inviscid=True)

with Timer('build operators'):
    operators = model.setup_operator(field_modes=field_modes, setup_eigen=False)
# A, B = model.setup_eigen_problem(operators, magnetic_ekman=Eeta, elsasser=elsasser)
op_dp, op_qp = model.setup_eigen_problem(operators, magnetic_ekman=Eeta, elsasser=elsasser,
                                         parity=True, u_parity='opposite')

targets_dp = [0.5779j - 49.45,
              -1.595j - 19.85,
              74.24j - 192.6]
targets_qp = [-41.77j - 88.52]
test_eig(op_dp[0], op_dp[1], targets_dp)
test_eig(op_qp[0], op_qp[1], targets_qp)


""" Poloidal field m = 3 """
nr, maxnl, m = 43, 43, 3
n_grid = 100
field_modes = [SphericalHarmonicMode("pol", 1, 0, "1/5 Sqrt[pi/3] r(5-3r^2)")]
model = MagnetoCoriolis(nr, maxnl, m, n_grid, inviscid=True)

with Timer('build operators'):
    operators = model.setup_operator(field_modes=field_modes, setup_eigen=False)
A, B = model.setup_eigen_problem(operators, magnetic_ekman=Eeta, elsasser=elsasser)

targets = [-434.6j - 883.5,
           21.24j - 137.4]
test_eig(A, B, targets)


""" Malkus field m=1 """
nr, maxnl, m = 41, 41, 1
n_grid = 100
field_modes = [SphericalHarmonicMode("tor", 1, 0, "2 Sqrt[pi/3] r")]
model = MagnetoCoriolis(nr, maxnl, m, n_grid, inviscid=True)

with Timer('build operators'):
    operators = model.setup_operator(field_modes=field_modes, setup_eigen=False)
# A, B = model.setup_eigen_problem(operators, magnetic_ekman=Eeta, elsasser=elsasser)
op_dp, op_qp = model.setup_eigen_problem(operators, magnetic_ekman=Eeta, elsasser=elsasser,
                                         parity=True, u_parity='same')

targets = [8.267j - 71.5,
           339.0j - 674.9]
test_eig(op_qp[0], op_qp[1], targets)
#
""" Malkus field m=5 """
nr, maxnl, m = 45, 45, 5
n_grid = 100
field_modes = [SphericalHarmonicMode("tor", 1, 0, "2 Sqrt[pi/3] r")]
model = MagnetoCoriolis(nr, maxnl, m, n_grid, inviscid=True)

with Timer('build operators'):
    operators = model.setup_operator(field_modes=field_modes, setup_eigen=False)
A, B = model.setup_eigen_problem(operators, magnetic_ekman=Eeta, elsasser=elsasser)

targets = [-23.49j - 206.5,
           485.3j - 360.2]
test_eig(A, B, targets)


""" Toroidal m=1 """
nr, maxnl, m = 41, 41, 1
n_grid = 100
field_modes = [SphericalHarmonicMode("tor", 1, 0, "3 Sqrt[pi] r(1-r^2)")]
model = MagnetoCoriolis(nr, maxnl, m, n_grid, inviscid=True)

with Timer('build operators'):
    operators = model.setup_operator(field_modes=field_modes, setup_eigen=False)
A, B = model.setup_eigen_problem(operators, magnetic_ekman=Eeta, elsasser=elsasser)

targets = [0.1487j - 19.26,
           245.4j - 345.7]
test_eig(A, B, targets)


""" Toroidal m=3 """
nr, maxnl, m = 43, 43, 3
n_grid = 100
field_modes = [SphericalHarmonicMode("tor", 1, 0, "3 Sqrt[pi] r(1-r^2)")]
model = MagnetoCoriolis(nr, maxnl, m, n_grid, inviscid=True)

with Timer('build operators'):
    operators = model.setup_operator(field_modes=field_modes, setup_eigen=False)
A, B = model.setup_eigen_problem(operators, magnetic_ekman=Eeta, elsasser=elsasser)

targets = [-79.42j - 129.3,
           683.3j - 742.9]
test_eig(A, B, targets)
