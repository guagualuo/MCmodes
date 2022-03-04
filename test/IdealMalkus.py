""" This module contains benchmarks for the ideal MC modes from Malkus configuration """
import matplotlib.pyplot as plt
import numpy as np

from models import MagnetoCoriolis
from operators.polynomials import SphericalHarmonicMode
from utils import *


def fast(wi, elsasser, magnetic_ekman, m):
    return 0.25/magnetic_ekman*wi*(1+np.sqrt(1+16*magnetic_ekman*elsasser*m*(m-wi)/wi**2+0.0j))


def slow(wi, elsasser, magnetic_ekman, m):
    return 0.25/magnetic_ekman*wi*(1-np.sqrt(1+16*magnetic_ekman*elsasser*m*(m-wi)/wi**2+0.0j))


nr, maxnl, m = 21, 21, 1
field_modes = [SphericalHarmonicMode("tor", 1, 0, "2 Sqrt[pi/3] r")]
model = MagnetoCoriolis(nr, maxnl, m, inviscid=True,
                        induction_eq_params={'galerkin': False, 'ideal': True, 'boundary_condition': False})
operators = model.setup_operator(field_modes=field_modes, setup_eigen=False)

Eeta = 1e-4
wi = 1.3060787169132158
ntest = 21
fast_eig, slow_eig = np.zeros(ntest, dtype=np.complex128), np.zeros(ntest, dtype=np.complex128)
gammas = np.linspace(0.1, 5000, ntest)
for i, gamma in enumerate(gammas):
    A, B = model.setup_eigen_problem(operators=operators, magnetic_ekman=Eeta, elsasser=gamma)

    target = fast(np.round(wi, 4), gamma, Eeta, m)*1.0j
    w, _ = single_eig(A, B, target, nev=1)
    fast_eig[i] = w[0]

    target = slow(np.round(wi, 4), gamma, Eeta, m)*1.0j
    w, _ = single_eig(A, B, target, nev=1)
    slow_eig[i] = w[0]

gammas_ = np.linspace(0, 5000, 5001)

w_fast = fast(wi, gammas_, Eeta, m)
w_slow = slow(wi, gammas_, Eeta, m)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.plot(gammas_, w_fast.real, '-k')
ax1.plot(gammas_, w_slow.real, '-k')
ax2.plot(gammas_, -w_fast.imag, ':k')
ax2.plot(gammas_, -w_slow.imag, ':k')
ax1.set_xlabel(r'$\Lambda$', fontsize=14)
ax1.set_ylabel(r'$Im(\omega)$', fontsize=14)
ax2.set_ylabel(r'$Re(\omega)$', fontsize=14)
ax1.scatter(gammas, fast_eig.imag, marker='o', color='k')
ax1.scatter(gammas, slow_eig.imag, marker='o', color='k')
ax2.scatter(gammas, fast_eig.real, marker='s', color='k')
ax2.scatter(gammas, slow_eig.real, marker='s', color='k')
plt.show()
# fig.savefig('../../data/IdealMalkus/benchmark.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
