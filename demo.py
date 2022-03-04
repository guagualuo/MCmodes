import numpy as np
import matplotlib.pyplot as plt

from models import MagnetoCoriolis
from operators.polynomials import SphericalHarmonicMode
from operators.worland_transform import WorlandTransform
from operators.associated_legendre_transform import AssociatedLegendreTransformSingleM
from fields import VectorFieldSingleM
from utils import single_eig

nr, maxnl, m = 41, 41, 1
field_modes = [SphericalHarmonicMode("tor", 1, 0, "3 Sqrt[pi] r(1-r^2)")]
model = MagnetoCoriolis(nr, maxnl, m)
A, B = model.setup_operator(field_modes=field_modes, setup_eigen=True, magnetic_ekman=1e-9, elsasser=1)

# compute eigenmode
w, v = single_eig(A, B, target=0.5779j - 49.45, nev=1)
# get spectrum object and normalise to b with energy 1
dim = v.shape[0] // 2
usp = VectorFieldSingleM(nr, maxnl, m, v[:dim, 0])
bsp = VectorFieldSingleM(nr, maxnl, m, v[dim:, 0])
norm = np.sqrt(bsp.energy)
usp.normalise(norm)
bsp.normalise(norm)

# compute physical field on a meridional slice
rg = np.linspace(0, 1, 201)
tg = np.linspace(0, np.pi/2, 201)
worland_transform = WorlandTransform(nr, maxnl, m, r_grid=rg)
legendre_transform = AssociatedLegendreTransformSingleM(maxnl, m, tg)

uphy = usp.physical_field(worland_transform, legendre_transform)
bphy = bsp.physical_field(worland_transform, legendre_transform)

# visualise field components
fig, axes = plt.subplots(figsize=(12, 8), ncols=3, nrows=2, sharey=True, sharex=True)
uphy.visualise(name='u', coord='cylindrical', ax=axes[0])
bphy.visualise(name='b', coord='spherical', ax=axes[1])
fig.tight_layout()

plt.show()
