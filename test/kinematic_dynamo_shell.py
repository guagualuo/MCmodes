from utils import single_eig

from models import FreeDecay, KinematicDynamo
from operators.polynomials import SphericalHarmonicMode

nr = 41
maxnl = 41
ri = 0.0001

t10 = SphericalHarmonicMode("tor", 1, 0, "Sqrt[4 pi / 3] Sin[pi r]")
s10 = SphericalHarmonicMode("pol", 1, 0, "17/100*Sqrt[4 pi / 3] Sin[pi r]")
s20 = SphericalHarmonicMode("pol", 2, 0, "13/100*Sqrt[4 pi / 5] r Sin[pi r]")
LJt10 = SphericalHarmonicMode("tor", 1, 0, "8.107929179422066 * r(1 - r^2)")
LJs20 = SphericalHarmonicMode("pol", 2, 0, "1.193271237996972 * r^2 (1 - r^2)^2")

# case 1
m = 1
model = KinematicDynamo(nr, maxnl, m, ri=ri)
A, B = model.setup_operator(flow_modes=[t10, s10], setup_eigen=True, Rm=160)
w, _ = single_eig(A, B, target=0.313-34.84j)
print(w[0])

# case 2
m = 0
model = KinematicDynamo(nr, maxnl, m, ri=ri)
A, B = model.setup_operator(flow_modes=[LJt10, LJs20], setup_eigen=True, Rm=100)
w, _ = single_eig(A, B, target=-6.9)
print(w[0])

