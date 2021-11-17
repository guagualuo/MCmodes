from abc import ABC, abstractmethod
from operators.physical_operators import *
from operators.worland_operator import WorlandTransform
from typing import Union, List


class BaseModel(ABC):
    def __init__(self, res, n_grid, *args, **kwargs):
        self.res = res
        self.n_grid = n_grid

    @abstractmethod
    def setup_equation(self, *args, **kwargs):
        pass


class KinematicDynamo(BaseModel):
    def __init__(self, res, n_grid):
        super(KinematicDynamo, self).__init__(res, n_grid)
        nr, maxnl, m = self.res
        self.transform = WorlandTransform(nr, maxnl, m, n_grid, require_curl=False)

    def setup_equation(self, flow_modes: List[SphericalHarmonicMode], *args, **kwargs):
        nr, maxnl, m = self.res
        mass_mat = induction_mass(nr, maxnl, m)
        induction_mat = -induction(self.transform, flow_modes)
        quasi_inverse = induction_quasi_inverse(nr, maxnl, m)
        induction_mat = quasi_inverse @ induction_mat
        diffusion_mat = induction_diffusion(nr, maxnl, m, bc=True)
        return mass_mat, induction_mat, diffusion_mat


class MagnetoCoriolis(BaseModel):
    def __init__(self, res):
        super(MagnetoCoriolis, self).__init__(res, n_grid)
        nr, maxnl, m = self.res
        self.transform = WorlandTransform(nr, maxnl, m, n_grid, require_curl=True)

    def setup_equation(self, field_modes: List[SphericalHarmonicMode],
                       flow_modes=Union[None, List[SphericalHarmonicMode]], *args, **kwargs):
        nr, maxnl, m = self.res
        ub = lorentz(transform, field_modes)
        bb = induction(transform, field_modes)
        #TODO: return in form of mass, mat propto Elsasser, mat propto U, rest operators
        pass
