from abc import ABC, abstractmethod
from operators.physical_operators import *
from operators.worland_operator import WorlandTransform
from typing import Union, List


class BaseModel(ABC):
    def __init__(self, nr, maxnl, m, n_grid, *args, **kwargs):
        self.res = (nr, maxnl, m)
        self.n_grid = n_grid

    @abstractmethod
    def setup_operator(self, *args, **kwargs):
        pass

    @abstractmethod
    def setup_eigen_problem(self, operators, **kwargs):
        pass


class KinematicDynamo(BaseModel):
    def __init__(self, nr, maxnl, m, n_grid):
        super(KinematicDynamo, self).__init__(nr, maxnl, m, n_grid)
        nr, maxnl, m = self.res
        self.transform = WorlandTransform(nr, maxnl, m, n_grid, require_curl=False)

    def setup_operator(self, flow_modes: List[SphericalHarmonicMode], setup_eigen=False, **kwargs):
        nr, maxnl, m = self.res
        mass_mat = induction_mass(nr, maxnl, m)
        induction_mat = induction(self.transform, flow_modes, imposed_flow=True)
        quasi_inverse = induction_quasi_inverse(nr, maxnl, m)
        induction_mat = quasi_inverse @ induction_mat
        diffusion_mat = induction_diffusion(nr, maxnl, m, bc=True)
        operators = {'mass': mass_mat, 'induction': induction_mat, 'diffusion': diffusion_mat}
        if setup_eigen:
            return self.setup_eigen_problem(operators, **kwargs)
        else:
            return operators

    def setup_eigen_problem(self, operators, **kwargs):
        Rm = kwargs.get('Rm')
        return Rm * operators['induction'] + operators['diffusion'], operators['mass']


class MagnetoCoriolis(BaseModel):
    def __init__(self, nr, maxnl, m, inviscid=True, bc=None):
        super(MagnetoCoriolis, self).__init__(nr, maxnl, m, n_grid)
        nr, maxnl, m = self.res
        self.transform = WorlandTransform(nr, maxnl, m, n_grid, require_curl=True)
        self.inviscid = inviscid
        if not inviscid:
            assert bc is not None
            self.bc = bc

    def setup_operator(self, field_modes: List[SphericalHarmonicMode],
                       flow_modes: Union[None, List[SphericalHarmonicMode]] = None, setup_eigen=False,
                       *args, **kwargs):
        if flow_modes is None:
            flow_modes = []
        nr, maxnl, m = self.res
        operators = {}
        induction_qi = induction_quasi_inverse(nr, maxnl, m)
        momentum_qi = momentum_quasi_inverse(nr, maxnl, m, inviscid=self.inviscid)
        operators['lorentz'] = momentum_qi @ lorentz(self.transform, field_modes)
        operators['inductionB'] = induction_qi @ induction(self.transform, field_modes, imposed_flow=False)
        if len(flow_modes) > 0:
            operators['advection'] = momentum_qi @ lorentz(self.transform, flow_modes)
            operators['inductionU'] = induction_qi @ induction(self.transform, flow_modes, imposed_flow=True)
        operators['magnetic_diffusion'] = induction_diffusion(nr, maxnl, m, bc=True)
        operators['coriolis'] = coriolis(nr, maxnl, m, inviscid=self.inviscid, bc=self.inviscid)
        if not self.inviscid:
            operators['viscous_diffusion'] = viscous_diffusion(nr, maxnl, m, bc=True, bc_type=self.bc)
        operators['induction_mass'] = induction_mass(nr, maxnl, m)
        operators['momentum_mass'] = momentum_mass(nr, maxnl, m, self.inviscid)
        if setup_eigen:
            return self.setup_eigen_problem(operators, **kwargs)
        else:
            return operators

    def setup_eigen_problem(self, operators, **kwargs):
        pass

