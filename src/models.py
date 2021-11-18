from abc import ABC, abstractmethod
import scipy.sparse.linalg as spla
from typing import Union, List

from operators.equations import *
from operators.worland_transform import WorlandTransform


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
        self.transform = WorlandTransform(nr, maxnl, m, n_grid, require_curl=False)
        self.induction_eq = InductionEquation(*self.res)

    def setup_operator(self, flow_modes: List[SphericalHarmonicMode], setup_eigen=False, **kwargs):
        mass_mat = self.induction_eq.mass()
        induction_mat = self.induction_eq.induction(self.transform, flow_modes, imposed_flow=True, quasi_inverse=True)
        diffusion_mat = self.induction_eq.diffusion(bc=True)
        operators = {'mass': mass_mat, 'induction': induction_mat, 'diffusion': diffusion_mat}
        if setup_eigen:
            return self.setup_eigen_problem(operators, **kwargs)
        else:
            return operators

    def setup_eigen_problem(self, operators, **kwargs):
        Rm = kwargs.get('Rm')
        return Rm * operators['induction'] + operators['diffusion'], operators['mass']


class InertialModes(BaseModel):
    def __init__(self, nr, maxnl, m, inviscid: bool, bc_type: str = None):
        super(InertialModes, self).__init__(nr, maxnl, m, None)
        self.momentum_eq = MomentumEquation(*self.res, inviscid, bc_type)

    def setup_operator(self, setup_eigen=False, **kwargs):
        nr, maxnl, m = self.res
        dim = nr * (maxnl - m)
        operators = {}
        operators['mass'] = self.momentum_eq.mass()
        operators['coriolis'] = self.momentum_eq.coriolis(bc=self.momentum_eq.inviscid)
        if not self.momentum_eq.inviscid:
            operators['diffusion'] = scsp.csc_matrix((2 * dim, 2 * dim))
        if setup_eigen:
            return self.setup_eigen_problem(operators, **kwargs)
        else:
            return operators

    def setup_eigen_problem(self, operators, **kwargs):
        if not self.momentum_eq.inviscid:
            ekman = kwargs.get('ekman')
            return -2*operators['coriolis'] + ekman*operators['diffusion'], operators['mass']
        else:
            return -2*operators['coriolis'], operators['mass']


class MagnetoCoriolis(BaseModel):
    def __init__(self, nr, maxnl, m, n_grid, inviscid=True, bc=None):
        super(MagnetoCoriolis, self).__init__(nr, maxnl, m, n_grid)
        self.transform = WorlandTransform(nr, maxnl, m, n_grid, require_curl=True)
        self.inviscid = inviscid
        if not inviscid:
            assert bc is not None
            self.bc = bc
        self.induction_eq = InductionEquation(*self.res)
        self.momentum_eq = MomentumEquation(*self.res, inviscid=inviscid, bc_type=bc)

    def setup_operator(self, field_modes: List[SphericalHarmonicMode],
                       flow_modes: Union[None, List[SphericalHarmonicMode]] = None, setup_eigen=False,
                       *args, **kwargs):
        if flow_modes is None:
            flow_modes = []
        nr, maxnl, m = self.res
        dim = nr*(maxnl - m)
        operators = {}
        operators['lorentz'] = self.momentum_eq.lorentz(self.transform, field_modes, quasi_inverse=True)
        operators['inductionB'] = self.induction_eq.induction(self.transform, field_modes, imposed_flow=False, quasi_inverse=True)
        operators['advection'] = self.momentum_eq.advection(self.transform, flow_modes, quasi_inverse=True)
        operators['inductionU'] = self.induction_eq.induction(self.transform, flow_modes, imposed_flow=True, quasi_inverse=True)
        operators['magnetic_diffusion'] = self.induction_eq.diffusion(bc=True)
        operators['coriolis'] = self.momentum_eq.coriolis(bc=self.inviscid)
        if self.inviscid:
            operators['viscous_diffusion'] = scsp.csc_matrix((2*dim, 2*dim))
        else:
            operators['viscous_diffusion'] = self.momentum_eq.diffusion(bc=True)
        operators['induction_mass'] = self.induction_eq.mass()
        operators['momentum_mass'] = self.momentum_eq.mass()
        if setup_eigen:
            return self.setup_eigen_problem(operators, **kwargs)
        else:
            return operators

    def setup_eigen_problem(self, operators, **kwargs):
        Eeta = kwargs.get('magnetic_ekman')
        elsasser = kwargs.get('elsasser')
        U = kwargs.get('U', 0)
        ekman = kwargs.get('ekman', 0)

        if Eeta == 0:
            clu = spla.splu(operators['coriolis'])
            u = clu.solve(operators['lorentz'].toarray())
            operators['ms_induction'] = operators['inductionB'] @ u

            B = operators['induction_mass']
            A = elsasser*operators['ms_induction'] + operators['magnetic_diffusion']
        else:
            B = scsp.block_diag((Eeta*operators['momentum_mass'], operators['induction_mass']))
            A = scsp.bmat([[-Eeta*U*operators['advection']-operators['coriolis'] + ekman*operators['viscous_diffusion'],
                            elsasser**0.5*operators['lorentz']],
                           [elsasser**0.5*operators['inductionB'],
                            U*operators['inductionU'] + operators['magnetic_diffusion']]
                           ])
        return A, B
