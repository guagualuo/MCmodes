import dataclasses
from dataclasses import dataclass
from typing import Union, Dict

from operators.equations import *
from operators.worland_transform import WorlandTransform
from utils import *


@dataclass
class _BaseModel(ABC):
    """
    Base model for single m models

    Parameters
    -----
    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    n_grid: the number of radial grids.

    """
    nr: int
    maxnl: int
    m: int
    n_grid: int = field(default=None)

    def __post_init__(self):
        self._check_params()
        self.res = self.nr, self.maxnl, self.m
        if self.n_grid is None:
            self.n_grid = self.nr + self.maxnl // 2 + 11

    @abstractmethod
    def setup_operator(self, *args, **kwargs):
        """
        Setup operators of the problem
        """
        pass

    @abstractmethod
    def setup_eigen_problem(self, operators, **kwargs):
        """
        Setup the matrices for eigenvalue problem
        """
        pass

    def _check_params(self):
        if self.nr <= 0:
            raise RuntimeWarning("nr must be positive")
        if self.maxnl <= 1:
            raise RuntimeWarning(f"maxnl={self.maxnl} contains no valid l")
        if self.m < 0:
            raise RuntimeWarning(f"m must be non-negative")
        if isinstance(self.n_grid, int) and self.n_grid <= 0:
            raise RuntimeWarning(f"n_grid must be positive")


@dataclass
class FreeDecay:
    """
    Class for free decay magnetic eigenmodes

    Parameters
    -----
    component: either 'tor' or 'pol'

    nr: number of radial modes, starting from 0

    l: spherical harmonic degree

    """
    component: str
    nr: int
    l: int

    def __post_init__(self):
        self._check_params()
        bcs = {'tor': {0: 10}, 'pol': {0: 13}}
        self.bc = bcs[self.component]

    def setup_eigen_problem(self):
        import quicc.geometry.spherical.sphere_radius_worland as rad
        A = rad.i2lapl(self.nr, self.l, self.bc, coeff=self.l*(self.l+1))
        B = rad.i2(self.nr, self.l, {0: 0}, coeff=self.l*(self.l+1))
        return A, B

    def _check_params(self):
        if self.component.lower() not in ['tor', 'pol']:
            raise RuntimeWarning(f"component {self.component} is neither 'tor' or 'pol'")
        if self.nr <= 0:
            raise RuntimeWarning("nr must be positive")
        if self.l <= 0:
            raise RuntimeWarning("l must be positive")


@dataclass
class KinematicDynamo(_BaseModel):
    """
    Class for the kinematic dynamo problem.

    Parameters
    -----
    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    n_grid: the number of radial grids.

    induction_eq_params: setup for the induction equation.
    """
    induction_eq_params: dict = field(default_factory=dict)

    def __post_init__(self):
        super(KinematicDynamo, self).__post_init__()
        self.transform = WorlandTransform(self.nr, self.maxnl, self.m, self.n_grid, require_curl=False)
        self.induction_eq = InductionEquation(self.nr, self.maxnl, self.m, **self.induction_eq_params)

    def setup_operator(self, flow_modes: List[SphericalHarmonicMode], setup_eigen=False, **kwargs):
        induction_mat = self.induction_eq.induction(self.transform, flow_modes, imposed_flow=True, quasi_inverse=True)
        operators = {'mass': self.induction_eq.mass,
                     'induction': induction_mat,
                     'diffusion': self.induction_eq.diffusion}
        if setup_eigen:
            return self.setup_eigen_problem(operators, **kwargs)
        else:
            return operators

    def setup_eigen_problem(self, operators, **kwargs):
        Rm = kwargs.get('Rm')
        return Rm * operators['induction'] + operators['diffusion'], operators['mass']


@dataclass
class InertialModes(_BaseModel):
    """
    Class for ideal inertial modes.

    Parameters
    -----
    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    n_grid: the number of radial grids.

    inviscid: If True, the viscous diffusion term is set to zero

    bc_type: boundary condition for viscous u. Ignored when `inviscid` is True.

    """
    inviscid: bool = True
    bc_type: str = None

    def __post_init__(self):
        super(InertialModes, self).__post_init__()
        self._check_params()
        self.momentum_eq = MomentumEquation(self.nr, self.maxnl, self.m, self.inviscid, self.bc_type)

    def setup_operator(self, setup_eigen=False, **kwargs):
        nr, maxnl, m = self.res
        dim = nr * (maxnl - m)
        operators = {}
        operators['mass'] = self.momentum_eq.mass
        operators['coriolis'] = self.momentum_eq.coriolis
        if self.momentum_eq.inviscid:
            operators['diffusion'] = scsp.csc_matrix((2 * dim, 2 * dim))
        else:
            operators['diffusion'] = self.momentum_eq.diffusion
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

    def _check_params(self):
        if not self.inviscid and self.bc_type is None:
            raise RuntimeWarning("no boundary condition of u is specified when viscous dissipation is present")


@dataclass
class MagnetoCoriolis(_BaseModel):
    """
    Class for Magneto-Coriolis modes

    Parameters
    -----
    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    n_grid: the number of radial grids.

    inviscid: If True, the viscous diffusion term is set to zero

    bc_type: boundary condition for viscous u. Ignored when `inviscid` is True.

    induction_eq_params: setup for the induction equation.

    """
    inviscid: bool = True
    bc_type: str = None
    induction_eq_params: dict = field(default_factory=dict)

    def __post_init__(self):
        super(MagnetoCoriolis, self).__post_init__()
        self._check_params()

        nr, maxnl, m = self.res
        self.transform = WorlandTransform(nr, maxnl, m, self.n_grid, require_curl=True)
        self.induction_eq = InductionEquation(*self.res, **self.induction_eq_params)
        self.momentum_eq = MomentumEquation(*self.res, inviscid=self.inviscid, bc_type=self.bc_type)
        if self.induction_eq.galerkin:
            self.dim = {'u': 2 * nr * (maxnl - m), 'b': 2 * (nr - 1) * (maxnl - m)}
        else:
            self.dim = {'u': 2 * nr * (maxnl - m), 'b': 2 * nr * (maxnl - m)}

    def _check_params(self):
        if not self.inviscid and self.bc_type is None:
            raise RuntimeWarning("no boundary condition of u is specified when viscous dissipation is present")

    def setup_operator(self, field_modes: List[SphericalHarmonicMode],
                       flow_modes: Union[None, List[SphericalHarmonicMode]] = None, setup_eigen=False,
                       *args, **kwargs):
        if flow_modes is None:
            flow_modes = []
        nr, maxnl, m = self.res
        dim = nr*(maxnl - m)
        operators = {}
        operators['lorentz'] = self.momentum_eq.lorentz(self.transform, field_modes, quasi_inverse=True)
        if self.induction_eq.galerkin:
            operators['lorentz'] = operators['lorentz'] @ self.induction_eq.stencil
        operators['inductionB'] = self.induction_eq.induction(self.transform, field_modes, imposed_flow=False,
                                                              quasi_inverse=True)
        operators['advection'] = self.momentum_eq.advection(self.transform, flow_modes, quasi_inverse=True)
        operators['inductionU'] = self.induction_eq.induction(self.transform, flow_modes, imposed_flow=True,
                                                              quasi_inverse=True)
        operators['magnetic_diffusion'] = self.induction_eq.diffusion
        operators['coriolis'] = self.momentum_eq.coriolis
        if self.inviscid:
            operators['viscous_diffusion'] = scsp.csr_matrix((2*dim, 2*dim))
        else:
            operators['viscous_diffusion'] = self.momentum_eq.diffusion
        operators['induction_mass'] = self.induction_eq.mass
        operators['momentum_mass'] = self.momentum_eq.mass
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
            operators['inv_coriolis'] = clu
            u = clu.solve(operators['lorentz'].toarray())
            operators['ms_induction'] = operators['inductionB'] @ u

            B = operators['induction_mass']
            A = elsasser*operators['ms_induction'] + operators['magnetic_diffusion']
            # separate parity
            if kwargs.get('parity', False):
                return self.separate_parity(A, B, b_parity='DP', u_parity=None), \
                       self.separate_parity(A, B, b_parity='QP', u_parity=None)
            else:
                return A, B
        else:
            B = scsp.block_diag((Eeta*operators['momentum_mass'], operators['induction_mass']))
            A = scsp.bmat([[-Eeta*U*operators['advection']-operators['coriolis'] + ekman*operators['viscous_diffusion'],
                            elsasser**0.5*operators['lorentz']],
                           [elsasser**0.5*operators['inductionB'],
                            U*operators['inductionU'] + operators['magnetic_diffusion']]
                           ])
            # separate parity
            if kwargs.get('parity', False):
                return self.separate_parity(A, B, b_parity='DP', u_parity=self.u_parity('DP', kwargs.get('u_parity'))),\
                       self.separate_parity(A, B, b_parity='QP', u_parity=self.u_parity('QP', kwargs.get('u_parity')))
            else:
                return A, B

    def separate_parity(self, A, B, b_parity, u_parity):
        nr, maxnl, m = self.res
        dimu = 2 * nr * (maxnl-m)
        A = scsp.lil_matrix(A)
        B = scsp.lil_matrix(B)

        if u_parity is None:
            row_idx = vector_parity_idx(nr, maxnl, m, b_parity, ngalerkin=int(self.induction_eq.galerkin))
            col_idx = vector_parity_idx(nr, maxnl, m, b_parity, ngalerkin=int(self.induction_eq.galerkin))
        else:
            row_idx = np.append(vector_parity_idx(nr, maxnl, m, u_parity),
                                dimu + vector_parity_idx(nr, maxnl, m, b_parity, ngalerkin=int(self.induction_eq.galerkin)))
            col_idx = np.append(vector_parity_idx(nr, maxnl, m, u_parity),
                                dimu + vector_parity_idx(nr, maxnl, m, b_parity, ngalerkin=int(self.induction_eq.galerkin)))
        return scsp.csr_matrix(A[row_idx[:, None], col_idx]), scsp.coo_matrix(B[row_idx[:, None], col_idx])

    def u_parity(self, b_parity, relation):
        if relation == 'same':
            return b_parity
        else:
            return 'DP' if b_parity == 'QP' else 'QP'


class IdealMagnetoCoriolis(MagnetoCoriolis):
    """
    Ideal Magneto-Coriolis modes, using the Alfven time scale formulation with Le number
    """

    def __init__(self, nr, maxnl, m, n_grid=None, **kwargs):
        # default using a galerkin basis, but can be used with no boundary condition (only for the aim of Malkus case)
        induction_eq_params = {'galerkin': kwargs.get('galerkin', True),
                               'ideal': True,
                               'boundary_condition': kwargs.get('boundary_condition', True)}
        super(IdealMagnetoCoriolis, self).__init__(nr, maxnl, m, n_grid,
                                                   inviscid=True,
                                                   induction_eq_params=induction_eq_params)

    def setup_eigen_problem(self, operators, **kwargs):
        Le = kwargs.get('lehnert')
        U = kwargs.get('U', 0)
        B = scsp.block_diag((operators['momentum_mass'], operators['induction_mass']))
        A = scsp.bmat(
            [[-U * operators['advection'] - 2/Le * operators['coriolis'], operators['lorentz']],
             [operators['inductionB'], U * operators['inductionU']]
             ])
        # separate parity
        if kwargs.get('parity', False):
            return self.separate_parity(A, B, b_parity='DP', u_parity=self.u_parity('DP', kwargs.get('u_parity'))), \
                   self.separate_parity(A, B, b_parity='QP', u_parity=self.u_parity('QP', kwargs.get('u_parity')))
        else:
            return A, B


class TorsionalOscillation(MagnetoCoriolis):
    """
    Torsional oscillation with magnetic diffusion and possibly with viscous diffusion
    Non-dimensional parameters are Le, Lu and Pm
    """
    def __init__(self, nr, maxnl, inviscid=True, n_grid=None, **kwargs):
        induction_eq_params = {'galerkin': kwargs.get('galerkin', False)}
        super(TorsionalOscillation, self).__init__(nr, maxnl, 0, n_grid,
                                                   inviscid=inviscid,
                                                   induction_eq_params=induction_eq_params)

    def setup_eigen_problem(self, operators, **kwargs):
        Le = kwargs.get('lehnert')
        Lu = kwargs.get('lundquist')
        Pm = kwargs.get('pm', 0)
        U = kwargs.get('U', 0)
        B = scsp.block_diag((operators['momentum_mass'], operators['induction_mass']))
        A = scsp.bmat(
            [[-U * operators['advection'] - 2 / Le * operators['coriolis'] + Pm/Lu*operators['viscous_diffusion'],
              operators['lorentz']],
             [operators['inductionB'], U * operators['inductionU'] + 1/Lu*operators['magnetic_diffusion']]
             ])
        # separate parity
        if kwargs.get('parity', False):
            return self.separate_parity(A, B, b_parity='DP', u_parity=self.u_parity('DP', kwargs.get('u_parity'))), \
                   self.separate_parity(A, B, b_parity='QP', u_parity=self.u_parity('QP', kwargs.get('u_parity')))
        else:
            return A, B


class IdealTorsionalOscillation(IdealMagnetoCoriolis):
    """
    Torsional oscillation with no diffusion, galerkin basis. Likely to be an ill-posed system.
    """
    def __init__(self, nr, maxnl, n_grid=None, **kwargs):
        super(IdealTorsionalOscillation, self).__init__(nr, maxnl, 0, n_grid, **kwargs)

    def setup_eigen_problem(self, operators, **kwargs):
        Le = kwargs.get('lehnert')
        U = kwargs.get('U', 0)
        B = scsp.block_diag((operators['momentum_mass'], operators['induction_mass']))
        A = scsp.bmat(
            [[-U * operators['advection'] - 2 / Le * operators['coriolis'],
              operators['lorentz']],
             [operators['inductionB'], U * operators['inductionU']]
             ])
        # separate parity
        if kwargs.get('parity', False):
            return self.separate_parity(A, B, b_parity='DP', u_parity=self.u_parity('DP', kwargs.get('u_parity'))), \
                   self.separate_parity(A, B, b_parity='QP', u_parity=self.u_parity('QP', kwargs.get('u_parity')))
        else:
            return A, B
