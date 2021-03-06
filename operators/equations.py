from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import List
import matplotlib.pyplot as plt
import scipy.sparse as scsp

from operators import WorlandTransform
from operators.polynomials import SphericalHarmonicMode
from utils import Timer
import quicc.geometry.spherical.sphere_worland as geo
import operators.quicc_supplements.sphere_worland as supp_geo
import quicc.geometry.spherical.sphere_radius_boundary_worland as wbc
from quicc.geometry.spherical.sphere_boundary_worland import no_bc


@dataclass
class _BaseEquation(ABC):
    """
    Base class for equation object at single m

    Parameters
    -----
    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number
    """
    nr: int
    maxnl: int
    m: int

    def __post_init__(self):
        self.res = self.nr, self.maxnl, self.m

    def _init_operators(self):
        """ Initialise operators that requires no physical transformations. """
        for attr in dir(self):
            if attr.startswith('_create'):
                getattr(self, attr)()


@dataclass
class InductionEquation(_BaseEquation):
    """
    Class for the induction equation

    Parameters
    -----

    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    galerkin: Whether to use a galerkin basis for the magnetic field.
        When set to be True, boundary_condition must be True.

    ideal: When set to be True, the diffusion operator is set to zero.
        When set to False, boundary_condition must be True.

    boundary_condition: Whether the magnetic field has a boundary condition.
        If set to False, galerkin must be False and ideal must True. This is used for the canonical Malkus case.

    """
    galerkin: bool = False
    ideal: bool = False
    boundary_condition: bool = True

    def __post_init__(self):
        super(InductionEquation, self).__post_init__()
        if self.boundary_condition:
            self.bc = {'tor': {0: 10}, 'pol': {0: 13}}
            if self.galerkin:
                self.bc = {'tor': {0: -10, 'rt': 1}, 'pol': {0: -13, 'rt': 1}}
        else:
            # specifically for the Malkus case of MC modes
            self.bc = None
            if not self.ideal:
                raise RuntimeError("trying to set no boundary condition for a non-ideal case")
            if self.galerkin:
                raise RuntimeError("Trying to set no boundary condition for a galerkin basis")
        # initialise simple linear operators
        self._init_operators()

    def _init_operators(self):
        """
        Initialise simple linear operators
        """
        if self.galerkin:
            self._construct_stencil()
        super(InductionEquation, self)._init_operators()

    def induction(self,
                  transform: WorlandTransform,
                  beta_modes: List[SphericalHarmonicMode],
                  imposed_flow: bool,
                  quasi_inverse: bool):
        """
        Induction term curl (u x B_0), in which B_0 is the background field.
                [ r.curl2(t_a x B_0), r.curl2(s_a x B_0)
                  r.curl1(t_a x B0), r.curl1(s_a x B_0)]

        Parameters
        -----
        transform: Worland transform obeject that implements all 8 possible combinations

        beta_modes: Spherical harmonic modes of B_0 or u

        imposed_flow: whether impose u or B in the induction term curl(u x B).
            If set True, it can be used for the kinematic dynamo problem, for example.

        quasi_inverse: whether to apply quasi-inverse operator to the induction.

        """
        nr, maxnl, m = self.res
        if len(beta_modes) > 0:
            tt = sum([transform.curl2tt(mode) + transform.curl2ts(mode) for mode in beta_modes])
            ts = sum([transform.curl2st(mode) + transform.curl2ss(mode) for mode in beta_modes])
            st = sum([transform.curl1tt(mode) + transform.curl1ts(mode) for mode in beta_modes])
            ss = sum([transform.curl1st(mode) + transform.curl1ss(mode) for mode in beta_modes])
            sign = -1.0 if imposed_flow else 1.0
            op = sign * scsp.bmat([[tt, ts], [st, ss]], format='csc')
            if quasi_inverse:
                op = self.quasi_inverse @ op
            # apply stencil operator for galerkin basis if the flow is imposed,
            # no need to apply stencil if B is imposed in the MC problem
            if self.galerkin and imposed_flow:
                op = op @ self.stencil
            return op
        else:
            if self.galerkin:
                return scsp.csc_matrix((2*(nr-1)*(maxnl-m), 2*(nr-1)*(maxnl-m)))
            else:
                return scsp.csc_matrix((2*nr*(maxnl-m), 2*nr*(maxnl-m)))

    def _construct_stencil(self):
        """
        Stencil operator for galerkin basis
        """
        nr, maxnl, m = self.res
        assert self.galerkin
        ops = [wbc.stencil(nr, l, self.bc['tor']) for l in range(m, maxnl)]
        ops += [wbc.stencil(nr, l, self.bc['pol']) for l in range(m, maxnl)]
        self._stencil = scsp.block_diag(ops, format='csc')

    def _create_quasi_inverse(self):
        """
        Quasi-inverse operator
        """
        nr, maxnl, m = self.res
        if self.bc is not None:
            bc = no_bc()
            # remove top tau line if using galerkin basis
            if self.galerkin: bc['rt'] = 1
            self._quasi_inverse = scsp.block_diag((geo.i2(nr, maxnl, m, bc, l_zero_fix='zero'),
                                                   geo.i2(nr, maxnl, m, bc, l_zero_fix='zero')))
        else:
            self._quasi_inverse = scsp.block_diag((supp_geo.i2_nobc(nr, maxnl, m, no_bc(), l_zero_fix='zero'),
                                                   supp_geo.i2_nobc(nr, maxnl, m, no_bc(), l_zero_fix='zero')))

    def _create_mass(self):
        """
        Mass operator
        """
        nr, maxnl, m = self.res
        if self.bc is not None:
            bc = no_bc()
            if self.galerkin: bc['rt'] = 1
            i2 = scsp.block_diag((geo.i2(nr, maxnl, m, bc, with_sh_coeff='laplh', l_zero_fix='zero'),
                                  geo.i2(nr, maxnl, m, bc, with_sh_coeff='laplh', l_zero_fix='zero')))
            self._mass = i2 @ self.stencil if self.galerkin else i2

        else:
            self._mass = scsp.block_diag((supp_geo.i2_nobc(nr, maxnl, m, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                                          supp_geo.i2_nobc(nr, maxnl, m, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero')))

    def _create_diffusion(self):
        """
        Build the dissipation matrix for the magnetic field, insulating boundary condition
        """
        nr, maxnl, m = self.res
        if self.bc is not None and not self.ideal:
            self._diffusion = scsp.block_diag((geo.i2lapl(nr, maxnl, m, bc=self.bc['tor'], with_sh_coeff='laplh', l_zero_fix='set'),
                                               geo.i2lapl(nr, maxnl, m, bc=self.bc['pol'], with_sh_coeff='laplh', l_zero_fix='set')))
        else:
            self._diffusion = scsp.coo_matrix((2*nr*(maxnl-m), 2*nr*(maxnl-m)))

    @property
    def stencil(self):
        return self._stencil

    @property
    def mass(self):
        return self._mass

    @property
    def quasi_inverse(self):
        return self._quasi_inverse

    @property
    def diffusion(self):
        return self._diffusion


@dataclass
class MomentumEquation(_BaseEquation):
    """
    Class for the linearised momentum equation at a single m

    Parameters
    -----

    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    inviscid: If set True, viscous dissipation is set 0

    bc_type: boundary condtion for u, 'no-slip' or 'stress-free'. Ignored when inviscid is True.

    """
    inviscid: bool = True
    bc_type: str = None

    def __post_init__(self):
        super(MomentumEquation, self).__post_init__()
        if not self.inviscid:
            assert self.bc_type.lower() in ['no-slip', 'stress-free']
        self._set_bc()
        self._init_operators()

    def _set_bc(self):
        if self.inviscid:
            self.bc = {'tor': no_bc(), 'pol': {0: 10}}
        else:
            if self.bc_type == 'no-slip':
                self.bc = {"tor": {0: 10}, "pol": {0: 20}}
            if self.bc_type == 'stress-free':
                self.bc = {"tor": {0: 12}, "pol": {0: 21}}

    def lorentz1(self,
                 transform: WorlandTransform,
                 modes: List[SphericalHarmonicMode]):
        """
        Lorentz term (curl B_0) x b, in which B_0 is the background field.
                    [ r.curl1(J0 x b), r.curl1(J0 x b)
                      r.curl2(J0 x b), r.curl2(J0 x b)]

        Parameters
        -----
        transform: Worland transform obeject that implements all 8 possible combinations

        modes: Spherical harmonic modes of B_0

        """
        nr, maxnl, m = self.res
        if len(modes) > 0:
            curl_modes = [mode.curl() for mode in modes]
            tt = sum([transform.curl1tt(mode) + transform.curl1ts(mode) for mode in curl_modes])
            ts = sum([transform.curl1st(mode) + transform.curl1ss(mode) for mode in curl_modes])
            st = sum([transform.curl2tt(mode) + transform.curl2ts(mode) for mode in curl_modes])
            ss = sum([transform.curl2st(mode) + transform.curl2ss(mode) for mode in curl_modes])
            return -scsp.bmat([[tt, ts], [st, ss]], format='csc')
        else:
            return scsp.csc_matrix((2*nr*(maxnl-m), 2*nr*(maxnl-m)))

    def lorentz2(self,
                 transform: WorlandTransform,
                 modes: List[SphericalHarmonicMode]):
        """
        Lorentz term (curl b) x B_0, in which B_0 is the background field.
                    [ r.curl1(j x B_0), r.curl1(j x B_0)
                      r.curl2(j x B_0), r.curl2(j x B_0)]

        Parameters
        -----
        transform: Worland transform obeject that implements all 8 possible combinations

        modes: Spherical harmonic modes of B_0

        """
        nr, maxnl, m = self.res
        if len(modes) > 0:
            tt = sum([transform.curl1curltt(mode) + transform.curl1curlts(mode) for mode in modes])
            ts = sum([transform.curl1curlst(mode) + transform.curl1curlss(mode) for mode in modes])
            st = sum([transform.curl2curltt(mode) + transform.curl2curlts(mode) for mode in modes])
            ss = sum([transform.curl2curlst(mode) + transform.curl2curlss(mode) for mode in modes])
            return scsp.bmat([[tt, ts], [st, ss]], format='csc')
        else:
            return scsp.csc_matrix((2*nr*(maxnl-m), 2*nr*(maxnl-m)))

    def lorentz(self,
                transform: WorlandTransform,
                field_modes: List[SphericalHarmonicMode],
                quasi_inverse=True):
        """
        The total linearised Lorentz term (curl B_0) x b + (curl b) x B_0

        Parameters
        -----
        transform: Worland transform obeject that implements all 8 possible combinations

        field_modes: Spherical harmonic modes of B_0

        quasi_inverse: Whether to apply quasi-inverse operator to the matrix

        """
        op = self.lorentz1(transform, field_modes) + self.lorentz2(transform, field_modes)
        if quasi_inverse:
            op = self.quasi_inverse @ op
        return op.tocsc()

    def advection(self,
                  transform: WorlandTransform,
                  flow_modes: List[SphericalHarmonicMode],
                  quasi_inverse=True):
        """
        The total linearised advection term (curl U_0) x u + (curl u) x U_0

        Parameters
        -----
        transform: Worland transform obeject that implements all 8 possible combinations

        flow_modes: Spherical harmonic modes of U_0

        quasi_inverse: Whether to apply quasi-inverse operator to the matrix

        """
        return self.lorentz(transform, flow_modes, quasi_inverse).tocsc()

    def _create_mass(self):
        nr, maxnl, m = self.res
        if self.inviscid:
            self._mass = scsp.block_diag((supp_geo.i2_nobc(nr, maxnl, m, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                                          geo.i2lapl(nr, maxnl, m, no_bc(), -1.0, with_sh_coeff='laplh', l_zero_fix='zero')))
        else:
            self._mass = scsp.block_diag((geo.i2(nr, maxnl, m, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                                          geo.i4lapl(nr, maxnl, m, no_bc(), -1.0, with_sh_coeff='laplh', l_zero_fix='zero')))

    def _create_quasi_inverse(self):
        nr, maxnl, m = self.res
        if self.inviscid:
            self._quasi_inverse = scsp.block_diag((supp_geo.i2_nobc(nr, maxnl, m, no_bc(), l_zero_fix='zero'),
                                                   geo.i2(nr, maxnl, m, no_bc(), l_zero_fix='zero')))
        else:
            self._quasi_inverse =  scsp.block_diag((geo.i2(nr, maxnl, m, no_bc(), l_zero_fix='zero'),
                                                    geo.i4(nr, maxnl, m, no_bc(), l_zero_fix='zero')))

    def _create_diffusion(self,):
        nr, maxnl, m = self.res
        dim = nr * (maxnl - m)
        if self.inviscid:
            self._diffusion = scsp.csc_matrix((2*dim, 2*dim))
        else:
            self._diffusion = scsp.block_diag((geo.i2lapl(nr, maxnl, m, bc=self.bc["tor"],
                                               with_sh_coeff='laplh', l_zero_fix='set'),
                                               geo.i4lapl(nr, maxnl, m, bc=self.bc["pol"], coeff=-1.0,
                                               with_sh_coeff='laplh', l_zero_fix='set')))

    def _create_coriolis(self,):
        nr, maxnl, m = self.res
        if self.inviscid:
            self._coriolis = scsp.bmat([[supp_geo.i2_nobc(nr, maxnl, m, bc=self.bc['tor'], coeff=-1.0j*m, l_zero_fix='set'),
                                         supp_geo.i2coriolis_nobc(nr, maxnl, m, bc=no_bc(), l_zero_fix='zero')],
                                        [geo.i2coriolis(nr, maxnl, m, bc=no_bc(), l_zero_fix='zero'),
                                         geo.i2lapl(nr, maxnl, m, bc=self.bc['pol'], coeff=1.0j*m, l_zero_fix='set')]], format='csc')
        else:
            self._coriolis = scsp.bmat([[geo.i2(nr, maxnl, m, bc=no_bc(), coeff=-1.0j*m, l_zero_fix='zero'),
                               geo.i2coriolis(nr, maxnl, m, bc=no_bc(), l_zero_fix='zero')],
                              [geo.i4coriolis(nr, maxnl, m, bc=no_bc(), l_zero_fix='zero'),
                               geo.i4lapl(nr, maxnl, m, bc=no_bc(), coeff=1.0j*m, l_zero_fix='zero')]])

    @property
    def mass(self):
        return self._mass

    @property
    def quasi_inverse(self):
        return self._quasi_inverse

    @property
    def diffusion(self):
        return self._diffusion

    @property
    def coriolis(self):
        return self._coriolis


if __name__ == "__main__":
    nr, maxnl, m = 11, 11, 1
    n_grid = 120
    with Timer("init op"):
        transform = WorlandTransform(nr, maxnl, m, n_grid)
    beta_mode = SphericalHarmonicMode("tor", 1, 0, "2 Sqrt[pi/3] r")
    with Timer("build induction"):
        induction_eq = InductionEquation(nr, maxnl, m)
        ind_op = induction_eq.induction(transform, [beta_mode], imposed_flow=False, quasi_inverse=False)
        # ind_op[np.abs(ind_op) < 1e-12] = 0
    plt.spy(ind_op).set_marker('.')
    plt.show()
