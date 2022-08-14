from dataclasses import dataclass
from abc import ABC
from typing import List
import numpy as np
import scipy.sparse as scsp

from operators import WorlandTransform, ChebyshevTransform
from operators.chebyshev_recurrence import quicc_norm, inv_quicc_norm
from operators.polynomials import SphericalHarmonicMode
import quicc.geometry.spherical.sphere_worland as geo
import quicc.geometry.spherical.shell as sgeo
import operators.quicc_supplements.sphere_worland as supp_geo
import operators.quicc_supplements.shell as supp_sgeo
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
            self.bc_type = self.bc_type.lower()
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


@dataclass
class InductionEquationShell(_BaseEquation):
    """
    Class for the induction equation in a spherical shell

    Parameters
    -----

    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    ri: inner core radius, outer core radius be ri+1

    """
    ri: float

    def __post_init__(self):
        super(InductionEquationShell, self).__post_init__()
        self.bc = {'tor': {0: 20}, 'pol': {0: 23}}
        a, b = self._get_ab_consts()
        self.bc['pol']['c'] = {'a': a, 'b': b, 'l': -1}
        # initialise simple linear operators
        self._init_operators()
        # set r factors to multiply to tor/pol component of the equation
        self.r_factor_tor = 3
        self.r_factor_pol = 2
        self.galerkin = False

    def _get_ab_consts(self):
        return 0.5, 0.5 + self.ri

    def _init_operators(self):
        """
        Initialise simple linear operators
        """
        super(InductionEquationShell, self)._init_operators()

    def induction(self,
                  transform: ChebyshevTransform,
                  beta_modes: List[SphericalHarmonicMode],
                  imposed_flow: bool,
                  quasi_inverse: bool,
                  ):
        """
        Induction term curl (u x B_0), in which B_0 is the background field.
                r^2 [ r.curl2(t_a x B_0), r.curl2(s_a x B_0)
                    r.curl1(t_a x B0), r.curl1(s_a x B_0)]
        r^2 factor to cast terms into polynomials, consistent with the diffusion term

        Parameters
        -----
        transform: Chebyshev transform obeject that implements all 8 possible combinations

        beta_modes: Spherical harmonic modes of B_0 or u

        imposed_flow: whether impose u or B in the induction term curl(u x B).
            If set True, it can be used for the kinematic dynamo problem, for example.

        quasi_inverse: whether to apply quasi-inverse operator to the induction.

        """
        nr, maxnl, m = self.res
        r_factor_tor, r_factor_pol = self.r_factor_tor, self.r_factor_pol
        if len(beta_modes) > 0:
            tt = sum([transform.curl2tt(mode, r_factor_tor) + transform.curl2ts(mode, r_factor_tor) for mode in beta_modes])
            ts = sum([transform.curl2st(mode, r_factor_tor) + transform.curl2ss(mode, r_factor_tor) for mode in beta_modes])
            st = sum([transform.curl1tt(mode) + transform.curl1ts(mode, r_factor_pol) for mode in beta_modes])
            ss = sum([transform.curl1st(mode, r_factor_pol) + transform.curl1ss(mode, r_factor_pol) for mode in beta_modes])
            sign = -1.0 if imposed_flow else 1.0
            op = sign * self._inv_quicc_norm @ scsp.bmat([[tt, ts], [st, ss]], format='csc') @ self._quicc_norm
            if quasi_inverse:
                op = self.quasi_inverse @ op
            return op
        else:
            return scsp.csc_matrix((2*nr*(maxnl-m), 2*nr*(maxnl-m)))

    def _create_quicc_norm(self):
        nr, maxnl, m = self.res
        self._quicc_norm = scsp.diags(np.concatenate([quicc_norm(nr) for _ in range(2*(maxnl-m))]))
        self._inv_quicc_norm = scsp.diags(np.concatenate([inv_quicc_norm(nr) for _ in range(2*(maxnl-m))]))

    def _create_quasi_inverse(self):
        """
        Quasi-inverse operator, only apply to the induction part with pre-multiplied r^2 factor
        """
        nr, maxnl, m = self.res
        a, b = self._get_ab_consts()

        if self.bc is not None:
            bc = no_bc()
            self._quasi_inverse = scsp.block_diag((sgeo.i2(nr, maxnl, m, a, b, bc, l_zero_fix='zero'),
                                                   sgeo.i2(nr, maxnl, m, a, b, bc, l_zero_fix='zero')))

    def _create_mass(self):
        """
        Mass operator
        """
        nr, maxnl, m = self.res
        bc = no_bc()
        a, b = self._get_ab_consts()

        self._mass = scsp.block_diag((sgeo.i2r3(nr, maxnl, m, a, b, bc, with_sh_coeff='laplh', l_zero_fix='zero'),
                                      sgeo.i2r2(nr, maxnl, m, a, b, bc, with_sh_coeff='laplh', l_zero_fix='zero')))

    def _create_diffusion(self):
        """
        Build the dissipation matrix for the magnetic field, insulating boundary condition
        """
        nr, maxnl, m = self.res
        a, b = self._get_ab_consts()
        self._diffusion = scsp.block_diag((sgeo.i2r3lapl(nr, maxnl, m, a, b, bc=self.bc['tor'], with_sh_coeff='laplh', l_zero_fix='set'),
                                           sgeo.i2r2lapl(nr, maxnl, m, a, b, bc=self.bc['pol'], with_sh_coeff='laplh', l_zero_fix='set')))

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
class MomentumEquationShell(_BaseEquation):
    """
    Class for the linearised momentum equation at a single m

    Parameters
    -----

    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    ri: inner core radius

    inviscid: If set True, viscous dissipation is set 0

    bc_type: boundary condtion for u, 'no-slip' or 'stress-free'. Ignored when inviscid is True.

    """
    ri: float
    inviscid: bool = True
    bc_type: str = None

    def __post_init__(self):
        super(MomentumEquationShell, self).__post_init__()
        if not self.inviscid:
            assert self.bc_type.lower() in ['no-slip', 'stress-free']
            self.bc_type = self.bc_type.lower()
        self._set_bc()
        self._init_operators()
        # r factors for tor/pol components of the equation
        self.r_factor_tor = 3
        self.r_factor_pol = 4

    def _get_ab_consts(self):
        return 0.5, 0.5 + self.ri

    def _set_bc(self):
        a, b = self._get_ab_consts()
        if self.inviscid:
            self.bc = {'tor': no_bc(), 'pol': {0: 20}}
        else:
            if self.bc_type == 'no-slip':
                self.bc = {"tor": {0: 20}, "pol": {0: 40}}
                self.bc['pol']['c'] = {'a': a, 'b': b}
            if self.bc_type == 'stress-free':
                self.bc = {"tor": {0: 22}, "pol": {0: 41}}
                self.bc['tor']['c'] = {'a': a, 'b': b}
                self.bc['pol']['c'] = {'a': a, 'b': b}

    def lorentz1(self,
                 transform: ChebyshevTransform,
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
        r_factor_tor, r_factor_pol = self.r_factor_tor, self.r_factor_pol
        if len(modes) > 0:
            curl_modes = [mode.curl() for mode in modes]
            tt = sum([transform.curl1tt(mode) + transform.curl1ts(mode, r_factor_tor) for mode in curl_modes])
            ts = sum([transform.curl1st(mode, r_factor_tor) + transform.curl1ss(mode, r_factor_tor) for mode in curl_modes])
            st = sum([transform.curl2tt(mode, r_factor_pol) + transform.curl2ts(mode, r_factor_pol) for mode in curl_modes])
            ss = sum([transform.curl2st(mode, r_factor_pol) + transform.curl2ss(mode, r_factor_pol) for mode in curl_modes])
            return -scsp.bmat([[tt, ts], [st, ss]], format='csc')
        else:
            return scsp.csc_matrix((2*nr*(maxnl-m), 2*nr*(maxnl-m)))

    def lorentz2(self,
                 transform: ChebyshevTransform,
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
        r_factor_tor, r_factor_pol = self.r_factor_tor, self.r_factor_pol
        if len(modes) > 0:
            tt = sum([transform.curl1curltt(mode, r_factor_tor) + transform.curl1curlts(mode, r_factor_tor) for mode in modes])
            ts = sum([transform.curl1curlst(mode) + transform.curl1curlss(mode, r_factor_tor) for mode in modes])
            st = sum([transform.curl2curltt(mode, r_factor_pol) + transform.curl2curlts(mode, r_factor_pol) for mode in modes])
            ss = sum([transform.curl2curlst(mode, r_factor_pol) + transform.curl2curlss(mode, r_factor_pol) for mode in modes])
            return scsp.bmat([[tt, ts], [st, ss]], format='csc')
        else:
            return scsp.csc_matrix((2*nr*(maxnl-m), 2*nr*(maxnl-m)))

    def lorentz(self,
                transform: ChebyshevTransform,
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
        op = self._inv_quicc_norm @ op @ self._quicc_norm
        if quasi_inverse:
            op = self.quasi_inverse @ op
        return op.tocsc()

    def advection(self,
                  transform: ChebyshevTransform,
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

    def _create_quicc_norm(self):
        nr, maxnl, m = self.res
        self._quicc_norm = scsp.diags(np.concatenate([quicc_norm(nr) for _ in range(2*(maxnl-m))]))
        self._inv_quicc_norm = scsp.diags(np.concatenate([inv_quicc_norm(nr) for _ in range(2*(maxnl-m))]))

    def _create_mass(self):
        nr, maxnl, m = self.res
        a, b = self._get_ab_consts()
        if self.inviscid:
            self._mass = scsp.block_diag((supp_sgeo.i2r3_nobc(nr, maxnl, m, a, b, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                                          supp_sgeo.i2r4lapl(nr, maxnl, m, a, b, no_bc(), -1.0, with_sh_coeff='laplh', l_zero_fix='zero')))
        else:
            self._mass = scsp.block_diag((sgeo.i2r3(nr, maxnl, m, a, b, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                                          sgeo.i4r4lapl(nr, maxnl, m, a, b, no_bc(), -1.0, with_sh_coeff='laplh', l_zero_fix='zero')))

    def _create_quasi_inverse(self):
        nr, maxnl, m = self.res
        a, b = self._get_ab_consts()
        if self.inviscid:
            self._quasi_inverse = scsp.block_diag((supp_sgeo.i2_nobc(nr, maxnl, m, a, b, no_bc(), l_zero_fix='zero'),
                                                   sgeo.i2(nr, maxnl, m, a, b, no_bc(), l_zero_fix='zero')))
        else:
            self._quasi_inverse = scsp.block_diag((sgeo.i2(nr, maxnl, m, a, b, no_bc(), l_zero_fix='zero'),
                                                   supp_sgeo.i4(nr, maxnl, m, a, b, no_bc(), l_zero_fix='zero')))

    def _create_diffusion(self,):
        nr, maxnl, m = self.res
        a, b = self._get_ab_consts()
        dim = nr * (maxnl - m)
        if self.inviscid:
            self._diffusion = scsp.csc_matrix((2*dim, 2*dim))
        else:
            self._diffusion = scsp.block_diag((sgeo.i2r3lapl(nr, maxnl, m, a, b, bc=self.bc["tor"],
                                               with_sh_coeff='laplh', l_zero_fix='set'),
                                               sgeo.i4r4lapl(nr, maxnl, m, a, b, bc=self.bc["pol"], coeff=-1.0,
                                               with_sh_coeff='laplh', l_zero_fix='set')))

    def _create_coriolis(self,):
        nr, maxnl, m = self.res
        a, b = self._get_ab_consts()
        if self.inviscid:
            self._coriolis = scsp.bmat([[supp_sgeo.i2r3_nobc(nr, maxnl, m, a, b, bc=self.bc['tor'], coeff=-1.0j*m, l_zero_fix='set'),
                                         supp_sgeo.i2r3coriolis_nobc(nr, maxnl, m, a, b, bc=no_bc(), l_zero_fix='zero')],
                                        [supp_sgeo.i2r4coriolis(nr, maxnl, m, a, b, bc=no_bc(), l_zero_fix='zero'),
                                         supp_sgeo.i2r4lapl(nr, maxnl, m, a, b, bc=self.bc['pol'], coeff=1.0j*m, l_zero_fix='set')]], format='csc')
        else:
            self._coriolis = scsp.bmat([[sgeo.i2r3(nr, maxnl, m, a, b, bc=no_bc(), coeff=-1.0j*m, l_zero_fix='zero'),
                                         supp_sgeo.i2r3coriolis(nr, maxnl, m, a, b, bc=no_bc(), l_zero_fix='zero')],
                                        [sgeo.i4r4coriolis(nr, maxnl, m, a, b, bc=no_bc(), l_zero_fix='zero'),
                                         sgeo.i4r4lapl(nr, maxnl, m, a, b, bc=no_bc(), coeff=1.0j*m, l_zero_fix='zero')]])

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
