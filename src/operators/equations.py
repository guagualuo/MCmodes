import matplotlib.pyplot as plt
import numpy as np
from typing import List
import scipy.sparse as scsp
from abc import ABC

from operators.worland_transform import WorlandTransform
from operators.polynomials import SphericalHarmonicMode
from utils import Timer
import quicc.geometry.spherical.sphere_worland as geo
import operators.quicc_supplements.sphere_worland as supp_geo
from quicc.geometry.spherical.sphere_boundary_worland import no_bc


class BaseEquation(ABC):
    def __init__(self, nr, maxnl, m):
        self.res = nr, maxnl, m


class InductionEquation(BaseEquation):
    """ Operators in the induction equations """
    def __init__(self, nr, maxnl, m):
        super(InductionEquation, self).__init__(nr, maxnl, m)
        self.res = nr, maxnl, m
        self.bc = {'tor': {0: 10}, 'pol': {0: 13}}

    def induction(self, transform: WorlandTransform, beta_modes: List[SphericalHarmonicMode],
                  imposed_flow: bool, quasi_inverse: bool):
        """ Induction term curl (u x B_0), in which B_0 is the background field.
                [ r.curl2(t_a x B_0), r.curl2(s_a x B_0)
                  r.curl1(t_a x B0), r.curl1(s_a x B_0)]"""
        nr, maxnl, m = self.res
        if len(beta_modes) > 0:
            tt = sum([transform.curl2tt(mode) + transform.curl2ts(mode) for mode in beta_modes])
            ts = sum([transform.curl2st(mode) + transform.curl2ss(mode) for mode in beta_modes])
            st = sum([transform.curl1tt(mode) + transform.curl1ts(mode) for mode in beta_modes])
            ss = sum([transform.curl1st(mode) + transform.curl1ss(mode) for mode in beta_modes])
            sign = -1.0 if imposed_flow else 1.0
            op = sign * scsp.bmat([[tt, ts], [st, ss]], format='csc')
            if quasi_inverse:
                op = self.quasi_inverse() @ op
            return op
        else:
            return scsp.csc_matrix((2*nr*(maxnl-m), 2*nr*(maxnl-m)))

    def quasi_inverse(self):
        nr, maxnl, m = self.res
        return scsp.block_diag((geo.i2(nr, maxnl, m, no_bc(), l_zero_fix='zero'),
                                geo.i2(nr, maxnl, m, no_bc(), l_zero_fix='zero')))

    def mass(self):
        nr, maxnl, m = self.res
        return scsp.block_diag((geo.i2(nr, maxnl, m, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                                geo.i2(nr, maxnl, m, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero')))

    def diffusion(self, bc=True):
        """ Build the dissipation matrix for the magnetic field, insulating boundary condition """
        nr, maxnl, m = self.res
        if bc:
            return scsp.block_diag((geo.i2lapl(nr, maxnl, m, bc=self.bc['tor'], with_sh_coeff='laplh', l_zero_fix='set'),
                                    geo.i2lapl(nr, maxnl, m, bc=self.bc['pol'], with_sh_coeff='laplh', l_zero_fix='set')))
        else:
            return scsp.block_diag((geo.i2lapl(nr, maxnl, m, bc=no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                                    geo.i2lapl(nr, maxnl, m, bc=no_bc(), with_sh_coeff='laplh', l_zero_fix='zero')))


class MomentumEquation(BaseEquation):
    """ Operators in the momentum equation """
    def __init__(self, nr, maxnl, m, inviscid: bool, bc_type: str = None):
        super(MomentumEquation, self).__init__(nr, maxnl, m)
        self.inviscid = inviscid
        if not inviscid:
            assert bc_type in ['no-slip', 'stress-free']
            self.bc_type = bc_type
        self.set_bc()

    def set_bc(self):
        if self.inviscid:
            self.bc = {'tor': no_bc(), 'pol': {0: 10}}
        else:
            if self.bc_type == 'no-slip':
                self.bc = {"tor": {0: 10}, "pol": {0: 20}}
            else:
                self.bc = {"tor": {0: 12}, "pol": {0: 21}}

    def lorentz1(self, transform: WorlandTransform, modes: List[SphericalHarmonicMode]):
        """ Lorentz term (curl B_0) x b, in which B_0 is the background field.
                    [ r.curl1(J0 x b), r.curl1(J0 x b)
                      r.curl2(J0 x b), r.curl2(J0 x b)]"""
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

    def lorentz2(self, transform: WorlandTransform, modes: List[SphericalHarmonicMode]):
        """ Lorentz term (curl b) x B_0, in which B_0 is the background field.
                    [ r.curl1(j x b), r.curl1(j x b)
                      r.curl2(j x b), r.curl2(j x b)]"""
        nr, maxnl, m = self.res
        if len(modes) > 0:
            tt = sum([transform.curl1curltt(mode) + transform.curl1curlts(mode) for mode in modes])
            ts = sum([transform.curl1curlst(mode) + transform.curl1curlss(mode) for mode in modes])
            st = sum([transform.curl2curltt(mode) + transform.curl2curlts(mode) for mode in modes])
            ss = sum([transform.curl2curlst(mode) + transform.curl2curlss(mode) for mode in modes])
            return scsp.bmat([[tt, ts], [st, ss]], format='csc')
        else:
            return scsp.csc_matrix((2*nr*(maxnl-m), 2*nr*(maxnl-m)))

    def lorentz(self, transform: WorlandTransform, field_modes: List[SphericalHarmonicMode], quasi_inverse=True):
        op = self.lorentz1(transform, field_modes) + self.lorentz2(transform, field_modes)
        if quasi_inverse:
            op = self.quasi_inverse() @ op
        return op

    def advection(self, transform: WorlandTransform, flow_modes: List[SphericalHarmonicMode], quasi_inverse=True):
        return self.lorentz(transform, flow_modes, quasi_inverse)

    def mass(self):
        nr, maxnl, m = self.res
        if self.inviscid:
            return scsp.block_diag((supp_geo.i2_nobc(nr, maxnl, m, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                                  geo.i2lapl(nr, maxnl, m, no_bc(), -1.0, with_sh_coeff='laplh', l_zero_fix='zero')))
        else:
            return scsp.block_diag((geo.i2(nr, maxnl, m, no_bc(), with_sh_coeff='laplh', l_zero_fix='zero'),
                                    geo.i4lapl(nr, maxnl, m, no_bc(), -1.0, with_sh_coeff='laplh', l_zero_fix='zero')))

    def quasi_inverse(self):
        nr, maxnl, m = self.res
        if self.inviscid:
            return scsp.block_diag((supp_geo.i2_nobc(nr, maxnl, m, no_bc(), l_zero_fix='zero'),
                                    geo.i2(nr, maxnl, m, no_bc(), l_zero_fix='zero')))
        else:
            return scsp.block_diag((geo.i2(nr, maxnl, m, no_bc(), l_zero_fix='zero'),
                                    geo.i4(nr, maxnl, m, no_bc(), l_zero_fix='zero')))

    def diffusion(self, bc=True):
        nr, maxnl, m = self.res
        dim = nr * (maxnl - m)
        if self.inviscid:
            return scsp.csc_matrix(2*dim, 2*dim)
        else:
            if bc:
                return scsp.block_diag((geo.i2lapl(nr, maxnl, m, bc=self.bc["tor"],
                                               with_sh_coeff='laplh', l_zero_fix='set'),
                                    geo.i4lapl(nr, maxnl, m, bc=self.bc["pol"], coeff=-1.0,
                                               with_sh_coeff='laplh', l_zero_fix='set')))
            else:
                return scsp.block_diag((geo.i2lapl(nr, maxnl, m, bc=no_bc(),
                                                   with_sh_coeff='laplh', l_zero_fix='zero'),
                                        geo.i4lapl(nr, maxnl, m, bc=no_bc(), coeff=-1.0,
                                                   with_sh_coeff='laplh', l_zero_fix='zero')))

    def coriolis(self, bc: bool):
        nr, maxnl, m = self.res
        if self.inviscid:
            if bc:
                return scsp.bmat([[supp_geo.i2_nobc(nr, maxnl, m, bc=self.bc['tor'], coeff=-1.0j*m, l_zero_fix='set'),
                                   supp_geo.i2coriolis_nobc(nr, maxnl, m, bc=no_bc(), l_zero_fix='zero')],
                                  [geo.i2coriolis(nr, maxnl, m, bc=no_bc(), l_zero_fix='zero'),
                                   geo.i2lapl(nr, maxnl, m, bc=self.bc['pol'], coeff=1.0j*m, l_zero_fix='set')]])
            else:
                return scsp.bmat([[supp_geo.i2_nobc(nr, maxnl, m, bc=no_bc(), coeff=-1.0j*m, l_zero_fix='zero'),
                                   supp_geo.i2coriolis_nobc(nr, maxnl, m, bc=no_bc(), l_zero_fix='zero')],
                                  [geo.i2coriolis(nr, maxnl, m, bc=no_bc(), l_zero_fix='zero'),
                                   geo.i2lapl(nr, maxnl, m, bc=no_bc(), coeff=1.0j*m, l_zero_fix='zero')]])
        else:
            return scsp.bmat([[geo.i2(nr, maxnl, m, bc=no_bc(), coeff=-1.0j*m, l_zero_fix='zero'),
                               geo.i2coriolis(nr, maxnl, m, bc=no_bc(), l_zero_fix='zero')],
                              [geo.i4coriolis(nr, maxnl, m, bc=no_bc(), l_zero_fix='zero'),
                               geo.i4lapl(nr, maxnl, m, bc=no_bc(), coeff=1.0j*m, l_zero_fix='zero')]])


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
