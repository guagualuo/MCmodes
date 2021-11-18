import quicc.geometry.spherical.sphere_worland as geo
import operators.quicc_supplements.sphere_radius_worland as supp_rad


def i2_nobc(nr, maxnl, m, bc, coeff=1.0, with_sh_coeff=None, l_zero_fix=False, restriction=None):
    """Create a i2 radial operator kronecker with an identity, with no tau lines, replace with line N+1, N+2"""

    return geo.make_sh_loperator(supp_rad.i2_nobc, nr, maxnl, m, bc, coeff, with_sh_coeff=with_sh_coeff, l_zero_fix=l_zero_fix,
                             restriction=restriction)


def i2coriolis_nobc(nr, maxnl, m, bc, coeff = 1.0, with_sh_coeff=None, l_zero_fix=False, restriction=None):
    """ Create i2 coriolis operator"""

    return geo.make_sh_qoperator(supp_rad.i2qm_nobc, supp_rad.i2qp_nobc, nr, maxnl, m, bc, coeff,
                                with_sh_coeff=with_sh_coeff, l_zero_fix=l_zero_fix, restriction=restriction)
