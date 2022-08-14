import quicc.geometry.spherical.shell as sgeo
import quicc.geometry.spherical.shell_radius as rad
import operators.quicc_supplements.shell_radius as supp_rad


def i4(nr, maxnl, m, a, b, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i4 radial operator kronecker with an identity"""

    return sgeo.make_sh_operator(rad.i4, nr, maxnl, m, a, b, bc, coeff,
                                 with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)


def i2_nobc(nr, maxnl, m, a, b, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i2 radial operator without tau lines kronecker with an identity"""

    return sgeo.make_sh_operator(supp_rad.i2_nobc, nr, maxnl, m, a, b, bc, coeff,
                                 with_sh_coeff=with_sh_coeff, l_zero_fix=l_zero_fix, restriction=restriction)


def i2r2_nobc(nr, maxnl, m, a, b, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i2r2 radial operator without tau lines kronecker with an identity"""

    return sgeo.make_sh_operator(supp_rad.i2r2_nobc, nr, maxnl, m, a, b, bc, coeff,
                                 with_sh_coeff=with_sh_coeff, l_zero_fix=l_zero_fix, restriction=restriction)


def i2r3_nobc(nr, maxnl, m, a, b, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i2r3 radial operator without tau lines kronecker with an identity"""

    return sgeo.make_sh_operator(supp_rad.i2r3_nobc, nr, maxnl, m, a, b, bc, coeff,
                                 with_sh_coeff=with_sh_coeff, l_zero_fix=l_zero_fix, restriction=restriction)


def i2r4lapl(nr, maxnl, m, a, b, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i2r3lapl radial operator kronecker with an identity"""

    return sgeo.make_sh_loperator(supp_rad.i2r4lapl, nr, maxnl, m, a, b, bc, coeff,
                                  with_sh_coeff=with_sh_coeff, l_zero_fix=l_zero_fix, restriction=restriction)


def i2r3coriolis(nr, maxnl, m, a, b, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i2r3 radial operator kronecker with coriolis Q term"""

    return sgeo.make_sh_qoperator(rad.i2r2, supp_rad.i2r3d1, nr, maxnl, m, a, b, bc, coeff,
                                  with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)


def i2r3coriolis_nobc(nr, maxnl, m, a, b, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i2r3 radial operator kronecker with coriolis Q term, without tau lines"""

    return sgeo.make_sh_qoperator(supp_rad.i2r2_nobc, supp_rad.i2r3d1_nobc, nr, maxnl, m, a, b, bc, coeff,
                                  with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)


def i2r4coriolis(nr, maxnl, m, a, b, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i2r4 radial operator kronecker with coriolis Q term"""

    return sgeo.make_sh_qoperator(rad.i2r3, supp_rad.i2r4d1, nr, maxnl, m, a, b, bc, coeff,
                                  with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)
