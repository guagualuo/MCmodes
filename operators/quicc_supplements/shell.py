import quicc.geometry.spherical.shell as sgeo
import quicc.geometry.spherical.shell_radius as rad


def i4(nr, maxnl, m, a, b, bc, coeff = 1.0, with_sh_coeff = None, l_zero_fix = False, restriction = None):
    """Create a i4 radial operator kronecker with an identity"""

    return sgeo.make_sh_operator(rad.i4, nr, maxnl, m, a, b, bc, coeff,
                                 with_sh_coeff = with_sh_coeff, l_zero_fix = l_zero_fix, restriction = restriction)
