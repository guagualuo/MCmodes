from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from operators.polynomials import *
from operators.chebyshev_recurrence import *
from operators.threeJ_integrals import gaunt_matrix, elsasser_matrix


@dataclass
class ChebyshevTransform:
    """
    Class for Chebyshev transform in a spherical shell at a single m.

    Parameters
    -----
    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    ri: inner core radius (outer core radius assumed to be 1 + ri)

    n_grid: the number of radial grids.

    r_grid: the radial grids. Default is None. If given,  n_grid is ignored. For visualisation on usr defined grids.
        For projection to spectral space, one should not set this parameter.

    require_curl: whether compute operators Laplacian_l T_n(r)

    """
    nr: int
    maxnl: int
    m: int
    ri: float
    n_grid: int = None
    r_grid: np.ndarray = field(repr=False, default=None)
    require_curl: bool = True

    def __post_init__(self):
        self.res = self.nr, self.maxnl, self.m
        if self.r_grid is None:
            if self.n_grid < self.nr // 2 + 20:
                raise RuntimeWarning("Check if the physical grids is enough")
            self.r_grid = chebyshev_grid(self.n_grid, self.ri)
        else:
            self.n_grid = self.r_grid.shape[0]
        self.weight = np.ones(self.n_grid) * chebyshev_weight(self.n_grid)

        # init operators
        self._init_operators()
        if self.require_curl:
            self._init_curl_op()

    def _init_operators(self, ):
        nr, maxnl, m = self.res
        r_grid = self.r_grid
        ri = self.ri

        self.operators = {}
        self.transformers = {}
        self.operators['T'] = []
        self.operators['divrT'] = []
        self.operators['divrdiffrT'] = []
        self.operators['diff2rT'] = []
        self.operators['laplacianlT'] = []
        for l in range(m, maxnl):
            mat = chebyshev(nr, r_grid, ri)
            self.operators['T'].append(mat)
            self.operators['divrT'].append(divrT(nr, r_grid, ri))
            self.operators['divrdiffrT'].append(divrdiffrT(nr, r_grid, ri))
            self.operators['diff2rT'].append(diff2rT(nr, r_grid, ri))
            self.operators['laplacianlT'].append(laplacianlT(nr, l, r_grid, ri))
        for k, v in self.operators.items():
            self.operators[k] = scsp.csc_matrix(scsp.block_diag(v))

    def _init_curl_op(self):
        nr, maxnl, m = self.res
        weight = scsp.kron(scsp.identity(maxnl-m), scsp.diags(self.weight))
        self.transformers['curl'] = self.operators['T'].T @ weight @ self.operators['laplacianlT']

    @staticmethod
    def _compute_block_unit(left_op, right_op, weight, factor_mat, transformer=None):
        weight = scsp.kron(factor_mat, weight)
        if transformer is not None:
            return left_op.T @ weight @ right_op @ transformer
        else:
            return left_op.T @ weight @ right_op

    def _compute_block(self, beta_mode: SphericalHarmonicMode, sh_factor: Dict, terms: List[Tuple], r_factor: int):
        """ compute matrix for all l """
        nr, maxnl, m = self.res
        lb = beta_mode.l
        mat = scsp.csc_matrix((nr*(maxnl-m), nr*(maxnl-m)))
        for term in terms:
            if len(term) == 3:
                beta_op, alpha_op, factor = term
                transformer = None
            else:
                beta_op, alpha_op, factor, transformer = term
            radial = beta_op.apply(beta_mode.radial_expr, self.r_grid)
            weight = scsp.diags(self.weight * radial * self.r_grid**r_factor)
            factor_mat = np.zeros((maxnl-m, maxnl-m), dtype=np.complex128)
            for i, lg in enumerate(range(m, maxnl)):
                for j, la in enumerate(range(m, maxnl)):
                    factor_mat[i, j] = factor(la, lb, lg) * sh_factor[(lg, la)]

            mat += self._compute_block_unit(self.operators['T'], self.operators[alpha_op], weight, factor_mat,
                                            self.transformers.get(transformer, None))

        return mat

    def curl1tt(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        dim = (maxnl - m) * nr
        return scsp.csc_matrix((dim, dim))

    def curl1st(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg): return la*(la+1)
            terms = [(SymDivr(), 'T', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms, r_factor)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl1ts(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg): return lb * (lb + 1) * (-1) ** (la+lb+lg-1)
            terms = [(SymDivr(), 'T', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms, r_factor)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl1ss(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        nr, maxnl, m = self.res
        def l2(l): return l*(l+1)
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg): return 0.5*l2(la)*(l2(la)-l2(lb)-l2(lg))
            def factor_func2(la, lb, lg): return 0.5*l2(lb)*(l2(la)-l2(lb)+l2(lg))
            terms = [(SymDivrDiffr(), 'divrT', factor_func1),
                     (SymDivr(), 'divrdiffrT', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms, r_factor)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2tt(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg): return lg * (lg + 1)
            terms = [(SymDivr(), 'T', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms, r_factor)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2st(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        nr, maxnl, m = self.res
        def l2(l): return l * (l + 1)
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg):
                return -0.5 * l2(lg) * (l2(la) + l2(lb) - l2(lg)) + 0.5 * l2(la) * (l2(la) - l2(lb) - l2(lg))

            def factor_func2(la, lb, lg): return 0.5 * l2(la) * (l2(la) - l2(lb) - l2(lg))

            terms = [(SymDivr(), 'divrdiffrT', factor_func1),
                     (SymDiffDivr(), 'T', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms, r_factor)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2ts(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        nr, maxnl, m = self.res
        def l2(l): return l * (l + 1)
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg): return 0.5 * l2(lg) * (l2(la) + l2(lb) - l2(lg))
            def factor_func2(la, lb, lg): return -0.5 * l2(lb) * (-l2(la) + l2(lb) - l2(lg))

            terms = [(SymDivrDiffr(), 'divrT', factor_func1),
                     (SymDivr(), 'divrdiffrT', factor_func2),
                     (SymDiffDivr(), 'T', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms, r_factor)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2ss(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        nr, maxnl, m = self.res
        def l2(l): return l * (l + 1)
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg): return -l2(la)
            def factor_func2(la, lb, lg): return -l2(lb)
            def factor_func3(la, lb, lg): return l2(lg)

            terms = [
                     (SymDivr2Diffr(), 'divrdiffrT', factor_func1),
                     (SymrDiffDivr2Diffr(), 'divrT', factor_func1),
                     (SymDivr2(), 'diff2rT', factor_func2),
                     (SymDiffDivr(), 'divrdiffrT', factor_func2),
                     (SymDivr2Diffr(), 'divrdiffrT', factor_func3)
                     ]
            return self._compute_block(beta_mode, sh_factor, terms, r_factor)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl1curltt(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        return self.curl1st(beta_mode, r_factor)

    def curl1curlts(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        return self.curl1ss(beta_mode, r_factor)

    def curl2curltt(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        return self.curl2st(beta_mode, r_factor)

    def curl2curlts(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        return self.curl2ss(beta_mode, r_factor)

    def curl1curlst(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        dim = (maxnl - m) * nr
        return scsp.csc_matrix((dim, dim))

    def curl1curlss(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg):
                return -lb * (lb + 1) * (-1) ** (la + lb + lg - 1)

            terms = [(SymDivr(), 'laplacianlT', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms, r_factor)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2curlst(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg):
                return -lg * (lg + 1)

            terms = [(SymDivr(), 'laplacianlT', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms, r_factor)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2curlss(self, beta_mode: SphericalHarmonicMode, r_factor: int):
        nr, maxnl, m = self.res
        def l2(l): return l * (l + 1)

        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg):
                return -0.5 * l2(lg) * (l2(la) + l2(lb) - l2(lg))
            def factor_func2(la, lb, lg):
                return +0.5 * l2(lb) * (-l2(la) + l2(lb) - l2(lg))

            terms = [(SymDivrDiffr(), 'divrT', factor_func1, 'curl'),
                     (SymDivr(), 'divrdiffrT', factor_func2, 'curl'),
                     (SymDiffDivr(), 'laplacianlT', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms, r_factor)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))
