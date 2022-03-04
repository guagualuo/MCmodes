from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from operators.polynomials import *
from operators.threeJ_integrals import gaunt_matrix, elsasser_matrix
from utils import Timer


@dataclass
class WorlandTransform:
    """
    Class for Worland transform in a full sphere at a single m.

    Parameters
    -----
    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    n_grid: the number of radial grids.

    r_grid: the radial grids. Default is None. If given,  n_grid is ignored. For visualisation on usr defined grids.
        For projection to spectral space, one should not set this parameter.

    require_curl: whether compute operators Laplacian_l W_n^l(r)

    """
    nr: int
    maxnl: int
    m: int
    n_grid: int
    r_grid: np.ndarray = field(repr=False, default=None)
    require_curl: bool = True

    def __post_init__(self):
        self.res = self.nr, self.maxnl, self.m
        if self.r_grid is None:
            if self.n_grid < self.nr + self.maxnl // 2 + 10:
                raise RuntimeWarning("Check if the physical grids is enough")
            self.r_grid = worland_grid(self.n_grid)
            self.weight = np.ones(self.n_grid) * worland_weight(self.n_grid)
        else:
            self.n_grid = self.r_grid.shape[0]

        # init operators
        self._init_operators()
        if self.require_curl:
            self._init_curl_op()

    def _init_operators(self, ):
        nr, maxnl, m = self.res
        r_grid = self.r_grid
        self.operators = {}
        self.transformers = {}
        self.operators['W'] = []
        self.operators['divrW'] = []
        self.operators['divrdiffrW'] = []
        self.operators['diff2rW'] = []
        self.operators['laplacianlW'] = []
        for l in range(m, maxnl):
            mat = worland(nr, l, r_grid)
            self.operators['W'].append(mat)
            self.operators['divrW'].append(divrW(nr, l, r_grid))
            self.operators['divrdiffrW'].append(divrdiffrW(nr, l, r_grid))
            self.operators['diff2rW'].append(diff2rW(nr, l, r_grid))
            self.operators['laplacianlW'].append(laplacianlW(nr, l, r_grid))
        for k, v in self.operators.items():
            self.operators[k] = scsp.csc_matrix(scsp.block_diag(v))

    def _init_curl_op(self):
        nr, maxnl, m = self.res
        weight = scsp.kron(scsp.identity(maxnl-m), scsp.diags(self.weight))
        self.transformers['curl'] = self.operators['W'].T @ weight @ self.operators['laplacianlW']

    @staticmethod
    def _compute_block_numba1(left_op, right_op, weight, factor_mat):
        weight = scsp.kron(factor_mat, weight)
        return left_op.T @ weight @ right_op

    @staticmethod
    def _compute_block_numba2(left_op, right_op, transformer, weight, factor_mat):
        weight = scsp.kron(factor_mat, weight)
        return left_op.T @ weight @ right_op @ transformer

    def _compute_block(self, beta_mode: SphericalHarmonicMode, sh_factor: Dict, terms: List[Tuple]):
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
            # weight = np.diag(self.weight * radial)
            weight = scsp.diags(self.weight * radial)
            factor_mat = np.zeros((maxnl-m, maxnl-m), dtype=np.complex128)
            for i, lg in enumerate(range(m, maxnl)):
                for j, la in enumerate(range(m, maxnl)):
                    factor_mat[i, j] = factor(la, lb, lg) * sh_factor[(lg, la)]

            if transformer is None:
                mat += self._compute_block_numba1(self.operators['W'], self.operators[alpha_op], weight, factor_mat)
            else:
                mat += self._compute_block_numba2(self.operators['W'], self.operators[alpha_op],
                                                         self.transformers[transformer], weight, factor_mat)
        return mat

    def curl1tt(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        dim = (maxnl - m) * nr
        return scsp.csc_matrix((dim, dim))

    def curl1st(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg): return la*(la+1)
            terms = [(SymDivr(), 'W', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl1ts(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg): return lb * (lb + 1) * (-1) ** (la+lb+lg-1)
            terms = [(SymDivr(), 'W', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl1ss(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        def l2(l): return l*(l+1)
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg): return 0.5*l2(la)*(l2(la)-l2(lb)-l2(lg))
            def factor_func2(la, lb, lg): return 0.5*l2(lb)*(l2(la)-l2(lb)+l2(lg))
            terms = [(SymDivrDiffr(), 'divrW', factor_func1),
                     (SymDivr(), 'divrdiffrW', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2tt(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg): return lg * (lg + 1)
            terms = [(SymDivr(), 'W', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2st(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        def l2(l): return l * (l + 1)
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg):
                return -0.5 * l2(lg) * (l2(la) + l2(lb) - l2(lg)) + 0.5 * l2(la) * (l2(la) - l2(lb) - l2(lg))

            def factor_func2(la, lb, lg): return 0.5 * l2(la) * (l2(la) - l2(lb) - l2(lg))

            terms = [(SymDivr(), 'divrdiffrW', factor_func1),
                     (SymDiffDivr(), 'W', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2ts(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        def l2(l): return l * (l + 1)
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg): return 0.5 * l2(lg) * (l2(la) + l2(lb) - l2(lg))
            def factor_func2(la, lb, lg): return -0.5 * l2(lb) * (-l2(la) + l2(lb) - l2(lg))

            terms = [(SymDivrDiffr(), 'divrW', factor_func1),
                     (SymDivr(), 'divrdiffrW', factor_func2),
                     (SymDiffDivr(), 'W', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2ss(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        def l2(l): return l * (l + 1)
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg): return -l2(la)
            def factor_func2(la, lb, lg): return -l2(lb)
            def factor_func3(la, lb, lg): return l2(lg)

            terms = [
                     (SymDivr2Diffr(), 'divrdiffrW', factor_func1),
                     (SymrDiffDivr2Diffr(), 'divrW', factor_func1),
                     (SymDivr2(), 'diff2rW', factor_func2),
                     (SymDiffDivr(), 'divrdiffrW', factor_func2),
                     (SymDivr2Diffr(), 'divrdiffrW', factor_func3)
                     ]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl1curltt(self, beta_mode: SphericalHarmonicMode):
        return self.curl1st(beta_mode)

    def curl1curlts(self, beta_mode: SphericalHarmonicMode):
        return self.curl1ss(beta_mode)

    def curl2curltt(self, beta_mode: SphericalHarmonicMode):
        return self.curl2st(beta_mode)

    def curl2curlts(self, beta_mode: SphericalHarmonicMode):
        return self.curl2ss(beta_mode)

    def curl1curlst(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        dim = (maxnl - m) * nr
        return scsp.csc_matrix((dim, dim))

    def curl1curlss(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg):
                return -lb * (lb + 1) * (-1) ** (la + lb + lg - 1)

            terms = [(SymDivr(), 'laplacianlW', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2curlst(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg):
                return -lg * (lg + 1)

            terms = [(SymDivr(), 'laplacianlW', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))

    def curl2curlss(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        def l2(l): return l * (l + 1)

        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg):
                return -0.5 * l2(lg) * (l2(la) + l2(lb) - l2(lg))
            def factor_func2(la, lb, lg):
                return +0.5 * l2(lb) * (-l2(la) + l2(lb) - l2(lg))

            terms = [(SymDivrDiffr(), 'divrW', factor_func1, 'curl'),
                     (SymDivr(), 'divrdiffrW', factor_func2, 'curl'),
                     (SymDiffDivr(), 'laplacianlW', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix((dim, dim))


if __name__ == "__main__":
    nr, maxnl, m = 11, 11, 1
    np.set_printoptions(16)
    n_grid = 100
    with Timer("init op"):
        transform = WorlandTransform(nr, maxnl, m, n_grid, require_curl=True)
    beta_mode = SphericalHarmonicMode("pol", 1, 0, "1/5 Sqrt[pi/3] r(5-3r^2)")
    with Timer("comp op"):
        # op = transform.curl1st(beta_mode)
        # op = transform.curl1ts(beta_mode)
        # op = transform.curl1ss(beta_mode)
        # op = transform.curl2tt(beta_mode)
        # op = transform.curl2st(beta_mode)
        # op = transform.curl2ts(beta_mode)
        # op = transform.curl2ss(beta_mode)
        op = transform.curl2curlss(beta_mode)
    # a = op.todense()[:nr, nr:2*nr]
    a = op.todense()[9*nr:10*nr, 8*nr:9*nr]
    a[np.abs(a)<np.max(np.abs(a))*1e-13]=0
    print(a)
    # print(op.diagonal())
    # print(np.abs(op-scsp.diags(op.diagonal())).max())
    plt.spy(op).set_marker('.')
    plt.show()
