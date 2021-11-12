import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as scsp
from typing import Union, List, Dict, Tuple
from numba import jit

from operators.polynomials import *
from operators.threeJ_integrals import gaunt_matrix, elsasser_matrix
from utils import Timer


class WorlandTransform:
    def __init__(self, maxnl, nr, m, n_grid):
        self.res = nr, maxnl, m
        self.n_grid = n_grid
        if n_grid < nr + maxnl//2 + 10:
            raise RuntimeWarning("Check if the physical grids is enough")
        self.r_grid = worland_grid(n_grid)
        self.weight = np.ones(n_grid) * worland_weight(n_grid)

        # init operators
        self._init_operators()

    def _init_operators(self):
        nr, maxnl, m = self.res
        r_grid = self.r_grid
        self.operators = {}
        self.operators['W'] = {}
        self.operators['divrW'] = {}
        self.operators['divrdiffrW'] = {}
        self.operators['diff2rW'] = {}
        for l in range(m, maxnl):
            mat = worland(nr, l, r_grid)
            self.operators['W'][l] = mat
            self.operators['divrW'][l] = np.array(scsp.diags(1/r_grid) @ mat)
            self.operators['divrdiffrW'][l] = divrdiffrW(nr, l, r_grid)
            self.operators['diff2rW'][l] = diff2rW(nr, l, r_grid)

    def _compute_per_l_block(self, la, lg, beta_mode, beta_op: SymOperatorBase, alpha_op: str, factor):
        """ compute single combination of la, lg """
        radial = beta_op.apply(beta_mode.radial_expr, self.r_grid)
        weight = scsp.diags(factor * self.weight * radial)
        return self.operators['W'][lg].T @ weight @ self.operators[alpha_op][la]

    def _compute_block(self, beta_mode: SphericalHarmonicMode, sh_factor: Dict, terms: List[Tuple]):
        """ compute matrix for all l """
        nr, maxnl, m = self.res
        lb = beta_mode.l
        blocks = []
        for lg in range(m, maxnl):
            for la in range(m, maxnl):
                if sh_factor[(lg, la)] == 0:
                    blocks.append(scsp.csc_matrix((nr, nr)))
                else:
                    mat = scsp.csc_matrix((nr, nr))
                    for term in terms:
                        beta_op, alpha_op, factor = term
                        mat += self._compute_per_l_block(la, lg, beta_mode, beta_op, alpha_op,
                                                         factor(la, lb, lg)*sh_factor[(lg, la)])
                    blocks.append(mat)
        return scsp.bmat(np.reshape(np.array(blocks, dtype=object), (maxnl-m, maxnl-m)), format='csc')

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


if __name__ == "__main__":
    nr, maxnl, m = 11, 11, 1
    n_grid = 100
    with Timer("init op"):
        transform = WorlandTransform(nr, maxnl, m, n_grid)
    beta_mode = SphericalHarmonicMode("pol", 2, 0, "2 Sqrt[pi/3] r^2(r^4+r^2-1)")
    with Timer("comp op"):
        # op = transform.curl1st(beta_mode)
        # op = transform.curl1ts(beta_mode)
        # op = transform.curl1ss(beta_mode)
        # op = transform.curl2tt(beta_mode)
        # op = transform.curl2st(beta_mode)
        # op = transform.curl2ts(beta_mode)
        op = transform.curl2ss(beta_mode)
    # a = op.todense()[:nr, :nr]
    a = op.todense()[nr:2*nr, :nr]
    a[np.abs(a)<np.max(np.abs(a))*1e-13]=0
    print(a)
    # print(op.diagonal())
    # print(np.abs(op-scsp.diags(op.diagonal())).max())
    plt.spy(op).set_marker('.')
    plt.show()
