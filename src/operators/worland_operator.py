import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as scsp
from typing import Union, List, Dict, Tuple

from polynomials import *
from threeJ_integrals import gaunt_matrix, elsasser_matrix, SphericalHarmonicMode
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
        for l in range(m, maxnl):
            mat = worland(nr, l, r_grid)
            self.operators['W'][l] = mat
            self.operators['divrW'][l] = np.array(scsp.diags(1/r_grid).dot(mat))
            self.operators['divrdiffrW'][l] = divrdiffrW(nr, l, r_grid)

    def _compute_per_l_block(self, la, lg, beta_mode, beta_op, alpha_op, factor):
        """ compute single combination of la, lg """
        beta_op = sym_operators(beta_op)
        radial = beta_op(beta_mode.radial_expr, self.r_grid)
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
                    blocks.append(None)
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
        return scsp.csc_matrix(dim, dim)

    def curl1st(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg): return la*(la+1)
            terms = [('divr', 'W', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix(dim, dim)

    def curl1ts(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg): return lb * (lb + 1) * (-1) ** (la+lb+lg-1)
            terms = [('divr', 'W', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms)
            # blocks = []
            # for lg in range(m, maxnl):
            #     for la in range(m, maxnl):
            #         if elsasser[(lg, la)] == 0:
            #             blocks.append(None)
            #         else:
            #             radial = sym_divr(beta_mode.radial_expr, self.r_grid)
            #             weight = scsp.diags(self.weight * radial * lb * (lb + 1) * (-1) ** (la+lb+lg-1))*elsasser[(lg, la)]
            #             blocks.append(self.W[lg].T @ weight @ self.W[la])
            # return scsp.bmat(np.reshape(np.array(blocks, dtype=object), (maxnl - m, maxnl - m)), format='csc')
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix(dim, dim)

    def curl1ss(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        def l2(l): return l*(l+1)
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg): return 0.5*l2(la)*(l2(la)-l2(lb)-l2(lg))
            def factor_func2(la, lb, lg): return 0.5*l2(lb)*(l2(la)-l2(lb)+l2(lg))
            terms = [('divrdiffr', 'divr', factor_func1),
                     ('divr', 'divrdiffr', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms)
            # blocks = []
            # for lg in range(m, maxnl):
            #     for la in range(m, maxnl):
            #         if gaunt[(lg, la)] == 0:
            #             blocks.append(None)
            #         else:
            #             radial1 = sym_divrdiffr(beta_mode.radial_expr, self.r_grid)
            #             radial2 = sym_divr(beta_mode.radial_expr, self.r_grid)
            #             factor1 = 0.5*l2(la)*(l2(la)-l2(lb)-l2(lg))
            #             factor2 = 0.5*l2(lb)*(l2(la)-l2(lb)+l2(lg))
            #             weight1 = scsp.diags(factor1*self.weight * radial1) * gaunt[(lg, la)]
            #             weight2 = scsp.diags(factor2*self.weight * radial2) * gaunt[(lg, la)]
            #
            #             mat = self.W[lg].T @ weight1 @ self.divrW[la]
            #             mat += self.W[lg].T @ weight2 @ self.divrdiffrW[la]
            #             blocks.append(mat)
            # return scsp.bmat(np.reshape(np.array(blocks, dtype=object), (maxnl - m, maxnl - m)), format='csc')
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix(dim, dim)

    def curl2tt(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = elsasser_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func(la, lb, lg): return lg * (lg + 1)
            terms = [('divr', 'W', factor_func)]
            return self._compute_block(beta_mode, sh_factor, terms)
            # blocks = []
            # for lg in range(m, maxnl):
            #     for la in range(m, maxnl):
            #         if elsasser[(lg, la)] == 0:
            #             blocks.append(None)
            #         else:
            #             radial = sym_divr(beta_mode.radial_expr, self.r_grid)
            #             weight = scsp.diags(self.weight * radial * lg * (lg + 1)) * elsasser[(lg, la)]
            #             blocks.append(self.W[lg].T @ weight @ self.W[la])
            # return scsp.bmat(np.reshape(np.array(blocks, dtype=object), (maxnl - m, maxnl - m)), format='csc')
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix(dim, dim)

    def curl2st(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        def l2(l): return l * (l + 1)
        if beta_mode.comp == 'tor':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg):
                return -0.5 * l2(lg) * (l2(la) + l2(lb) - l2(lg)) + 0.5 * l2(la) * (l2(la) - l2(lb) - l2(lg))

            def factor_func2(la, lb, lg): return 0.5 * l2(la) * (l2(la) - l2(lb) - l2(lg))

            terms = [('divr', 'divrdiffr', factor_func1),
                     ('diffdivr', 'W', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms)

            # blocks = []
            # for lg in range(m, maxnl):
            #     for la in range(m, maxnl):
            #         if gaunt[(lg, la)] == 0:
            #             blocks.append(None)
            #         else:
            #             factor1 = -0.5 * l2(lg) * (l2(la) + l2(lb) - l2(lg))
            #             factor2 = 0.5 * l2(la) * (l2(la) - l2(lb) - l2(lg))
            #             radial = sym_divr(beta_mode.radial_expr, self.r_grid)
            #             weight = scsp.diags((factor1 + factor2) * self.weight * radial) * gaunt[(lg, la)]
            #             mat = self.W[lg].T @ weight @ self.divrdiffrW[la]
            #
            #             radial = sym_diffdivr(beta_mode.radial_expr, self.r_grid)
            #             weight = scsp.diags(factor2 * self.weight * radial) * gaunt[(lg, la)]
            #             mat += self.W[lg].T @ weight @ self.W[la]
            #             blocks.append(mat)
            # return scsp.bmat(np.reshape(np.array(blocks, dtype=object), (maxnl - m, maxnl - m)), format='csc')
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix(dim, dim)

    def curl2ts(self, beta_mode: SphericalHarmonicMode):
        nr, maxnl, m = self.res
        def l2(l): return l * (l + 1)
        if beta_mode.comp == 'pol':
            lb = beta_mode.l
            sh_factor = gaunt_matrix(maxnl, m, lb, return_matrix=False)

            def factor_func1(la, lb, lg): return 0.5 * l2(lg) * (l2(la) + l2(lb) - l2(lg))
            def factor_func2(la, lb, lg): return -0.5 * l2(lb) * (-l2(la) + l2(lb) - l2(lg))

            terms = [('divrdiffr', 'divr', factor_func1),
                     ('divr', 'divrdiffr', factor_func2),
                     ('diffdivr', 'W', factor_func2)]
            return self._compute_block(beta_mode, sh_factor, terms)
        else:
            dim = (maxnl - m) * nr
            return scsp.csc_matrix(dim, dim)

            # blocks = []
            # for lg in range(m, maxnl):
            #     for la in range(m, maxnl):
            #         if gaunt[(lg, la)] == 0:
            #             blocks.append(None)
            #         else:
            #             factor1 = 0.5 * l2(lg) * (l2(la) + l2(lb) - l2(lg))
            #             factor2 = -0.5 * l2(lb) * (-l2(la) + l2(lb) - l2(lg))
            #             radial = sym_divrdiffr(beta_mode.radial_expr, self.r_grid)
            #             weight = scsp.diags(factor1 * self.weight * radial) * gaunt[(lg, la)]
            #             mat = self.W[lg].T @ weight @ self.divrW[la]
            #
            #             radial = sym_divr(beta_mode.radial_expr, self.r_grid)
            #             weight = scsp.diags(factor2 * self.weight * radial) * gaunt[(lg, la)]
            #             mat += self.W[lg].T @ weight @ self.divrdiffrW[la]
            #
            #             radial = sym_diffdivr(beta_mode.radial_expr, self.r_grid)
            #             weight = scsp.diags(factor2 * self.weight * radial) * gaunt[(lg, la)]
            #             mat += self.W[lg].T @ weight @ self.W[la]
            #             blocks.append(mat)
            # return scsp.bmat(np.reshape(np.array(blocks, dtype=object), (maxnl - m, maxnl - m)), format='csc')

    def curl2ss(self, maxnl, m, beta_modes: Union[SphericalHarmonicMode, List[SphericalHarmonicMode]]):
        pass
        # TODO


if __name__ == "__main__":
    nr, maxnl, m = 11, 11, 1
    n_grid = 100
    with Timer("init op"):
        transform = WorlandTransform(nr, maxnl, m, n_grid)
    beta_mode = SphericalHarmonicMode("tor", 1, 0, "2 Sqrt[pi/3] r")
    with Timer("comp op"):
        # op = transform.curl1st(beta_mode)
        op = transform.curl2tt(beta_mode)
        # op = transform.curl2st(beta_mode)
    print(op.diagonal())
    print(np.abs(op-scsp.diags(op.diagonal())).max())
    # plt.spy(op).set_marker('.')
    # plt.show()
