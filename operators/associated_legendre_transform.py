from abc import ABC, abstractmethod
import numpy as np

from operators.polynomials import *
from utils import Timer


class AssociatedLegendreTransformBase(ABC):
    def __init__(self, theta_grid: np.ndarray, *args, **kwargs):
        self.grid = theta_grid

    @abstractmethod
    def _init_operators(self):
        pass


class AssociatedLegendreTransformSingleM(AssociatedLegendreTransformBase):
    """ This is class is only used for from spectral space to physical space for visualisation
     Thus a theta grid is given as an input """
    def __init__(self, maxnl, m, theta_grid):
        super(AssociatedLegendreTransformSingleM, self).__init__(theta_grid)
        self.m = m
        self.maxnl = maxnl
        self._init_operators()

    def _init_operators(self,):
        self.operators = {}
        self.operators['plm'] = Plm(self.m, self.maxnl-1, self.grid)
        self.operators['plmdivsin'] = PlmDivSin(self.m, self.maxnl, self.grid)
        self.operators['dthetaplm'] = DthetaPlm(self.m, self.maxnl-1, self.operators['plm'],
                                                self.operators['plmdivsin'], self.grid)
        self.operators['plmdivsin'] = self.operators['plmdivsin'][:, :-1]


if __name__ == "__main__":
    with Timer("init op"):
        transform = AssociatedLegendreTransformSingleM(41, 1, np.linspace(0, np.pi, 501))
