from dataclasses import dataclass, field

from operators.polynomials import *
from utils import Timer


@dataclass
class AssociatedLegendreTransformSingleM(ABC):
    """
    The transforms of associated Legendre functions from spectral space to physical space
    """
    maxnl: int
    m: int
    grid: np.ndarray = field(repr=False)

    def __post_init__(self):
        self._operators = dict()
        self._operators['plm'] = Plm(self.m, self.maxnl - 1, self.grid)
        self._operators['plmdivsin'] = PlmDivSin(self.m, self.maxnl, self.grid)
        self._operators['dthetaplm'] = DthetaPlm(self.m, self.maxnl - 1, self._operators['plm'],
                                                 self._operators['plmdivsin'], self.grid)
        self._operators['plmdivsin'] = self._operators['plmdivsin'][:, :-1]

    @property
    def operators(self):
        return self._operators


if __name__ == "__main__":
    with Timer("init op"):
        transform = AssociatedLegendreTransformSingleM(41, 1, np.linspace(0, np.pi, 501))
