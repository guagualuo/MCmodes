from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import scipy.sparse as scsp

from operators.worland_transform import WorlandTransform
from operators.associated_legendre_transform import AssociatedLegendreTransformSingleM
from fields.physical import *
from operators.polynomials import energy_weight_tor, energy_weight_pol
from utils import *


class SpectrumOrderingBase(ABC):
    def __init__(self, res, *args, **kwargs):
        self.res = res

    @abstractmethod
    def index(self, *args):
        pass


class SpectrumOrderingSingleM(SpectrumOrderingBase):
    """ class for spectrum ordering for a single m """
    def __init__(self, res, m):
        super(SpectrumOrderingSingleM, self).__init__(res)
        self.nr = res[0]
        self.maxnl = res[1]
        self.m = m
        self.dim = self.nr * (self.maxnl-self.m)

    def index(self, l, n):
        return (l - self.m) * self.nr + n

    def mode_l(self, l):
        return self.index(l, 0), self.index(l, self.nr-1) + 1


class SpectralComponentBase(ABC):
    """ Base class for a spectral component """
    def __init__(self, *args, **kwargs):
        pass


class SpectralComponentSingleM(SpectralComponentBase):
    """ class for a single wave number m field, spectral data """
    def __init__(self, res, m, component: str, data: np.ndarray):
        super(SpectralComponentSingleM, self).__init__()
        self.ordering = SpectrumOrderingSingleM(res, m)
        assert component in ['tor', 'pol']
        self.component = component
        if self.ordering.dim != data.shape[0]:
            raise RuntimeError("Data shape does not match input resolution")
        else:
            self.spectrum = data
        self.energy_spectrum = None

    @classmethod
    def from_modes(cls, res, m, component: str, modes: List[Tuple]):
        ordering = SpectrumOrderingSingleM(res, m)
        data = np.zeros((ordering.dim,), dtype=np.complex128)
        for l, n, value in modes:
            data[ordering.index(l, n)] = value
        return cls(res, m, component, data)

    @classmethod
    def from_parity_spectrum(cls, res, m, component, data: np.ndarray, parity):
        """ given a spectrum with a certain parity, padding with zeros """
        nr, maxnl = res
        ordering = SpectrumOrderingSingleM(res, m)
        sp = np.zeros((ordering.dim,), dtype=np.complex128)
        a_idx, s_idx = parity_idx(nr, maxnl, m)
        if component == 'pol':
            idx = {'dp': a_idx, 'qp': s_idx}
        else:
            idx = {'dp': s_idx, 'qp': a_idx}
        sp[idx[parity]] = data
        return cls(res, m, component, sp)

    def mode_l(self, l):
        a, b = self.ordering.mode_l(l)
        return self.spectrum[a: b]

    def energy(self):
        nr, maxnl, m = self.ordering.nr, self.ordering.maxnl, self.ordering.m
        factor = 1 if m == 0 else 2
        energy_spectrum = np.zeros(maxnl-m)
        weight = energy_weight_tor if self.component == 'tor' else energy_weight_pol
        for l in range(m, maxnl):
            mat = weight(l, nr-1)
            c = self.mode_l(l)
            energy_spectrum[l-m] = factor*np.real(np.linalg.multi_dot([c.T, mat, c.conj()]))
        self.energy_spectrum = energy_spectrum
        return energy_spectrum.sum()

    def physical_field(self, worland_transform: WorlandTransform,
                       legendre_transform: AssociatedLegendreTransformSingleM):
        m = self.ordering.m
        maxnl = self.ordering.maxnl
        nrg = worland_transform.r_grid.shape[0]
        ntg = legendre_transform.grid.shape[0]
        if self.component == 'tor':
            radial = (worland_transform.operators['W'] @ self.spectrum).reshape(-1, nrg)
            r_comp = np.zeros((ntg, nrg))
            theta_comp = 1.0j * m * legendre_transform.operators['plmdivsin'] @ radial
            phi_comp = -legendre_transform.operators['dthetaplm'] @ radial
        if self.component == 'pol':
            radial1 = (worland_transform.operators['divrW'] @ self.spectrum).reshape(-1, nrg)
            radial2 = (worland_transform.operators['divrdiffrW'] @ self.spectrum).reshape(-1, nrg)
            l_factor = scsp.diags([l * (l + 1) for l in range(m, maxnl)])
            r_comp = legendre_transform.operators['plm'] @ l_factor @ radial1
            theta_comp = legendre_transform.operators['dthetaplm'] @ radial2
            phi_comp = 1.0j * m * legendre_transform.operators['plmdivsin'] @ radial2
        field = {'r': r_comp, 'theta': theta_comp, 'phi': phi_comp}
        return MeridionalSlice(field, m, worland_transform.r_grid, legendre_transform.grid)

    def normalise(self, factor):
        self.spectrum /= factor
        if self.energy_spectrum is not None:
            self.energy_spectrum /= np.abs(factor)**2

    def restrict_parity(self, parity):
        """ set the spectrum of certain parity to be zero """
        nr, maxnl, m = self.ordering.nr, self.ordering.maxnl, self.ordering.m
        a_idx, s_idx = parity_idx(nr, maxnl, m)
        if self.component == 'pol':
            idx = {'dp': a_idx, 'qp': s_idx}
        else:
            idx = {'dp': s_idx, 'qp': a_idx}
        self.spectrum[idx[parity]] = 0


class VectorFieldSingleM:
    """ Class for a vector field at single m """
    def __init__(self, res, m, data: np.ndarray):
        dim = data.shape[0] // 2
        self.components = {"tor": SpectralComponentSingleM(res, m, "tor", data[:dim]),
                           "pol": SpectralComponentSingleM(res, m, "pol", data[dim:])}
        self.energy_spectrum = 0

    def energy(self):
        total_energy = 0
        for comp in self.components.keys():
            total_energy += self.components[comp].energy()
        self.energy_spectrum = self.components["tor"].energy_spectrum + self.components["pol"].energy_spectrum
        return total_energy

    def physical_field(self, worland_transform: WorlandTransform,
                       legendre_transform: AssociatedLegendreTransformSingleM) -> MeridionalSlice:
        return self.components["tor"].physical_field(worland_transform, legendre_transform) + \
            self.components["pol"].physical_field(worland_transform, legendre_transform)

    def normalise(self, factor):
        for comp in self.components.keys():
            component = self.components[comp]
            component.spectrum /= factor
            if component.energy_spectrum is not None:
                component.energy_spectrum /= np.abs(factor)**2
        if self.energy_spectrum is not None:
            self.energy_spectrum /= np.abs(factor)**2

    def restrict_parity(self, parity):
        """ set the spectrum of certain parity to be zero """
        self.components['tor'].restrict_parity(parity)
        self.components["pol"].restrict_parity(parity)

