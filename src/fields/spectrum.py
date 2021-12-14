from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import numpy as np
import scipy.sparse as scsp

from operators.worland_transform import WorlandTransform
from operators.associated_legendre_transform import AssociatedLegendreTransformSingleM
from fields.physical import *
from operators.polynomials import *
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

    def cylindrical_integration(self, sg, n_jobs=-1, **kwargs) -> Dict[str, Callable]:
        """ compute cylindrical average from spectrum """
        nr, maxnl, m = self.ordering.nr, self.ordering.maxnl, self.ordering.m
        zorder = nr + maxnl // 2
        zorder += zorder % 2 + 8
        x, w = np.polynomial.legendre.leggauss(zorder)

        if self.component == 'tor':
            def integrate(s):
                zg = x * np.sqrt(1 - s**2)
                rg = np.sqrt(s**2+zg**2)
                tg = np.arccos(zg/rg)
                legendre_transform = AssociatedLegendreTransformSingleM(maxnl, m, tg)

                radial = self._Wtransform(self.spectrum, nr, maxnl, m, rg)
                theta_comp = (1.0j * m * legendre_transform.operators['plmdivsin'] * radial).sum(axis=1)
                phi_comp = (-legendre_transform.operators['dthetaplm'] * radial).sum(axis=1)

                s_comp = np.cos(tg)*theta_comp
                z_comp = -np.sin(tg)*theta_comp
                return [0.5*(s_comp*w).sum(), 0.5*(phi_comp*w).sum(), 0.5*(z_comp*w).sum()]
        if self.component == "pol":
            def integrate(s):
                zg = x * np.sqrt(1 - s**2)
                rg = np.sqrt(s**2+zg**2)
                tg = np.arccos(zg/rg)
                legendre_transform = AssociatedLegendreTransformSingleM(maxnl, m, tg)

                radial1 = self._divrWtransform(self.spectrum, nr, maxnl, m, rg)
                radial2 = self._divrdiffrWtransform(self.spectrum, nr, maxnl, m, rg)
                l_factor = scsp.diags([l * (l + 1) for l in range(m, maxnl)])
                r_comp = (legendre_transform.operators['plm'] * np.array(radial1 @ l_factor)).sum(axis=1)
                theta_comp = (legendre_transform.operators['dthetaplm'] * radial2).sum(axis=1)
                phi_comp = (1.0j * m * legendre_transform.operators['plmdivsin'] * radial2).sum(axis=1)

                s_comp = np.cos(tg) * theta_comp + np.sin(tg)*r_comp
                z_comp = -np.sin(tg) * theta_comp + np.cos(tg)*r_comp
                return [0.5*(s_comp*w).sum(), 0.5*(phi_comp*w).sum(), 0.5*(z_comp*w).sum()]

        from joblib import Parallel, delayed
        from scipy.interpolate import interpolate
        tmp = np.array(Parallel(n_jobs=n_jobs,
                                verbose=kwargs.get('verbose', 0),
                                batch_size=kwargs.get('batch_size', 1))(delayed(integrate)(s) for s in sg))
        kind = kwargs.get('interp_kind', 'cubic')
        return {'s': interpolate.interp1d(sg, tmp[:, 0], kind=kind),
                'phi': interpolate.interp1d(sg, tmp[:, 1], kind=kind),
                'z': interpolate.interp1d(sg, tmp[:, 2], kind=kind)}


    @staticmethod
    def _Wtransform(spectrum, nr, maxnl, m, rg):
        radial = np.full((maxnl-m, rg.shape[0]), fill_value=np.NaN, dtype=np.complex128)
        for l in range(m, maxnl):
            poly = worland(nr, l, rg)
            a, b = (l-m)*nr, (l-m+1)*nr
            radial[l-m, :] = poly @ spectrum[a:b]
        return radial.T

    @staticmethod
    def _divrWtransform(spectrum, nr, maxnl, m, rg):
        radial = np.full((maxnl - m, rg.shape[0]), fill_value=np.NaN, dtype=np.complex128)
        for l in range(m, maxnl):
            poly = divrW(nr, l, rg)
            a, b = (l - m) * nr, (l - m + 1) * nr
            radial[l - m, :] = poly @ spectrum[a:b]
        return radial.T

    @staticmethod
    def _divrdiffrWtransform(spectrum, nr, maxnl, m, rg):
        radial = np.full((maxnl - m, rg.shape[0]), fill_value=np.NaN, dtype=np.complex128)
        for l in range(m, maxnl):
            poly = divrdiffrW(nr, l, rg)
            a, b = (l - m) * nr, (l - m + 1) * nr
            radial[l - m, :] = poly @ spectrum[a:b]
        return radial.T

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


if __name__ == "__main__":
    sp = SpectralComponentSingleM.from_modes((101, 201), 1, 'tor', [(2, 2, 1), (3, 2, 1), (4, 2, 1)])
    sg = np.linspace(0.0, 1, 101)
    with Timer("compute cylindrical integration"):
        u_av = sp.cylindrical_integration(sg, n_jobs=8)

    def reference(s):
        from math import sqrt, pi
        return (1 / (1155 * sqrt(21) * pi)) * (2409 * sqrt(2) + 1264 * sqrt(55) - \
                                               4 * (17457 * sqrt(2) + 11749 * sqrt(55)) * s ** 2 + 24 * (9416 * sqrt(2) + 10735 * sqrt(55)) * s ** 4 - 256 * (660 * sqrt(2) + 1723 * sqrt(55)) * s ** 6 + 232960 * sqrt(55) * s ** 8)

    plt.plot(sg, u_av['phi'](sg).real)
    plt.plot(sg, u_av['phi'](sg).imag)
    plt.plot(sg, reference(sg)-u_av['phi'](sg).real)
    plt.show()
