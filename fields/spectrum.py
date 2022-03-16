from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Callable

import numpy as np

from operators.worland_transform import WorlandTransform
from operators.associated_legendre_transform import AssociatedLegendreTransformSingleM
from fields.physical import MeridionalSlice, EquatorialSlice
from operators.polynomials import *
from utils import *


class _SpectrumOrderingBase(ABC):
    """
    Base class for indexing
    """

    @abstractmethod
    def index(self, *args):
        pass


@dataclass
class SpectrumOrderingSingleM(_SpectrumOrderingBase):
    """
    class for spectrum ordering for a single m

    Parameters
    -----
    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    """
    nr: int
    maxnl: int
    m: int

    def __post_init__(self):
        self.res = (self.nr, self.maxnl, self.m)
        self._dim = self.nr * (self.maxnl-self.m)

    def index(self, l, n):
        return (l - self.m) * self.nr + n

    def mode_l(self, l):
        return self.index(l, 0), self.index(l, self.nr-1) + 1

    @property
    def dim(self):
        return self._dim


@dataclass
class SpectralComponentSingleM(ABC):
    """
    class for a single wave number m field's spectral data

    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    component: which component, "tor" or "pol"

    data: the spectral coefficients.
    """
    nr: int
    maxnl: int
    m: int
    component: str
    data: np.ndarray = field(repr=False)

    def __post_init__(self):
        self.ordering = SpectrumOrderingSingleM(self.nr, self.maxnl, self.m)
        self.component = self.component.lower()
        assert self.component in ['tor', 'pol'], "field component mush be either 'tor' or 'pol'."
        if self.ordering.dim != self.data.shape[0]:
            raise RuntimeError("Data shape does not match input resolution")
        self.calculate_energy()

    @classmethod
    def from_modes(cls, nr, maxnl, m,
                   component: str,
                   modes: List[Tuple]):
        """
        Spectral construction from a list of modes in form of List[(l, n, coe)]
        """
        ordering = SpectrumOrderingSingleM(nr, maxnl, m)
        data = np.zeros((ordering.dim,), dtype=np.complex128)
        for l, n, value in modes:
            data[ordering.index(l, n)] = value
        return cls(nr, maxnl, m, component, data)

    @classmethod
    def from_parity_spectrum(cls, nr, maxnl, m,
                             component: str,
                             data: np.ndarray,
                             parity: str):
        """
        Spectral construction given spectral coefficients for a given parity
        """
        ordering = SpectrumOrderingSingleM(nr, maxnl, m)
        sp = np.zeros((ordering.dim,), dtype=np.complex128)
        a_idx, s_idx = parity_idx(nr, maxnl, m)
        if component.lower() == 'pol':
            idx = {'dp': a_idx, 'qp': s_idx}
        elif component.lower() == 'tor':
            idx = {'dp': s_idx, 'qp': a_idx}
        else:
            raise RuntimeError(f"Unknown component {component}, must be either 'pol' or 'tor'.")
        sp[idx[parity]] = data
        return cls(nr, maxnl, m, component, sp)

    def mode_l(self, l):
        a, b = self.ordering.mode_l(l)
        return self.spectrum[a: b]

    def calculate_energy(self):
        """
        Compute energy spectrum in l
        """
        nr, maxnl, m = self.nr, self.maxnl, self.m
        factor = 1 if m == 0 else 2
        energy_spectrum = np.zeros(maxnl-m)
        weight = energy_weight_tor if self.component == 'tor' else energy_weight_pol
        for l in range(m, maxnl):
            mat = weight(l, nr-1)
            c = self.mode_l(l)
            energy_spectrum[l-m] = factor*np.real(np.linalg.multi_dot([c.T, mat, c.conj()]))

        self._energy_spectrum = energy_spectrum

    def _physical_field(self,
                        worland_transform: WorlandTransform,
                        legendre_transform: AssociatedLegendreTransformSingleM
                        ) -> Dict[str, np.ndarray]:
        m = self.m
        maxnl = self.maxnl
        nrg = worland_transform.r_grid.shape[0]
        ntg = legendre_transform.grid.shape[0]
        if self.component == 'tor':
            radial = (worland_transform.operators['W'] @ self.spectrum).reshape(-1, nrg)
            r_comp = np.zeros((ntg, nrg))
            theta_comp = 1.0j * m * legendre_transform._operators['plmdivsin'] @ radial
            phi_comp = -legendre_transform._operators['dthetaplm'] @ radial
        elif self.component == 'pol':
            radial1 = (worland_transform.operators['divrW'] @ self.spectrum).reshape(-1, nrg)
            radial2 = (worland_transform.operators['divrdiffrW'] @ self.spectrum).reshape(-1, nrg)
            l_factor = scsp.diags([l * (l + 1) for l in range(m, maxnl)])
            r_comp = legendre_transform._operators['plm'] @ l_factor @ radial1
            theta_comp = legendre_transform._operators['dthetaplm'] @ radial2
            phi_comp = 1.0j * m * legendre_transform._operators['plmdivsin'] @ radial2
        else:
            raise RuntimeError(f"Unknown component {self.component}, must be either 'pol' or 'tor'.")
        return {'r': r_comp, 'theta': theta_comp, 'phi': phi_comp}

    def physical_field(self,
                       worland_transform: WorlandTransform,
                       legendre_transform: AssociatedLegendreTransformSingleM
                       ) -> MeridionalSlice:
        """
        Compute physical fields given the Worland transform and associated Legendre transform.
        Physical grids are set in the transform objects.
        """
        field = self._physical_field(worland_transform, legendre_transform)
        return MeridionalSlice(field, self.m, worland_transform.r_grid, legendre_transform.grid)

    def equatorial_slice(self,
                         worland_transform: WorlandTransform,
                         ) -> EquatorialSlice:
        """
        Compute the field at a equatorial slice.
        """
        legendre_transform = AssociatedLegendreTransformSingleM(self.maxnl, self.m, np.array([np.pi/2]))
        field = self._physical_field(worland_transform, legendre_transform)
        return EquatorialSlice(field, self.m, worland_transform.r_grid)

    def cylindrical_integration(self,
                                sg: np.ndarray,
                                n_jobs=-1,
                                **kwargs
                                ) -> Dict[str, Callable]:
        """
        compute cylindrical average from spectrum

        Parameters
        -----
        sg: np.ndarray
            The grids in s for calculation the cylindrical average

        n_jobs: int
            Number of processors for embarrassingly parallel using joblib.

        kwargs:
            One can set `verbose`, `batch_size` for joblib Parallel
            also can set `interp_kind` for the returned 1d function

        Returns
        -----
            Dict[str, Callable]
            Cylindrical average of s, phi, z components, as 1d interpolation functions

        """
        nr, maxnl, m = self.nr, self.maxnl, self.m
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
                theta_comp = (1.0j * m * legendre_transform._operators['plmdivsin'] * radial).sum(axis=1)
                phi_comp = (-legendre_transform._operators['dthetaplm'] * radial).sum(axis=1)

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
                r_comp = (legendre_transform._operators['plm'] * np.array(radial1 @ l_factor)).sum(axis=1)
                theta_comp = (legendre_transform._operators['dthetaplm'] * radial2).sum(axis=1)
                phi_comp = (1.0j * m * legendre_transform._operators['plmdivsin'] * radial2).sum(axis=1)

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

    @staticmethod
    def _laplacianlWtransform(spectrum, nr, maxnl, m, rg):
        radial = np.full((maxnl - m, rg.shape[0]), fill_value=np.NaN, dtype=np.complex128)
        for l in range(m, maxnl):
            poly = laplacianlW(nr, l, rg)
            a, b = (l - m) * nr, (l - m + 1) * nr
            radial[l - m, :] = poly @ spectrum[a:b]
        return radial.T

    def normalise(self, factor):
        self.data /= factor
        self._energy_spectrum /= np.abs(factor)**2

    def curl(self):
        """
        Take curl of the component
        """
        if self.component == "tor":
            self.component = "pol"
        else:
            nr, maxnl, m = self.nr, self.maxnl, self.m
            for l in range(m, maxnl):
                n_grid = nr + maxnl // 2 + 10
                rg = worland_grid(n_grid)
                weight = scsp.diags(np.ones(n_grid) * worland_weight(n_grid))
                poly = worland(nr, l, rg)
                lapl_poly = laplacianlW(nr, l, rg)
                a, b = (l - m) * nr, (l - m + 1) * nr
                self.spectrum[a:b] = -poly.T @ weight @ lapl_poly.dot(self.spectrum[a:b])
            self.component = "tor"
        return self

    def restrict_parity(self, parity: str):
        """
        Set the spectrum of certain parity to be zero
        """
        parity = parity.lower()
        nr, maxnl, m = self.nr, self.maxnl, self.m
        a_idx, s_idx = parity_idx(nr, maxnl, m)
        if self.component == 'pol':
            idx = {'dp': a_idx, 'qp': s_idx}
        else:
            idx = {'dp': s_idx, 'qp': a_idx}
        self.spectrum[idx[parity]] = 0

    def padding(self, nr, maxnl):
        """
        Pad to higher resolution
        """
        c0 = self.spectrum
        nr0, maxnl0, m = self.nr, self.maxnl, self.m
        idx = []
        k = 0
        for l in range(m, maxnl0):
            for n in range(nr):
                if n < nr0:
                    idx.append(k)
                k += 1
        c = np.zeros(nr*(maxnl-m), dtype=c0.dtype)
        c[idx] = c0
        return SpectralComponentSingleM(nr, maxnl, m, self.component, c)

    @property
    def energy(self):
        return self._energy_spectrum.sum()

    @property
    def energy_spectrum(self):
        return self._energy_spectrum

    @property
    def spectrum(self):
        return self.data


@dataclass
class VectorFieldSingleM:
    """
    Class for a vector field at single m

    nr: number of radial modes, starting from 0

    maxnl: maximal spherical harmonic degree L + 1 = maxnl

    m: azimuthal wave number

    data: spectrum coefficient, first half toroidal, second half poloidal

    """
    nr: int
    maxnl: int
    m: int
    data: np.ndarray = field(repr=False)

    def __post_init__(self):
        dim = self.data.shape[0] // 2
        self.components = {"tor": SpectralComponentSingleM(self.nr, self.maxnl, self.m, "tor", self.data[:dim]),
                           "pol": SpectralComponentSingleM(self.nr, self.maxnl, self.m, "pol", self.data[dim:])}

    @classmethod
    def from_components(cls,
                        tor: SpectralComponentSingleM,
                        pol: SpectralComponentSingleM):
        """
        Construction from toroidal and poloidal components
        """
        assert tor.ordering.res == pol.ordering.res, "Incompatible resolutions for tor and pol components"
        nr, maxnl, m = tor.ordering.res
        data = np.concatenate([tor.spectrum, pol.spectrum])
        return cls(nr, maxnl, m, data)

    @classmethod
    def from_parity_spectrum(cls, nr, maxnl, m,
                             data: np.ndarray,
                             parity: str):
        """
        Construction from spectral coefficients for a given parity
        """
        a_idx, s_idx = parity_idx(nr, maxnl, m)
        dim = nr*(maxnl-m)
        coe = np.zeros(2*dim, dtype=data.dtype)
        toridx, polidx = (a_idx, s_idx+dim) if parity.lower() == 'qp' else (s_idx, a_idx+dim)
        coe[toridx] = data[:len(toridx)]
        coe[polidx] = data[len(toridx):]
        return cls(nr, maxnl, m, coe)

    def physical_field(self,
                       worland_transform: WorlandTransform,
                       legendre_transform: AssociatedLegendreTransformSingleM
                       ) -> MeridionalSlice:
        """
        Compute physical fields given Worland transform and associated Legendre transform
        """
        return self.components["tor"].physical_field(worland_transform, legendre_transform) + \
            self.components["pol"].physical_field(worland_transform, legendre_transform)

    def curl(self):
        """ Transform to curl of the field """
        self.components["tor"].curl()
        self.components["pol"].curl()
        new_pol = self.components["tor"]
        new_tor = self.components["pol"]
        self.components = {"tor": new_tor, "pol": new_pol}
        return self

    def normalise(self, factor: float):
        for comp in self.components.keys():
            self.components[comp].normalise(factor)

    def restrict_parity(self, parity):
        """
        Set the spectrum of certain parity to be zero
        """
        self.components['tor'].restrict_parity(parity)
        self.components["pol"].restrict_parity(parity)

    def padding(self, nr, maxnl):
        """
        Pad zero to a higher resolution
        """
        for comp in self.components.keys():
            self.components[comp] = self.components[comp].padding(nr, maxnl)

    def cylindrical_average(self,
                            sg,
                            n_jobs=-1,
                            **kwargs
                            ) -> Dict[str, Callable]:
        """
        Compute cylindrical average of s, phi, z components and square of them
        (can be used to compute columnarity)
        """
        # first need to integrate u_s**2 + u_phi**2 on cylinder
        ordering = self.components['tor'].ordering
        nr, maxnl, m = ordering.nr, ordering.maxnl, ordering.m
        zorder = 2*nr + maxnl
        zorder += zorder % 2 + 8
        x, w = np.polynomial.legendre.leggauss(zorder)

        def integrate(s):
            zg = x * np.sqrt(1 - s ** 2)
            rg = np.sqrt(s ** 2 + zg ** 2)
            tg = np.arccos(zg / rg)
            legendre_transform = AssociatedLegendreTransformSingleM(maxnl, m, tg)
            # toroidal
            radial = self.components['tor']._Wtransform(self.components['tor'].spectrum, nr, maxnl, m, rg)
            theta_comp = (1.0j * m * legendre_transform._operators['plmdivsin'] * radial).sum(axis=1)
            phi_comp = (-legendre_transform._operators['dthetaplm'] * radial).sum(axis=1)
            # poloidal
            radial1 = self.components['pol']._divrWtransform(self.components['pol'].spectrum, nr, maxnl, m, rg)
            radial2 = self.components['tor']._divrdiffrWtransform(self.components['pol'].spectrum, nr, maxnl, m, rg)
            l_factor = scsp.diags([l * (l + 1) for l in range(m, maxnl)])
            r_comp = (legendre_transform._operators['plm'] * np.array(radial1 @ l_factor)).sum(axis=1)
            theta_comp += (legendre_transform._operators['dthetaplm'] * radial2).sum(axis=1)
            phi_comp += (1.0j * m * legendre_transform._operators['plmdivsin'] * radial2).sum(axis=1)

            s_comp = np.cos(tg) * theta_comp + np.sin(tg) * r_comp
            z_comp = -np.sin(tg) * theta_comp + np.cos(tg) * r_comp
            s_comp2 = np.abs(s_comp)**2
            phi_comp2 = np.abs(phi_comp)**2
            z_comp2 = np.abs(z_comp)**2
            return [0.5 * (s_comp * w).sum(), 0.5 * (phi_comp * w).sum(), 0.5 * (z_comp * w).sum(),
                    0.5 * (s_comp2 * w).sum(), 0.5 * (phi_comp2 * w).sum(), 0.5 * (z_comp2 * w).sum()]

        from joblib import Parallel, delayed
        from scipy.interpolate import interpolate
        tmp = np.array(Parallel(n_jobs=n_jobs,
                                verbose=kwargs.get('verbose', 0),
                                batch_size=kwargs.get('batch_size', 1))(delayed(integrate)(s) for s in sg))
        kind = kwargs.get('interp_kind', 'cubic')
        return {'s': interpolate.interp1d(sg, tmp[:, 0], kind=kind),
                'phi': interpolate.interp1d(sg, tmp[:, 1], kind=kind),
                'z': interpolate.interp1d(sg, tmp[:, 2], kind=kind),
                's_square': interpolate.interp1d(sg, tmp[:, 3], kind=kind),
                'phi_square': interpolate.interp1d(sg, tmp[:, 4], kind=kind),
                'z_square': interpolate.interp1d(sg, tmp[:, 5], kind=kind)
                }

    @property
    def energy(self):
        return self.components['tor'].energy + self.components['pol'].energy

    @property
    def energy_spectrum(self):
        return self.components['tor'].energy_spectrum + self.components['pol'].energy_spectrum

    @property
    def spectrum(self):
        return np.concatenate([self.components['tor'].spectrum, self.components['pol'].spectrum])


if __name__ == "__main__":
    tor_sp = SpectralComponentSingleM.from_modes(21, 41, 1, 'tor', [(1, 1, 1), (2, 1, 1)])
    pol_sp = SpectralComponentSingleM.from_modes(21, 41, 1, 'pol', [(1, 1, 1), (2, 1, 1)])
    sp = VectorFieldSingleM.from_components(tor_sp, pol_sp)
    sg = np.linspace(0.0, 1, 101)
    av = sp.cylindrical_average(sg, n_jobs=1)

    def reference(s, c):
        from math import sqrt, pi, exp
        if c == 's':
            return (sqrt(2)*(14j + 5*sqrt(3) - 2*(19j + 10*sqrt(3))*s**2 + 24j*s**4))/(15*pi)
        elif c == 'phi':
            return (sqrt(2)*(-14 + 5j*sqrt(3) + 16*(8 - 5j*sqrt(3))*s**2 - 144*s**4))/(15*pi)
        elif c == 'z':
            return s*(156 - 5j*sqrt(3) + 8j*(42j + sqrt(3))*s**2)/(3*sqrt(2)*pi)
        elif c == 's_square':
            return (15927 - 56147*s**2 + 80320*s**4 - 40192*s**6 + 512*s**8)/(70*pi**2)
        elif c == 'phi_square':
            return (47781 - 96121*s**2 + 232432*s**4 - 193920*s**6 + 43008*s**8)/(210*pi**2)
        elif c == 'z_square':
            return (299819*s**2 - 1253216*s**4 + 1336320*s**6 - 4608*s**8)/(210*pi**2)

    for k in av.keys():
        print(np.allclose(av[k](sg), reference(sg, k)))
