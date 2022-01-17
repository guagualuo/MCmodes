from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import numpy as np
import scipy.sparse as scsp
from numba import njit

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

    @staticmethod
    def _laplacianlWtransform(spectrum, nr, maxnl, m, rg):
        radial = np.full((maxnl - m, rg.shape[0]), fill_value=np.NaN, dtype=np.complex128)
        for l in range(m, maxnl):
            poly = laplacianlW(nr, l, rg)
            a, b = (l - m) * nr, (l - m + 1) * nr
            radial[l - m, :] = poly @ spectrum[a:b]
        return radial.T

    def normalise(self, factor):
        self.spectrum /= factor
        if self.energy_spectrum is not None:
            self.energy_spectrum /= np.abs(factor)**2

    def curl(self):
        """ take curl of the component """
        if self.component == "tor":
            self.component = "pol"
        else:
            nr, maxnl, m = self.ordering.nr, self.ordering.maxnl, self.ordering.m
            for l in range(m, maxnl):
                n_grid = nr + maxnl // 2 + 10
                rg = worland_grid(n_grid)
                weight = scsp.diags(np.ones(n_grid) * worland_weight(n_grid))
                poly = worland(nr, l, rg)
                lapl_poly = laplacianlW(nr, l, rg)
                a, b = (l - m) * nr, (l - m + 1) * nr
                self.spectrum[a:b] = -poly.T @ weight @ lapl_poly.dot(self.spectrum[a:b])
            self.component = "tor"

    def restrict_parity(self, parity):
        """ set the spectrum of certain parity to be zero """
        nr, maxnl, m = self.ordering.nr, self.ordering.maxnl, self.ordering.m
        a_idx, s_idx = parity_idx(nr, maxnl, m)
        if self.component == 'pol':
            idx = {'dp': a_idx, 'qp': s_idx}
        else:
            idx = {'dp': s_idx, 'qp': a_idx}
        self.spectrum[idx[parity]] = 0

    def padding(self, nr, maxnl):
        """ padding to higher resolution """
        c0 = self.spectrum
        nr0, maxnl0, m = self.ordering.nr, self.ordering.maxnl, self.ordering.m
        idx = []
        k = 0
        for l in range(m, maxnl0):
            for n in range(nr):
                if n < nr0:
                    idx.append(k)
                k += 1
        c = np.zeros(nr*(maxnl-m), dtype=c0.dtype)
        c[idx] = c0
        return SpectralComponentSingleM((nr, maxnl), m, self.component, c)


class VectorFieldSingleM:
    """ Class for a vector field at single m """
    def __init__(self, res, m, data: np.ndarray):
        dim = data.shape[0] // 2
        self.components = {"tor": SpectralComponentSingleM(res, m, "tor", data[:dim]),
                           "pol": SpectralComponentSingleM(res, m, "pol", data[dim:])}
        self.energy_spectrum = 0

    @classmethod
    def from_components(cls, tor: SpectralComponentSingleM, pol: SpectralComponentSingleM):
        assert tor.ordering.res == pol.ordering.res and tor.ordering.m == pol.ordering.m
        res = tor.ordering.res
        m = tor.ordering.m
        data = np.concatenate([tor.spectrum, pol.spectrum])
        return cls(res, m, data)

    @classmethod
    def from_parity_spectrum(cls, res, m, data: np.ndarray, parity):
        """ given a spectrum with a certain parity, padding with zeros """
        nr, maxnl = res
        a_idx, s_idx = parity_idx(nr, maxnl, m)
        dim = nr*(maxnl-m)
        coe = np.zeros(2*dim, dtype=data.dtype)
        toridx, polidx = (a_idx, s_idx+dim) if parity in ['QP', 'qp'] else (s_idx, a_idx+dim)
        coe[toridx] = data[:len(toridx)]
        coe[polidx] = data[len(toridx):]
        return cls(res, m, coe)

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

    def curl(self):
        """ transform to curl of the field """
        self.components["tor"].curl()
        self.components["pol"].curl()
        new_pol = self.components["tor"]
        new_tor = self.components["pol"]
        self.components = {"tor": new_tor, "pol": new_pol}

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

    def padding(self, nr, maxnl):
        for comp in self.components.keys():
            self.components[comp] = self.components[comp].padding(nr, maxnl)

    def spectrum(self):
        return np.concatenate([self.components['tor'].spectrum, self.components['pol'].spectrum])

    def cylindrical_average(self, sg, n_jobs=-1, **kwargs) -> Dict[str, Callable]:
        """ compute cylindrical average of cylindrical components and square of them
        (can be used to compute columnarity) """
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
            theta_comp = (1.0j * m * legendre_transform.operators['plmdivsin'] * radial).sum(axis=1)
            phi_comp = (-legendre_transform.operators['dthetaplm'] * radial).sum(axis=1)
            # poloidal
            radial1 = self.components['pol']._divrWtransform(self.components['pol'].spectrum, nr, maxnl, m, rg)
            radial2 = self.components['tor']._divrdiffrWtransform(self.components['pol'].spectrum, nr, maxnl, m, rg)
            l_factor = scsp.diags([l * (l + 1) for l in range(m, maxnl)])
            r_comp = (legendre_transform.operators['plm'] * np.array(radial1 @ l_factor)).sum(axis=1)
            theta_comp += (legendre_transform.operators['dthetaplm'] * radial2).sum(axis=1)
            phi_comp += (1.0j * m * legendre_transform.operators['plmdivsin'] * radial2).sum(axis=1)

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


if __name__ == "__main__":
    # sp = SpectralComponentSingleM.from_modes((101, 201), 1, 'tor', [(2, 2, 1), (3, 2, 1), (4, 2, 1)])
    # sg = np.linspace(0.0, 1, 101)
    # with Timer("compute cylindrical integration"):
    #     u_av = sp.cylindrical_integration(sg, n_jobs=1)
    #
    # def reference(s):
    #     from math import sqrt, pi
    #     return (1 / (1155 * sqrt(21) * pi)) * (2409 * sqrt(2) + 1264 * sqrt(55) - \
    #                                            4 * (17457 * sqrt(2) + 11749 * sqrt(55)) * s ** 2 + 24 * (9416 * sqrt(2) + 10735 * sqrt(55)) * s ** 4 - 256 * (660 * sqrt(2) + 1723 * sqrt(55)) * s ** 6 + 232960 * sqrt(55) * s ** 8)
    #
    # plt.plot(sg, u_av['phi'](sg).real)
    # plt.plot(sg, u_av['phi'](sg).imag)
    # plt.plot(sg, reference(sg)-u_av['phi'](sg).real)
    # plt.show()

    tor_sp = SpectralComponentSingleM.from_modes((21, 41), 1, 'tor', [(1, 1, 1), (2, 1, 1)])
    pol_sp = SpectralComponentSingleM.from_modes((21, 41), 1, 'pol', [(1, 1, 1), (2, 1, 1)])
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
        print(np.max(np.abs(av[k](sg) - reference(sg, k))))
        
    

