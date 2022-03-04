""" Class for the physical fields """
from dataclasses import dataclass
from abc import ABC
from typing import Callable, Dict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import Timer


@dataclass
class MeridionalSlice(ABC):
    """
    Physical field of a vector field on the meridional plane for a single m
    Parameters
    -----

    data: Dict {str: ndarray}
        field components

    m: int
        azimuthal wave number

    r: np.ndarray
        grids in r

    theta: np.ndarray
        grids in theta

    """
    data: Dict[str, np.ndarray]
    m: int
    r: np.ndarray
    theta: np.ndarray

    def __post_init__(self):
        self.grid = {'r': self.r, 'theta': self.theta}

    def __add__(self, other):
        sum = {}
        for comp in self.data.keys():
            sum[comp] = self.data[comp] + other.data[comp]
        return MeridionalSlice(sum, self.m, **self.grid)

    def at_phi(self, phi=0):
        field = {}
        factor = 1 if self.m == 0 else 2
        for comp in self.data.keys():
            # field[comp] = factor * np.real(self.data[comp] * np.exp(1.0j*self.m*phi))
            field[comp] = np.real(self.data[comp] * np.exp(1.0j * self.m * phi))
        return field

    def to_cyl_coord(self):
        rr, tt = np.meshgrid(self.grid['r'], self.grid['theta'])
        cy_field = {}
        cy_field['s'] = self.data['r'] * np.sin(tt) + self.data['theta'] * np.cos(tt)
        cy_field['phi'] = self.data['phi']
        cy_field['z'] = self.data['r'] * np.cos(tt) - self.data['theta'] * np.sin(tt)
        return cy_field

    def _s_quadrature(self, ns):
        """
        quadrature assuming a prefactor of 2*s*sqrt(1-s^2), integrating in s
        """
        from operators.polynomials import worland_grid, worland_weight
        sg = worland_grid(ns)
        weight = np.ones(ns) * worland_weight(ns) * sg * 2 * (1-sg**2)
        return sg, weight

    def geostrophic_flow(self, ns, nz, kind='cubic'):
        """
        Compute the geostrophic component using interpolation.
        Try to use spectrum objects for better accuracy
        """
        cy_field = self.to_cyl_coord()
        sg = np.linspace(0, 1, ns)
        return cylindrical_integration(cy_field['phi'], self.grid['r'], self.grid['theta'], sg, nz,
                                          average=True, kind=kind)

    def columnarity(self, ns, nz,
                    integration=True,
                    sg=None,
                    kind='cubic'):
        """
        Compute columnarity of the field, defined by
            \sqrt( (\int \tilde{us}^2 + \tilde{uphi}^2) / (\int us^2+uphi^2) )
        """
        cy_field = self.to_cyl_coord()
        if integration:
            sg, weight = self._s_quadrature(ns)
        else:
            if sg is None:
                sg = np.linspace(0, 1, ns)
        phi_geo = cylindrical_integration(cy_field['phi'], self.grid['r'], self.grid['theta'], sg, nz,
                                          average=True, kind=kind)
        s_geo = cylindrical_integration(cy_field['s'], self.grid['r'], self.grid['theta'], sg, nz,
                                          average=True, kind=kind)
        square_phis = cylindrical_integration(np.abs(cy_field['phi'])**2+np.abs(cy_field['s'])**2,
                                              self.grid['r'], self.grid['theta'], sg, nz, average=True, kind=kind)
        if integration:
            col = np.sum((np.abs(phi_geo(sg))**2 + np.abs(s_geo(sg))**2) * weight)
            col /= np.sum(square_phis(sg) * weight)
        else:
            def col(s):
                return (np.abs(phi_geo(s))**2 + np.abs(s_geo(s))**2) / square_phis(s)
        return col

    def visualise(self,
                  phi=0,
                  coord: str = 'spherical',
                  name: str = '',
                  title=True,
                  **kwargs):
        assert coord in ['spherical', 'cylindrical']
        rr, tt = np.meshgrid(self.grid['r'], self.grid['theta'])
        X2 = rr * np.cos(tt)
        X1 = rr * np.sin(tt)
        field = self.at_phi(phi=phi)
        if coord == 'cylindrical':
            cy_field = {}
            cy_field['s'] = field['r']*np.sin(tt) + field['theta']*np.cos(tt)
            cy_field['phi'] = field['phi']
            cy_field['z'] = field['r']*np.cos(tt) - field['theta']*np.sin(tt)
            field = cy_field
            titles = [fr"${name}_s$", fr"${name}_\phi$", fr"${name}_z$"]
        else:
            titles = [fr"${name}_r$", fr"${name}_\theta$", fr"${name}_\phi$"]
        if 'ax' in kwargs:
            axes = kwargs['ax']
            assert len(axes) == 3
        else:
            fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
        s = kwargs.get('s', 16)
        delta = kwargs.get('exclude_BL', None)
        j = None if delta is None else np.argmax(self.grid['r'] > 1 - delta)-1
        for k, comp in enumerate(field.keys()):
            ax = axes[k]
            r = np.abs(field[comp]).max() if j is None else np.abs(field[comp][:, :j]).max()
            vmin = kwargs.get('vmin', -r)
            vmax = kwargs.get('vmax', r)
            im = ax.pcolormesh(X1, X2, field[comp], shading='gouraud', cmap=plt.get_cmap('coolwarm'),
                               vmin=vmin, vmax=vmax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.tick_params(labelsize=s)
            ax.set_xlim(kwargs.get('xlim', [0, 1]))
            ax.set_ylim(kwargs.get('ylim', [0, 1]))
            ax.set_aspect('equal', 'box')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=s)
            if title:
                ax.set_title(titles[k], fontsize=s)
            plt.colorbar(im, cax=cax)

    def visualise_strength(self,
                           name: str = '',
                           title=True,
                           **kwargs):
        rr, tt = np.meshgrid(self.grid['r'], self.grid['theta'])
        X2 = rr * np.cos(tt)
        X1 = rr * np.sin(tt)
        # compute field strength
        field = 0
        for comp in self.data.keys():
            field += np.abs(self.data[comp])**2
        field = np.sqrt(field)

        title_ = fr'${name}$'
        if 'ax' in kwargs:
            ax = kwargs['ax']
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
        s = kwargs.get('s', 16)
        delta = kwargs.get('exclude_BL', None)
        j = None if delta is None else np.argmax(self.grid['r'] > 1 - delta) - 1

        r = field.max() if j is None else field[:, :j].max()
        vmin = 0
        vmax = kwargs.get('vmax', r)
        im = ax.pcolormesh(X1, X2, field, shading='gouraud', cmap=plt.get_cmap('coolwarm'),
                           vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.tick_params(labelsize=s)
        ax.set_xlim(kwargs.get('xlim', [0, 1]))
        ax.set_ylim(kwargs.get('ylim', [0, 1]))
        ax.set_aspect('equal', 'box')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=s)
        if title:
            ax.set_title(title_, fontsize=s)
        plt.colorbar(im, cax=cax)


def cylindrical_integration(data, rg, tg, sg, n, average=False, kind='cubic') -> Callable:
    """
    cylindrical integration of data on spherical grids, using interpolation, not exact
    """
    from scipy.interpolate import griddata, interp2d, interpolate
    dtype = data.dtype
    # prepare grids for interpolation
    nr, ntheta, ns = rg.shape[0], tg.shape[0], sg.shape[0]
    values = np.reshape(data, (nr*ntheta, ))
    rr, tt = np.meshgrid(rg, tg)
    ss, zz = rr * np.sin(tt), rr * np.cos(tt)
    points = np.concatenate([ss.reshape(-1, 1), zz.reshape(-1, 1)], axis=1)
    x, w = np.polynomial.legendre.leggauss(n)
    sgrid = np.zeros((sg.shape[0], x.shape[0]))
    zgrid = np.zeros((sg.shape[0], x.shape[0]))
    for i in range(sg.shape[0]):
        sgrid[i, :] = sg[i]
        zgrid[i, :] = x * np.sqrt(1. - sg[i] ** 2)

    # interpolation
    interp_cubic = griddata(points, values, (sgrid, zgrid), method='cubic')
    interp_nearest = griddata(points, values, (sgrid, zgrid), method='nearest')
    idx = np.isnan(interp_cubic)
    interp_cubic[np.isnan(interp_cubic)] = 0.
    interp_boundary = np.zeros(interp_cubic.shape, dtype=dtype)
    interp_boundary[idx] = interp_nearest[idx]
    cyl_data = interp_cubic + interp_boundary
    # integration
    cyl_intg = np.zeros(sg.shape, dtype=dtype)
    for i in range(sg.shape[0]):
        if average:
            cyl_intg[i] = 0.5 * w.dot(cyl_data[i, :])
        else:
            cyl_intg[i] = 0.5 * w.dot(cyl_data[i, :]) * 2*np.sqrt(1-sg[i]**2)

    return interpolate.interp1d(sg, cyl_intg, kind=kind)


if __name__ == "__main__":
    from spectrum import SpectralComponentSingleM
    from operators.worland_transform import WorlandTransform
    from operators.associated_legendre_transform import AssociatedLegendreTransformSingleM
    nr, maxnl, m = 11, 21, 1
    nrg, ntg = 201, 201
    r_grid = np.linspace(0, 1.0, nrg)
    theta_grid = np.linspace(-np.pi/ntg/2, np.pi+np.pi/ntg/2, ntg)
    with Timer("transforms"):
        worland_transform = WorlandTransform(nr, maxnl, m, None, r_grid)
        legendre_transform = AssociatedLegendreTransformSingleM(maxnl, m, theta_grid)

        sp = SpectralComponentSingleM.from_modes((11, 21), 1, 'tor', [(2, 2, 1), (3, 2, 1), (4, 2, 1)])
        phy = sp.physical_field(worland_transform, legendre_transform)

    """ test cylindrical average """
    cyl_phy = phy.to_cyl_coord()
    sg = np.linspace(0.0, 1, 101)
    fuphi_g = cylindrical_integration(cyl_phy['phi'], phy.grid['r'], phy.grid['theta'], sg,
                                      n=(maxnl+2*nr)//2+8, average=True)

    def reference(s):
        from math import sqrt, pi
        return (1 / (1155 * sqrt(21) * pi)) * (2409 * sqrt(2) + 1264 * sqrt(55) - \
                                               4 * (17457 * sqrt(2) + 11749 * sqrt(55)) * s ** 2 + 24 * (9416 * sqrt(2) + 10735 * sqrt(55)) * s ** 4 - 256 * (660 * sqrt(2) + 1723 * sqrt(55)) * s ** 6 + 232960 * sqrt(55) * s ** 8)

    # plt.plot(sg, fuphi_g(sg).real)
    # plt.plot(sg, fuphi_g(sg).imag)
    plt.plot(sg, reference(sg)-fuphi_g(sg).real)
    plt.show()

    """ test columnarity """
    # with Timer("columnarity"):
    #     # col = phy.columnarity(ns=(maxnl+2*nr)//2+50, nz=maxnl+2*nr+100, integration=True, kind='cubic')
    #     col = phy.columnarity(ns=(maxnl+2*nr)//2+50, nz=maxnl+2*nr+100, integration=False, kind='cubic')
    #     sg = np.linspace(0, 1, 101)
    #     plt.plot(sg, col(sg))
    #     plt.show()
