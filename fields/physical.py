""" Class for the physical fields """
from dataclasses import dataclass
from abc import ABC
from typing import Callable, Dict, Union, List, Literal
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
                  field_name: str = '',
                  xlim=(0, 1),
                  ylim=(0, 1),
                  vmax: List[float] = None,
                  vmin: List[float] = None,
                  **kwargs):
        assert coord in ['spherical', 'cylindrical']
        rr, tt = np.meshgrid(self.grid['r'], self.grid['theta'])
        X2 = rr * np.cos(tt)
        X1 = rr * np.sin(tt)
        field = self.at_phi(phi=phi)
        if coord == 'cylindrical':
            comps = ['s', 'phi', 'z']
            cy_field = {}
            cy_field['s'] = field['r']*np.sin(tt) + field['theta']*np.cos(tt)
            cy_field['phi'] = field['phi']
            cy_field['z'] = field['r']*np.cos(tt) - field['theta']*np.sin(tt)
            field = cy_field
            titles = [fr"${field_name}_s$", fr"${field_name}_\phi$", fr"${field_name}_z$"]
        else:
            comps = ['r', 'theta', 'phi']
            titles = [fr"${field_name}_r$", fr"${field_name}_\theta$", fr"${field_name}_\phi$"]

        if vmax is None:
            vmax = [None, None, None]
        else:
            assert len(vmax) == 3, "Number of vmaxs is not 3."
        if vmin is None:
            vmin = [None, None, None]
        else:
            assert len(vmin) == 3, "Number of vmins is not 3."
        for i in range(3):
            r = np.abs(field[comps[i]]).max()
            if vmax[i] is None and vmin[i] is None:
                vmax[i], vmin[i] = r, -r
            elif vmin[i] is None:
                vmin[i] = -r
            elif vmax[i] is None:
                vmax[i] = r

        visu_components(X1, X2, field, titles=titles, vmax=vmax, vmin=vmin, xlim=xlim, ylim=ylim, **kwargs)

    def visualise_strength(self,
                           title: str = None,
                           vmax: float = None,
                           xlim=(0, 1),
                           ylim=(0, 1),
                           **kwargs):
        rr, tt = np.meshgrid(self.grid['r'], self.grid['theta'])
        X2 = rr * np.cos(tt)
        X1 = rr * np.sin(tt)
        # compute field strength
        field = 0
        for comp in self.data.keys():
            field += np.abs(self.data[comp])**2
        field = np.sqrt(field)

        if vmax is None:
            vmax = field.max()
        visu_component(X1, X2, field, title=title, vmax=vmax, vmin=0, xlim=xlim, ylim=ylim, **kwargs)


@dataclass
class CMBSlice(ABC):
    """
    Physical field of a vector field on the CMB surface for a single m
    Parameters
    -----
    data: Dict {str: ndarray}
        field components

    m: int
        azimuthal wave number

    tg: np.ndarray
        grids in theta

    """
    data: Dict[str, np.ndarray]
    m: int
    tg: np.ndarray

    def __post_init__(self):
        for k, v in self.data.items():
            if len(v.shape) == 1:
                self.data[k] = self.data[k].reshape(-1, 1)
            elif len(v.shape) == 2:
                if v.shape[1] == 1:
                    pass
                elif v.shape[0] == 1:
                    self.data[k] = v.reshape(-1, 1)
                else:
                    raise RuntimeError("The data has to be 1D in theta direction")
            else:
                raise RuntimeError("The data has to be 1D in theta direction")

    def __add__(self, other):
        sum = {}
        for comp in self.data.keys():
            sum[comp] = self.data[comp] + other.data[comp]
        return CMBSlice(sum, self.m, self.tg)

    def at_cmb(self,
                   pg: np.ndarray,
                   phase: float = 0.):
        field = {}
        for k, v in self.data.items():
            field[k] = np.real(np.dot(v, np.exp(1.0j*self.m*pg).reshape(1, -1)) * np.exp(1.0j*phase))
        return field

    def visualise(self,
                  nphi: int,
                  component: str = Literal["r", "theta", "phi"],
                  phase: float = 0.,
                  field_name: str = '',
                  **kwargs):
        pg = np.linspace(-np.pi, np.pi, nphi+1)
        field = self.at_cmb(pg, phase=phase)[component]
        lon, lat = np.meshgrid(pg, np.pi/2-self.tg)

        ax = plt.subplot(111, projection="aitoff")
        vmax, vmin = kwargs.get("vmax", None), kwargs.get("vmin", None)
        if vmax is None and vmin is None:
            r = np.abs(field).max()
            vmax, vmin = r, -r

        s = kwargs.get('s', 16)
        plt.pcolormesh(lon, lat, field, shading='gouraud', vmin=vmin, vmax=vmax, cmap=plt.get_cmap('coolwarm'))
        titles = {"r": fr"${field_name}_r$", "theta": fr"${field_name}_\theta$", "phi": fr"${field_name}_\phi$"}
        ax.set_title(titles[component], fontsize=s)
        plt.colorbar(orientation="horizontal")
        plt.axis('off')


@dataclass
class EquatorialSlice(ABC):
    """
    Physical field of a vector field on the equatorial plane for a single m
    Parameters
    -----

    data: Dict {str: ndarray}
        field components

    m: int
        azimuthal wave number

    rg: np.ndarray
        grids in r

    """
    data: Dict[str, np.ndarray]
    m: int
    rg: np.ndarray

    def __post_init__(self):
        for k, v in self.data.items():
            if len(v.shape) == 1:
                self.data[k] = self.data[k].reshape(1, -1)
            elif len(v.shape) == 2:
                if v.shape[0] == 1:
                    pass
                elif v.shape[1] == 1:
                    self.data[k] = v.reshape(1, -1)
                else:
                    raise RuntimeError("The data has to be 1D in r direction")
            else:
                raise RuntimeError("The data has to be 1D in r direction")

    def __add__(self, other):
        sum = {}
        for comp in self.data.keys():
            sum[comp] = self.data[comp] + other.data[comp]
        return EquatorialSlice(sum, self.m, self.rg)

    def at_equator(self,
                   pg: np.ndarray,
                   phase: float = 0.):
        field = {}
        for k, v in self.data.items():
            field[k] = np.real(np.dot(np.exp(1.0j*self.m*pg).reshape(-1, 1), v) * np.exp(1.0j*phase))
        return field

    def visualise(self,
                  nphi: int,
                  phase: float = 0.,
                  coord: str = 'spherical',
                  field_name: str = '',
                  xlim=(-1, 1),
                  ylim=(-1, 1),
                  vmax: List[float] = None,
                  vmin: List[float] = None,
                  **kwargs):
        assert coord in ['spherical', 'cylindrical']
        pg = np.linspace(0, np.pi*2, nphi+1)
        rr, pp = np.meshgrid(self.rg, pg)
        X2 = rr * np.sin(pp)
        X1 = rr * np.cos(pp)
        field = self.at_equator(pg, phase=phase)
        if coord == 'cylindrical':
            comps = ['s', 'phi', 'z']
            cy_field = {}
            cy_field['s'] = field['r']
            cy_field['phi'] = field['phi']
            cy_field['z'] = -field['theta']
            field = cy_field
            titles = [fr"${field_name}_s$", fr"${field_name}_\phi$", fr"${field_name}_z$"]
        else:
            comps = ['r', 'theta', 'phi']
            titles = [fr"${field_name}_r$", fr"${field_name}_\theta$", fr"${field_name}_\phi$"]

        if vmax is None:
            vmax = [None, None, None]
        else:
            assert len(vmax) == 3, "Number of vmaxs is not 3."
        if vmin is None:
            vmin = [None, None, None]
        else:
            assert len(vmin) == 3, "Number of vmins is not 3."
        for i in range(3):
            r = np.abs(field[comps[i]]).max()
            if vmax[i] is None and vmin[i] is None:
                vmax[i], vmin[i] = r, -r
            elif vmin[i] is None:
                vmin = -r
            elif vmax[i] is None:
                vmax = r

        visu_components(X1, X2, field, titles=titles, vmax=vmax, vmin=vmin, xlim=xlim, ylim=ylim, **kwargs)


def visu_component(X1, X2,
                   field: np.ndarray,
                   title: str = None,
                   vmax=None,
                   vmin=None,
                   **kwargs):
    """
    Visualise one component
    """
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
    s = kwargs.get('s', 16)

    im = ax.pcolormesh(X1, X2, field, shading='gouraud', cmap=plt.get_cmap('coolwarm'),
                       vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(labelsize=s)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(kwargs.get('xlim', None))
    ax.set_ylim(kwargs.get('ylim', None))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if xmin < 0:
        ax.spines['left'].set_visible(False)
        ax.axes.yaxis.set_visible(False)
    if xmax > 0:
        ax.spines['right'].set_visible(False)
    if ymin < 0:
        ax.spines['bottom'].set_visible(False)
        ax.axes.xaxis.set_visible(False)
    if ymax > 0:
        ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=s)
    if title is not None:
        ax.set_title(title, fontsize=s)
    plt.colorbar(im, cax=cax)


def visu_components(X1, X2,
                    field: Dict[str, np.ndarray],
                    titles: List[str] = None,
                    vmax: Union[float, List[float]] = None,
                    vmin: Union[float, List[float]] = None,
                    **kwargs):
    """
    Visualise field components
    """
    assert len(field) == 3, "Not 3 components field."
    if titles is not None:
        assert len(titles) == 3, "Number of titles is not 3."
    if vmax is not None:
        if isinstance(vmax, float):
            vmax = [vmax]*3
        assert len(vmax) == 3, "Number of vmaxs is not 3."
    if vmin is not None:
        if isinstance(vmin, float):
            vmin = [vmin]*3
        assert len(vmin) == 3, "Number of vmins is not 3."

    if 'ax' in kwargs:
        axes = kwargs['ax']
        assert len(axes) == 3, "Number of axes is not 3."
    else:
        fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

    for k, comp in enumerate(field.keys()):
        ax = axes[k]
        visu_component(X1, X2, field[comp], title=titles[k], vmax=vmax[k], vmin=vmin[k], ax=ax, **kwargs)


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
    theta_grid = np.linspace(0, np.pi, 2*ntg)
    worland_transform = WorlandTransform(nr, maxnl, m, None, r_grid)
    legendre_transform = AssociatedLegendreTransformSingleM(maxnl, m, theta_grid)

    sp = SpectralComponentSingleM.from_modes(11, 21, 1, 'tor', [(2, 2, 1), (3, 2, 1), (4, 2, 1)])
    # phy = sp.physical_field(worland_transform, legendre_transform)
    # phy.visualise(phi=np.pi/2, ylim=(-1, 1))

    """ test equatorial slice """
    # phy = sp.equatorial_slice(worland_transform)
    # phy.visualise(nphi=200, coord="cylindrical", field_name='u')
    # plt.show()

    """ test CMB plot """
    phy = sp.cmb_slice(201)
    phy.visualise(201, component="phi")
    plt.show()

    """ test cylindrical average """
    # cyl_phy = phy.to_cyl_coord()
    # sg = np.linspace(0.0, 1, 101)
    # fuphi_g = cylindrical_integration(cyl_phy['phi'], phy.grid['r'], phy.grid['theta'], sg,
    #                                   n=(maxnl+2*nr)//2+8, average=True)
    #
    # def reference(s):
    #     from math import sqrt, pi
    #     return (1 / (1155 * sqrt(21) * pi)) * (2409 * sqrt(2) + 1264 * sqrt(55) - \
    #                                            4 * (17457 * sqrt(2) + 11749 * sqrt(55)) * s ** 2 + 24 * (9416 * sqrt(2) + 10735 * sqrt(55)) * s ** 4 - 256 * (660 * sqrt(2) + 1723 * sqrt(55)) * s ** 6 + 232960 * sqrt(55) * s ** 8)

    # plt.plot(sg, fuphi_g(sg).real)
    # plt.plot(sg, fuphi_g(sg).imag)
    # plt.plot(sg, reference(sg)-fuphi_g(sg).real)
    # plt.show()

    """ test columnarity """
    # with Timer("columnarity"):
    #     # col = phy.columnarity(ns=(maxnl+2*nr)//2+50, nz=maxnl+2*nr+100, integration=True, kind='cubic')
    #     col = phy.columnarity(ns=(maxnl+2*nr)//2+50, nz=maxnl+2*nr+100, integration=False, kind='cubic')
    #     sg = np.linspace(0, 1, 101)
    #     plt.plot(sg, col(sg))
    #     plt.show()
