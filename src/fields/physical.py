""" Class for the physical fields """
from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class PhysicalFieldBase(ABC):
    def __init__(self, data, *args, **kwargs):
        self.data = data


class MeridionalSlice(PhysicalFieldBase):
    def __init__(self, data, m, r, theta):
        super(MeridionalSlice, self).__init__(data)
        self.m = m
        self.grid = {'r': r, 'theta': theta}

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

    def visualise(self, phi=0, coord: str = 'spherical', name: str = '', title=True, **kwargs):
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
