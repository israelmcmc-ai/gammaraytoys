from abc import ABC, abstractmethod
from gammaraytoys.coordinates import Cartesian2D
import numpy as np
import astropy.units as u
from .event import Photon
from .spectrum import MonoenergeticSpectrum
from copy import copy
import matplotlib.pyplot as plt
from histpy import Histogram, Axis

class Source(ABC):

    @property
    @abstractmethod
    def flux(self):
        # Total
        pass

    @property
    @abstractmethod
    def spectrum(self):
        pass

    def diff_flux(self, energy):
        return self._flux * self.spectrum.pdf(energy)

    def integrate_flux(self, lo_energy, hi_energy):
        return self._flux * self.spectrum.integrate(lo_energy, hi_energy)

    def discretize_spectrum(self, axis):

        binned_spec = Histogram(axis,
                                unit = self.flux.unit,
                                contents = self.integrate_flux(axis.lower_bounds,
                                                               axis.upper_bounds)
                                )

        return binned_spec

    def plot_spectrum(self, ax = None, e2 = False,
                      energy_units = None, y_units = None,
                      discretize_axis = None,
                      **kwargs):

        if self.flux is None:
            raise RuntimeError("Set a flux before plotting the spectrum")

        if ax is None:
            fig,ax = plt.subplots()

        if isinstance(self.spectrum, MonoenergeticSpectrum):
            raise RuntimeError("Can't plot monoenergetic spectrum")

        if energy_units is None:
            energy_units = u.MeV
        else:
            energy_units = u.Unit(energy_units)

        if discretize_axis is None:
            energy = np.geomspace(self.spectrum.min_energy, self.spectrum.max_energy, 1000).to(energy_units)
            y = self.diff_flux(energy)
        else:
            discretize_axis = Axis(discretize_axis)
            energy = discretize_axis.centers
            binned_spec = self.discretize_spectrum(discretize_axis)
            y = binned_spec / binned_spec.axis.widths

        if e2:
            if y_units is None:
                y_units = u.Unit(u.erg/u.cm/u.s)
            else:
                y_units = u.Unit(y_units)

            y *= energy**2
            y_label = f'$E^2 dN/dE$ [{y_units}]'
        else:
            if y_units is None:
                y_units = u.Unit(1/u.erg/u.cm/u.s)
            else:
                y_units = u.Unit(y_units)

            y_label = f'$dN/dE$ [{y_units}]'

        y = y.to(y_units)

        if discretize_axis is None:
            ax.plot(energy.value, y.value, **kwargs)
        else:
            y.plot(ax, **kwargs)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(f'Energy [{energy_units}]')

        ax.set_ylabel(y_label)

        return ax

    @abstractmethod
    def random_photon(self, detector):
        pass

class PointSource(Source):

    def __init__(self, offaxis_angle, spectrum,
                 flux = None, flux_pivot = None, pivot_energy = None,
                 chirality = None, chirality_degree = 1):
        """
        chirality_degree [0,1]
        flux needed for normalization
        """

        self._spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree

        self.offaxis_angle = offaxis_angle

        if flux is not None:
            self._flux = flux
        else:
            if flux_pivot is None or pivot_energy is None:
                self._flux = None
            else:
                self._flux = (flux_pivot/spectrum.pdf(pivot_energy)).to(1/u.cm/u.s)

        #cache
        self._plane_origin = None
        self._detector = None

    @property
    def flux(self):
        return self._flux

    @property
    def spectrum(self):
        return self._spectrum

    def random_injection_position(self, detector):

        if detector is not self._detector:
            self._detector = detector
            self._plane_origin, self._throw_parallel = detector.throwing_plane(self.offaxis_angle)

        perp_norm_dist = np.random.uniform(-1,1)
        return  Cartesian2D(self._plane_origin.x + self._throw_parallel.x * perp_norm_dist,
                            self._plane_origin.y + self._throw_parallel.y * perp_norm_dist)

    def random_photon(self, detector):

        chirality = copy(self.chirality)
        if chirality is not None:
            if np.random.uniform() > 0.5 + self.chirality_degree/2:
                # Flip to non-dominant chirality
                chirality *= -1

        return Photon(position = self.random_injection_position(detector),
                      direction = 270*u.deg - self.offaxis_angle,
                      energy = self.spectrum.random_energy(),
                      chirality = chirality)

class IsotropicSource(Source):

    def __init__(self, spectrum, flux = None, chirality = None, chirality_degree = 0):

        self._spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree
        self._flux = flux

    @property
    def flux(self):
        return self._flux

    @property
    def spectrum(self):
        return self._spectrum

    def random_photon(self, detector):

        direction = np.random.uniform(0,360)*u.deg

        ps = PointSource(offaxis_angle = direction,
                         spectrum = self.spectrum,
                         chirality = self.chirality,
                         chirality_degree = self.chirality_degree)

        return ps.random_photon(detector = detector)
