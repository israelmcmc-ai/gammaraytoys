from abc import ABC, abstractmethod
import numpy as np
import astropy.units as u
from scipy.stats.sampling import NumericalInverseHermite

class Spectrum(ABC):

    @property
    @abstractmethod
    def min_energy(self):
        pass

    @property
    @abstractmethod
    def max_energy(self):
        pass

    @abstractmethod
    def pdf(self, energy):
        # Normalized to 1
        pass

    @abstractmethod
    def cdf(self, energy):
        # Normalized to 1
        pass

    def integrate(self, lo_energy, hi_energy):
        return self.cdf(hi_energy) - self.cdf(lo_energy)

    @abstractmethod
    def random_energy(self, size = None):
        pass

class MonoenergeticSpectrum(Spectrum):

    def __init__(self, energy):
        self.energy = energy

    @property
    def min_energy(self):
        return 0*u.keV

    @property
    def max_energy(self):
        return np.inf*u.keV

    def pdf(self, energy):
        raise ValueError("Do not use PDF for Mono, only CDF")

    def cdf(self, energy):
        return np.array(energy >= self.energy, dtype = int)

    def random_energy(self, size = None):

        if size is None:
            return self.energy

        return np.full(size, self.energy.value) * self.energy.unit

class PowerLawSpectrum(Spectrum):

    def __init__(self, index, min_energy, max_energy):
        self.index = index
        self._min_energy = min_energy
        self._eunit = min_energy.unit
        self._max_energy = max_energy.to(self._eunit)

        if self.index == -1:
            # Special case
            self._norm = 1/min_energy/np.log(max_energy/min_energy)
        else:
            self._norm = ((1+index)/(max_energy*np.power(max_energy/min_energy, index)-min_energy)).to(1/self._eunit)

        class AuxEnergyPDF:
            pdf = lambda energy: self._pdf(energy)
            cdf = lambda energy: self._cdf(energy)

        self._rvs = NumericalInverseHermite(AuxEnergyPDF,
                                            domain = (self.min_energy.value,
                                                      self.max_energy.value))

    @property
    def min_energy(self):
        return self._min_energy

    @property
    def max_energy(self):
        return self._max_energy

    def _log_pdf(self, log_energy):
        return (self.index * (log_energy - np.log(self.min_energy.value)) + np.log(self._norm.value))/(self.index * (np.log(self.max_energy.value) - np.log(self.min_energy.value)) + 2*np.log(self._norm.value))

    def _pdf(self, energy):
        # in min_energy units
        values = self._norm.value*np.power(energy/self.min_energy.value, self.index)

        if np.ndim(values) == 0:
            if energy > self.max_energy.value or energy < self.min_energy.value:
                values = 0
        else:
            values[energy < self.min_energy.value] = 0
            values[energy > self.max_energy.value] = 0

        return values

    def random_energy(self, size = None):

        return self._rvs.rvs(size) * self.min_energy.unit

    def pdf(self, energy):

        return self._pdf(energy.to_value(self._eunit)) * self._norm.unit

    def _cdf(self, energy):
        if self.index == -1:
            # Special case
            cumm = self._norm*self.min_energy*np.log(energy/self.min_energy.value)
            cumm = cumm.to_value('')
        else:
            cumm = self._norm.value*(energy*np.power(energy/self.min_energy.value, self.index)-self.min_energy.value)/(1+self.index)

        if np.ndim(cumm) == 0:
            if energy < self.min_energy.value:
                cumm = 0
            elif energy > self.max_energy.value:
                cumm = 1
        else:
            cumm[energy < self.min_energy.value] = 0
            cumm[energy > self.max_energy.value] = 1

        return cumm

    def cdf(self, energy):

        return self._cdf(energy.to_value(self._eunit))


class MultiComponentSpectrum(Spectrum):

    def __init__(self, *components, weights = None):

        if weights is None:
            self.weights = np.ones(len(components))
        else:
            self.weights = np.array(weights, dtype = float)

        self.weights /= np.sum(self.weights)

        self.components = components

        self._min_energy = np.min(u.Quantity([c.min_energy for c in components]))
        self._max_energy = np.max(u.Quantity([c.max_energy for c in components]))

    @property
    def ncomponents(self):
        return len(self.components)

    @property
    def min_energy(self):
        return self._min_energy

    @property
    def max_energy(self):
        return self._max_energy

    def random_energy(self, size = None):

        component_idx = np.random.choice(self.ncomponents, size = size, p = self.weights)

        energies = []

        for ncomponent in range(self.ncomponents):

            nsamples = np.sum(component_idx == ncomponent)

            energies.append(self.components[ncomponent].random_energy(size = nsamples))

        energies = u.Quantity(np.concatenate(energies))

        # Undo the grouping by component above -- shuffle indices rather than
        # the array itself, since np.random.shuffle isn't guaranteed to
        # preserve the Quantity subclass/unit in place.
        energies = energies[np.random.permutation(energies.size)]

        return energies

    def pdf(self, energy):

        prob = u.Quantity([w*c.pdf(energy) for c,w in zip(self.components, self.weights)])

        prob = np.sum(prob, axis = None if np.ndim(energy) == 0 else 0)

        return prob

    def cdf(self, energy):

        cdf = [w*c.cdf(energy) for c,w in zip(self.components, self.weights)]

        cdf = np.sum(cdf, axis = None if np.ndim(energy) == 0 else 0)

        return cdf
