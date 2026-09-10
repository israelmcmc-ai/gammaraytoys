from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np
import astropy.units as u
from scipy.stats.sampling import NumericalInverseHermite

from .config_utils import (_as_mapping, _check_keys, _dispatch_type_name,
                           _format_quantity, _number, _quantity)

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

    @classmethod
    def from_config(cls, config, where = 'spectrum'):
        """
        Build a `Spectrum` from its configuration block.

        ```yaml
        {type: Monoenergetic, energy: 511 keV}
        {type: PowerLaw, index: -2, min_energy: 0.2 MeV, max_energy: 10 MeV}
        {type: MultiComponent, components: [...], weights: [1, 3]}
        ```

        Called on `Spectrum`, the block's `type` chooses the class. Called
        on one of the three concrete classes, that class is what gets
        built: `type` may name it or be left out, and naming a different
        one raises rather than quietly handing back the other class.

        Parameters
        ----------
        config : mapping
            The `spectrum` block.
        where : str
            Label for this block in error messages.

        Returns
        -------
        `Spectrum`

        Raises
        ------
        ValueError
            On an unknown type or key, a missing required key, a bad quantity,
            a non-positive `min_energy`, a `max_energy` that does not exceed
            it, or weights that do not match the components.
        """

        block = _as_mapping(config, where)
        name = _dispatch_type_name(cls, Spectrum, block, where, _SPECTRUM_TYPES,
                                   _SPECTRUM_CLASSES, 'spectrum')

        if name == 'Monoenergetic':
            _check_keys(block, where, ('type', 'energy'), required = ('energy',))

            return MonoenergeticSpectrum(
                energy = _quantity(block, 'energy', where, u.keV, required = True))

        if name == 'PowerLaw':
            keys = ('type', 'index', 'min_energy', 'max_energy')
            _check_keys(block, where, keys, required = keys[1:])

            index = _number(block, 'index', where, required = True)
            min_energy = _quantity(block, 'min_energy', where, u.keV, required = True)
            max_energy = _quantity(block, 'max_energy', where, u.keV, required = True)

            # Both checked here rather than left to the sampler: a non-positive
            # or inverted range reaches `NumericalInverseHermite` as a log of
            # zero or a backwards domain, and what comes back is either a
            # failure from deep inside scipy or -- worse -- a table that
            # silently samples nonsense.
            if min_energy <= 0 * min_energy.unit:
                raise ValueError(
                    f"{where}: key 'min_energy' must be positive, got {min_energy}.")

            if max_energy <= min_energy:
                raise ValueError(
                    f"{where}: key 'max_energy' ({max_energy}) must be greater than "
                    f"'min_energy' ({min_energy}).")

            return PowerLawSpectrum(index = index,
                                    min_energy = min_energy,
                                    max_energy = max_energy)

        keys = ('type', 'components', 'weights')
        _check_keys(block, where, keys, required = ('components',))

        blocks = block['components']

        if (not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes))
                or len(blocks) == 0):
            raise ValueError(
                f"{where}: key 'components' must be a non-empty list of spectrum "
                f"blocks, got {blocks!r}.")

        # `Spectrum`, not `cls`: a component is a spectrum of any type, and
        # inside `MultiComponentSpectrum.from_config` `cls` is the multi.
        components = [Spectrum.from_config(item, f"{where}.components[{i}]")
                      for i, item in enumerate(blocks)]

        weights = _number(block, 'weights', where, minimum = 0, shape = 'any')

        if weights is not None:
            if np.ndim(weights) == 0:
                weights = [weights]
            if len(weights) != len(components):
                raise ValueError(
                    f"{where}: key 'weights' has {len(weights)} entries but there "
                    f"are {len(components)} components.")
            if sum(weights) <= 0:
                raise ValueError(f"{where}: key 'weights' must not sum to zero.")

        return MultiComponentSpectrum(*components, weights = weights)

    def to_config(self):
        """
        Write this spectrum back out as a configuration block.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A block `Spectrum.from_config` reads back into an equal
            spectrum. Note two canonicalizations: `PowerLawSpectrum` holds
            `max_energy` converted to `min_energy`'s unit, and
            `MultiComponentSpectrum` holds its weights normalized to sum to
            one (and equal weights are omitted, since that is the default).

        Raises
        ------
        ValueError
            If this is not one of the three types a configuration can name.
            The three override this method; anything else that subclasses
            `Spectrum` lands here, which is the only honest answer -- a
            configuration has no `type` name for it.
        """

        # One method rather than three overrides, so that the whole written
        # schema for spectra reads top to bottom in one place, next to the
        # `from_config` that reads it back.
        if isinstance(self, MonoenergeticSpectrum):
            return {'type': 'Monoenergetic',
                    'energy': _format_quantity(self.energy)}

        if isinstance(self, PowerLawSpectrum):
            return {'type': 'PowerLaw',
                    'index': float(self.index),
                    'min_energy': _format_quantity(self.min_energy),
                    'max_energy': _format_quantity(self.max_energy)}

        if isinstance(self, MultiComponentSpectrum):
            block = {'type': 'MultiComponent',
                     'components': [component.to_config()
                                    for component in self.components]}

            weights = np.asarray(self.weights, dtype = float)

            if not np.all(weights == weights[0]):
                block['weights'] = [float(weight) for weight in weights]

            return block

        raise ValueError(
            f"{type(self).__name__} is not a spectrum a configuration can "
            f"describe; the types that are: "
            f"{sorted(set(_SPECTRUM_TYPES.values()))}.")


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

        if size is None:
            # One photon, one scalar -- the same shape MonoenergeticSpectrum
            # and PowerLawSpectrum return, and what the simulators expect of
            # any spectrum. The array path below works by grouping the draws
            # by component, and a group of one is still an array of one: it
            # cannot produce a scalar, so a single draw is its own case.
            component = np.random.choice(self.ncomponents, p = self.weights)

            return self.components[component].random_energy()

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


# ---------------------------------------------------------------------------
# What a configuration may call each of these classes.
#
# Both tables sit at the bottom of the file because the second one names the
# classes above: `Spectrum.from_config` looks them up when it runs, long
# after this module has finished importing.
# ---------------------------------------------------------------------------


#: Accepted spellings of every spectrum type, mapped to the canonical one.
#: Both the short name the plan's Section 7 sketch uses and the full class
#: name are accepted; the short one is what is written back out.
_SPECTRUM_TYPES = {'Monoenergetic': 'Monoenergetic',
                   'MonoenergeticSpectrum': 'Monoenergetic',
                   'PowerLaw': 'PowerLaw',
                   'PowerLawSpectrum': 'PowerLaw',
                   'MultiComponent': 'MultiComponent',
                   'MultiComponentSpectrum': 'MultiComponent'}

#: The class each canonical spectrum type builds.
_SPECTRUM_CLASSES = {'Monoenergetic': MonoenergeticSpectrum,
                     'PowerLaw': PowerLawSpectrum,
                     'MultiComponent': MultiComponentSpectrum}
