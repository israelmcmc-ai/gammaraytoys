"""
Tests for the Source / FarFieldSource / NearFieldSource split introduced in
PR 1 (plan section 5.1):

    Source (ABC)
    |-- FarFieldSource (ABC)
    |   |-- PointSource
    |   `-- IsotropicSource
    `-- NearFieldSource (ABC)

These are pure hierarchy/contract checks -- no physics, no randomness --
so nothing here needs the `_seed_random` fixture beyond what conftest
already applies.
"""

import astropy.units as u
import pytest

from gammaraytoys.sims import (Source, FarFieldSource, NearFieldSource,
                               PointSource, IsotropicSource,
                               MonoenergeticSpectrum)


def test_pointsource_is_farfieldsource():
    assert issubclass(PointSource, FarFieldSource)


def test_isotropicsource_is_farfieldsource():
    assert issubclass(IsotropicSource, FarFieldSource)


def test_farfieldsource_is_source():
    assert issubclass(FarFieldSource, Source)


def test_nearfieldsource_is_source():
    assert issubclass(NearFieldSource, Source)


def test_source_cannot_be_instantiated_directly():
    # Source declares flux/spectrum/random_photon/simulated_rate as
    # abstractmethod: it is a pure interface, never a concrete source.
    with pytest.raises(TypeError):
        Source()


def test_farfieldsource_cannot_be_instantiated_directly():
    # FarFieldSource fills in simulated_rate() (the one formula shared by
    # every far-field geometry) but still leaves spectrum, flux and
    # random_photon abstract -- there is no way to draw a photon from a
    # bare FarFieldSource, only from a concrete geometry like PointSource.
    with pytest.raises(TypeError):
        FarFieldSource()


def test_nearfieldsource_cannot_be_instantiated_directly():
    # NearFieldSource fills in flux (always None) but leaves spectrum,
    # random_photon and simulated_rate abstract, since acceptance is
    # geometry-specific for every near-field source.
    with pytest.raises(TypeError):
        NearFieldSource()


class _MinimalNearFieldSource(NearFieldSource):
    """
    The smallest possible concrete `NearFieldSource`.

    PR 1 introduces `NearFieldSource` as an abstract base with no concrete
    subclass yet -- `NearPointSource` is added in PR 4. This stand-in exists
    only so the test below can check the inherited `flux` property on a
    real instance, without depending on physics that PR 1 does not add.
    """

    def __init__(self, spectrum):
        self._spectrum = spectrum

    @property
    def spectrum(self):
        return self._spectrum

    def random_photon(self, detector, pose=None):
        raise NotImplementedError

    def simulated_rate(self, detector, pose=None):
        raise NotImplementedError


def test_nearfieldsource_flux_is_none():
    # Plan section 5.1: a NearFieldSource is normalized by a rate (1/s),
    # not a flux (1/cm/s) -- a brightness per unit sky length is not
    # meaningful for a source close enough that distance matters. flux is
    # therefore unconditionally None, for every near-field source.
    source = _MinimalNearFieldSource(MonoenergeticSpectrum(1 * u.MeV))
    assert source.flux is None
