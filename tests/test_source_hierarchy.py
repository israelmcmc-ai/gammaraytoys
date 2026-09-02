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
import numpy as np
import pytest

from gammaraytoys.sims import (Source, FarFieldSource, NearFieldSource,
                               PointSource, IsotropicSource,
                               MonoenergeticSpectrum, Simulator,
                               SimpleTraditionalReconstructor)


def test_pointsource_is_farfieldsource():
    assert issubclass(PointSource, FarFieldSource)


def test_isotropicsource_is_farfieldsource():
    assert issubclass(IsotropicSource, FarFieldSource)


def test_farfieldsource_is_source():
    assert issubclass(FarFieldSource, Source)


def test_nearfieldsource_is_source():
    assert issubclass(NearFieldSource, Source)


def test_source_cannot_be_instantiated_directly():
    # Source declares normalization/spectrum/random_photon/simulated_rate as
    # abstractmethod: it is a pure interface, never a concrete source.
    with pytest.raises(TypeError):
        Source()


def test_farfieldsource_cannot_be_instantiated_directly():
    # FarFieldSource fills in flux(), normalization and simulated_rate()
    # (the formulas shared by every far-field geometry) but still leaves
    # spectrum and random_photon abstract -- there is no way to draw a
    # photon from a bare FarFieldSource, only from a concrete geometry like
    # PointSource.
    with pytest.raises(TypeError):
        FarFieldSource()


def test_nearfieldsource_cannot_be_instantiated_directly():
    # NearFieldSource fills in normalization (= rate) but leaves spectrum,
    # random_photon, simulated_rate and rate abstract, since acceptance --
    # and the rate itself -- is geometry-specific for every near-field
    # source.
    with pytest.raises(TypeError):
        NearFieldSource()


class _MinimalNearFieldSource(NearFieldSource):
    """
    The smallest possible concrete `NearFieldSource`.

    PR 1 introduces `NearFieldSource` as an abstract base with no concrete
    subclass yet -- `NearPointSource` is added in PR 4. This stand-in fills
    in `simulated_rate` (a fixed rate, ignoring any geometric acceptance --
    the acceptance formulas in plan 5.4 are `NearPointSource`'s job, not
    this base class's) and `random_photon` (delegated to an internal
    on-axis `PointSource`, reusing its throwing-plane logic rather than
    duplicating it) so that a real `Simulator` can mix an instance of this
    class with far-field sources -- see
    `test_mixed_far_and_near_field_sources_in_one_simulator` below, which is
    the entire justification for `simulated_rate()` per plan 5.2.
    """

    def __init__(self, spectrum, rate=None, position=None):
        self._spectrum = spectrum
        self._rate = rate
        self._position = position
        self._point_source = PointSource(offaxis_angle=0 * u.deg, spectrum=spectrum)

    @property
    def spectrum(self):
        return self._spectrum

    def random_photon(self, detector, pose=None, earth=None):
        return self._point_source.random_photon(detector=detector)

    def simulated_rate(self, detector, pose=None):
        return self._rate

    @property
    def rate(self):
        return self._rate

    @property
    def position(self):
        # PR 1 adds `NearFieldSource.position` as an abstract property (used
        # by `NearFieldSource.plot`); not exercised by this hierarchy stub's
        # own tests, so a fixed default is enough to keep it instantiable.
        return self._position


# --- PR 1: mixing a far-field and a near-field source in one Simulator -----
#
# Plan section 5.2: `simulated_rate()` "is what lets the simulator mix
# flux-normalised and rate-normalised sources in one run: it sums rates,
# not fluxes." That is the entire reason `simulated_rate()` exists, but
# until `_MinimalNearFieldSource` had a working `simulated_rate` (above) it
# raised `NotImplementedError` and could never be put into a `Simulator` at
# all, so this path had zero coverage.

def test_mixed_far_and_near_field_sources_in_one_simulator(tracker):
    far_flux = 2e-3 / u.cm / u.s
    far_source = PointSource(offaxis_angle=0 * u.deg,
                             spectrum=MonoenergeticSpectrum(1 * u.MeV),
                             flux=far_flux)

    near_rate = 5.0 / u.s
    near_source = _MinimalNearFieldSource(MonoenergeticSpectrum(2 * u.MeV), rate=near_rate)

    # Computed independently of the Simulator, from the formula in plan 5.2:
    # simulated_rate = flux * throwing_plane_size, for every far-field source.
    far_rate = far_flux * tracker.throwing_plane_size

    sim = Simulator(detector=tracker, sources=[far_source, near_source],
                    reconstructor=SimpleTraditionalReconstructor())

    assert sim.total_simulated_rate.to_value(u.Hz) == pytest.approx(
        (far_rate + near_rate).to_value(u.Hz))

    expected_p = np.array([far_rate.to_value(u.Hz), near_rate.to_value(u.Hz)])
    expected_p /= np.sum(expected_p)
    np.testing.assert_allclose(sim._relative_rate, expected_p)

    # The actual point of this test: run_events must complete over the mix.
    # _MinimalNearFieldSource previously raised NotImplementedError from
    # simulated_rate/random_photon, so a Simulator could not even be built,
    # let alone run, with a near-field source in it.
    events = list(sim.run_events(nsim=50))
    assert len(events) == 50
    assert sim.nsim == 50
