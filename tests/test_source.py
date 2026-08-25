import astropy.units as u
import numpy as np
import pytest

from gammaraytoys.detectors import (MonoenergeticSpectrum, PowerLawSpectrum,
                                    MultiComponentSpectrum)


def test_monoenergetic_scalar_draw():
    spec = MonoenergeticSpectrum(5 * u.MeV)

    assert spec.random_energy() == 5 * u.MeV


def test_monoenergetic_sized_draw():
    # Regression test: random_energy() used to reject the size kwarg,
    # which MultiComponentSpectrum relies on.
    spec = MonoenergeticSpectrum(5 * u.MeV)

    energies = spec.random_energy(size=10)

    assert energies.size == 10
    assert u.allclose(energies, 5 * u.MeV)


@pytest.mark.xfail(reason="Known bug: MonoenergeticSpectrum.cdf() returns 0.5/1.0 "
                          "instead of 0.0/1.0, silently dropping half the flux "
                          "when the spectrum is discretized. Not yet fixed.",
                   strict=True)
def test_monoenergetic_cdf_step():
    spec = MonoenergeticSpectrum(5 * u.MeV)

    assert spec.cdf(4 * u.MeV) == 0
    assert spec.cdf(6 * u.MeV) == 1


def test_powerlaw_pdf_integrates_to_one():
    spec = PowerLawSpectrum(index=-2, min_energy=1 * u.MeV, max_energy=10 * u.MeV)

    assert spec.integrate(spec.min_energy, spec.max_energy) == pytest.approx(1, rel=1e-6)


def test_powerlaw_random_energy_within_bounds():
    spec = PowerLawSpectrum(index=-1.5, min_energy=1 * u.MeV, max_energy=10 * u.MeV)

    energies = spec.random_energy(size=2000)

    assert energies.size == 2000
    assert np.all(energies >= spec.min_energy)
    assert np.all(energies <= spec.max_energy)


def test_multicomponent_ncomponents_is_property():
    spec = MultiComponentSpectrum(MonoenergeticSpectrum(1 * u.MeV),
                                  MonoenergeticSpectrum(2 * u.MeV))

    assert spec.ncomponents == 2


def test_multicomponent_random_energy_draws_from_all_components():
    # Regression test: random_energy() used to index components with the
    # full index array instead of the loop variable, and called the
    # nonexistent np.shuffle.
    spec = MultiComponentSpectrum(MonoenergeticSpectrum(1 * u.MeV),
                                  MonoenergeticSpectrum(2 * u.MeV),
                                  weights=[1, 1])

    energies = spec.random_energy(size=200)

    assert energies.size == 200
    values = set(np.unique(energies.to_value(u.MeV)))
    assert values <= {1.0, 2.0}
    # Both components should show up with a reasonably large sample
    assert values == {1.0, 2.0}


def test_multicomponent_random_energy_mixed_component_types():
    # Regression test: MonoenergeticSpectrum.random_energy() previously
    # didn't accept `size`, so mixing it with another spectrum type raised
    # TypeError as soon as more than one sample was requested.
    mono = MonoenergeticSpectrum(3 * u.MeV)
    powerlaw = PowerLawSpectrum(index=-2, min_energy=1 * u.MeV, max_energy=10 * u.MeV)
    spec = MultiComponentSpectrum(mono, powerlaw, weights=[0.5, 0.5])

    energies = spec.random_energy(size=500)

    assert energies.size == 500
    assert np.all(energies >= 1 * u.MeV)
    assert np.all(energies <= 10 * u.MeV)


def test_multicomponent_pdf_is_weighted_sum():
    mono_energy = 5 * u.MeV
    powerlaw = PowerLawSpectrum(index=-2, min_energy=1 * u.MeV, max_energy=10 * u.MeV)
    spec = MultiComponentSpectrum(powerlaw, powerlaw, weights=[0.25, 0.75])

    energy = 2 * u.MeV
    assert spec.pdf(energy).to_value(powerlaw.pdf(energy).unit) == pytest.approx(
        powerlaw.pdf(energy).to_value(powerlaw.pdf(energy).unit), rel=1e-6)
