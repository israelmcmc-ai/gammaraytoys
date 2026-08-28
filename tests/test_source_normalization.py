"""
Tests for the polymorphic `Source.normalization` introduced on top of the
`FarFieldSource` / `NearFieldSource` split (plan section 5.1): `flux` for a
far-field source, a new abstract `rate` for a near-field source, with
`diff_flux`, `integrate_flux`, `discretize_spectrum` and `plot_spectrum` all
routed through `normalization` instead of hard-coding a flux.

Uses the Agg backend throughout, since every test here touches
`plot_spectrum`.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import astropy.units as u
import numpy as np
import pytest

from gammaraytoys.sims import (FarFieldSource, NearFieldSource,
                               PointSource, PowerLawSpectrum)


# --- Stubs -------------------------------------------------------------

class _FarFieldFluxStub(FarFieldSource):
    """
    Minimal concrete `FarFieldSource` that deliberately never sets a
    `_flux` attribute (its value lives under a differently-named attribute
    instead), to prove `diff_flux`/`integrate_flux` route through the
    `normalization` property (-> `flux()`) rather than a private `_flux`
    name that a concrete subclass might not define.
    """

    def __init__(self, spectrum, flux):
        self._spectrum = spectrum
        self._flux_value = flux

    def flux(self, pose=None):
        return self._flux_value

    @property
    def spectrum(self):
        return self._spectrum

    def random_photon(self, detector, pose=None):
        raise NotImplementedError


class _NearFieldRateStub(NearFieldSource):
    """
    Minimal concrete `NearFieldSource` with a `rate` set, standing in for
    `NearPointSource` (added in PR 4) so `diff_flux`, `integrate_flux`,
    `discretize_spectrum` and `plot_spectrum` can be exercised on a
    near-field source in PR 1, before any concrete near-field geometry
    exists.
    """

    def __init__(self, spectrum, rate):
        self._spectrum = spectrum
        self._rate_value = rate

    @property
    def rate(self):
        return self._rate_value

    @property
    def spectrum(self):
        return self._spectrum

    def random_photon(self, detector, pose=None):
        raise NotImplementedError

    def simulated_rate(self, detector, pose=None):
        raise NotImplementedError


def _plotted_curve(source, e2):
    """
    Call `plot_spectrum` with the default (smooth-curve) path and return the
    plotted (x, y) arrays, as plain floats in whatever units `plot_spectrum`
    chose (energy_units defaults to MeV for x; y is whatever `y_units`
    defaulted to).
    """
    fig, ax = plt.subplots()
    try:
        source.plot_spectrum(ax=ax, e2=e2)
        line = ax.lines[-1]
        return np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
    finally:
        plt.close(fig)


def _assert_default_y_units(source, e2, expected_unit):
    """
    `plot_spectrum` does not return the `y_units` it defaulted to, so the
    only way to pin it down from the outside is to recompute the curve
    independently (via `diff_flux`, the same primitive `plot_spectrum` uses)
    in the *expected* unit and compare it against what was actually
    plotted. If `plot_spectrum` had defaulted to a different unit, the
    plotted values would differ from ours by that unit's conversion factor
    (e.g. MeV/erg ~ 1.6e-6) -- not a coincidental rounding difference -- so
    this pins down the unit, not just the numbers.
    """
    x, y = _plotted_curve(source, e2=e2)
    energy = x * u.MeV

    diff = source.diff_flux(energy)
    if e2:
        expected = (diff * energy**2).to(expected_unit)
    else:
        expected = diff.to(expected_unit)

    np.testing.assert_allclose(y, expected.value, rtol=1e-6)


# --- far-field: normalization is flux ------------------------------------

def test_farfieldsource_normalization_is_flux():
    spec = PowerLawSpectrum(index=-2, min_energy=1 * u.MeV, max_energy=10 * u.MeV)
    flux = 3e-4 / u.cm / u.s
    source = PointSource(offaxis_angle=0 * u.deg, spectrum=spec, flux=flux)

    assert source.normalization is source.flux()


# --- far-field: plot_spectrum default y-units -----------------------------

def test_farfieldsource_plot_spectrum_default_y_units_dNdE():
    spec = PowerLawSpectrum(index=-2, min_energy=1 * u.MeV, max_energy=10 * u.MeV)
    source = PointSource(offaxis_angle=0 * u.deg, spectrum=spec, flux=3e-4 / u.cm / u.s)

    # A far-field source's normalization is a flux, 1/(cm s) -- direct
    # astropy unit equality, not a rendered label string.
    assert u.Unit(source.normalization.unit) == u.Unit('1/(cm s)')

    _assert_default_y_units(source, e2=False, expected_unit=u.Unit('1/(erg cm s)'))


def test_farfieldsource_plot_spectrum_default_y_units_e2():
    spec = PowerLawSpectrum(index=-2, min_energy=1 * u.MeV, max_energy=10 * u.MeV)
    source = PointSource(offaxis_angle=0 * u.deg, spectrum=spec, flux=3e-4 / u.cm / u.s)

    _assert_default_y_units(source, e2=True, expected_unit=u.Unit('erg/(cm s)'))


# --- near-field stub: all four spectrum helpers -----------------------

def test_nearfieldsource_stub_diff_flux_and_integrate_flux():
    spec = PowerLawSpectrum(index=-1.5, min_energy=0.5 * u.MeV, max_energy=20 * u.MeV)
    rate = 12.0 / u.s
    source = _NearFieldRateStub(spec, rate)

    energy = 2 * u.MeV
    expected_diff = rate * spec.pdf(energy)
    assert source.diff_flux(energy).to_value(rate.unit / u.MeV) == pytest.approx(
        expected_diff.to_value(rate.unit / u.MeV), rel=1e-9)

    lo, hi = 1 * u.MeV, 5 * u.MeV
    expected_integral = rate * spec.integrate(lo, hi)
    assert source.integrate_flux(lo, hi).to_value(rate.unit) == pytest.approx(
        expected_integral.to_value(rate.unit), rel=1e-9)


def test_nearfieldsource_stub_discretize_spectrum():
    from histpy import Axis

    spec = PowerLawSpectrum(index=-1.5, min_energy=0.5 * u.MeV, max_energy=20 * u.MeV)
    rate = 12.0 / u.s
    source = _NearFieldRateStub(spec, rate)

    axis = Axis(np.array([0.5, 2, 5, 20]) * u.MeV, label='E')
    binned = source.discretize_spectrum(axis)

    assert np.sum(binned.contents).to_value(rate.unit) == pytest.approx(rate.value, rel=1e-9)


def test_nearfieldsource_stub_plot_spectrum_default_y_units():
    spec = PowerLawSpectrum(index=-1.5, min_energy=0.5 * u.MeV, max_energy=20 * u.MeV)
    source = _NearFieldRateStub(spec, 12.0 / u.s)

    # A near-field source's normalization is a rate, 1/s -- direct astropy
    # unit equality, not a rendered label string.
    assert u.Unit(source.normalization.unit) == u.Unit('1/s')

    _assert_default_y_units(source, e2=False, expected_unit=u.Unit('1/(erg s)'))
    _assert_default_y_units(source, e2=True, expected_unit=u.Unit('erg/s'))


# --- diff_flux/integrate_flux route through normalization, not `_flux` -----

def test_diff_flux_and_integrate_flux_work_without_a_private_flux_attribute():
    spec = PowerLawSpectrum(index=-2, min_energy=1 * u.MeV, max_energy=10 * u.MeV)
    flux = 5e-5 / u.cm / u.s
    source = _FarFieldFluxStub(spec, flux)

    # The whole point of the stub: it never sets `_flux` anywhere.
    assert not hasattr(source, '_flux')

    energy = 3 * u.MeV
    expected_diff = flux * spec.pdf(energy)
    assert source.diff_flux(energy).to_value(flux.unit / u.MeV) == pytest.approx(
        expected_diff.to_value(flux.unit / u.MeV), rel=1e-9)

    lo, hi = 2 * u.MeV, 5 * u.MeV
    expected_integral = flux * spec.integrate(lo, hi)
    assert source.integrate_flux(lo, hi).to_value(flux.unit) == pytest.approx(
        expected_integral.to_value(flux.unit), rel=1e-9)
