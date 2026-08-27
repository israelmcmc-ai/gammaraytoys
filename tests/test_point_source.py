import astropy.units as u
import numpy as np
import pytest

from gammaraytoys.sims import (PointSource, IsotropicSource,
                               MonoenergeticSpectrum, PowerLawSpectrum,
                               Photon)


def test_pointsource_flux_from_pivot():
    spec = PowerLawSpectrum(index=-2, min_energy=1 * u.MeV, max_energy=10 * u.MeV)

    flux_pivot = 1e-4 / u.cm / u.s / u.MeV
    pivot_energy = 2 * u.MeV

    source = PointSource(offaxis_angle=0 * u.deg, spectrum=spec,
                         flux_pivot=flux_pivot, pivot_energy=pivot_energy)

    # By construction, diff_flux at the pivot energy should reproduce
    # flux_pivot
    assert source.diff_flux(pivot_energy).to(flux_pivot.unit).value == pytest.approx(
        flux_pivot.value, rel=1e-6)


def test_pointsource_flux_none_without_normalization():
    spec = PowerLawSpectrum(index=-2, min_energy=1 * u.MeV, max_energy=10 * u.MeV)

    source = PointSource(offaxis_angle=0 * u.deg, spectrum=spec)

    assert source.flux is None


def test_pointsource_integrate_flux_matches_spectrum_integral():
    spec = PowerLawSpectrum(index=-2, min_energy=1 * u.MeV, max_energy=10 * u.MeV)
    flux = 5e-5 / u.cm / u.s

    source = PointSource(offaxis_angle=0 * u.deg, spectrum=spec, flux=flux)

    lo, hi = 2 * u.MeV, 5 * u.MeV
    expected = flux * spec.integrate(lo, hi)

    assert source.integrate_flux(lo, hi).to_value(flux.unit) == pytest.approx(
        expected.to_value(flux.unit), rel=1e-6)


def test_pointsource_discretize_spectrum_conserves_flux():
    spec = MonoenergeticSpectrum(5 * u.MeV)
    flux = 1 / u.cm / u.s

    source = PointSource(offaxis_angle=0 * u.deg, spectrum=spec, flux=flux)

    from histpy import Axis
    axis = Axis(np.array([0, 3, 4, 6, 10]) * u.MeV, label='E')

    binned = source.discretize_spectrum(axis)

    assert np.sum(binned.contents).to_value(flux.unit) == pytest.approx(flux.value)


def test_pointsource_random_photon_originates_outside_detector(tracker):
    spec = MonoenergeticSpectrum(1 * u.MeV)
    source = PointSource(offaxis_angle=0 * u.deg, spectrum=spec, flux=1 / u.cm / u.s)

    photon = source.random_photon(tracker)

    assert isinstance(photon, Photon)
    assert photon.energy == 1 * u.MeV

    # The photon should be thrown from a plane tangent to the detector's
    # bounding (surrounding) circle, so it never originates strictly inside it
    center = tracker.surrounding_circle_center
    radius = tracker.surrounding_circle_radius
    dist = np.sqrt((photon.position.x - center.x)**2 + (photon.position.y - center.y)**2)
    assert dist.to_value(radius.unit) >= radius.to_value(radius.unit) - 1e-9


def test_pointsource_random_injection_position_within_throwing_plane(tracker):
    spec = MonoenergeticSpectrum(1 * u.MeV)
    source = PointSource(offaxis_angle=30 * u.deg, spectrum=spec, flux=1 / u.cm / u.s)

    plane_origin, throw_parallel = tracker.throwing_plane(30 * u.deg)

    for _ in range(20):
        pos = source.random_injection_position(tracker)

        # The injection point must lie on the line through plane_origin in
        # the throw_parallel direction, within +/- the surrounding radius
        dx = pos.x - plane_origin.x
        dy = pos.y - plane_origin.y

        # dx/throw_parallel.x should equal dy/throw_parallel.y (colinear),
        # guard division by (near) zero components
        if abs(throw_parallel.x.value) > 1e-9:
            t = (dx / throw_parallel.x).to_value('')
        else:
            t = (dy / throw_parallel.y).to_value('')

        assert -1 <= t <= 1


def test_isotropic_source_random_photon_directions_vary(tracker):
    spec = MonoenergeticSpectrum(1 * u.MeV)
    source = IsotropicSource(spectrum=spec, flux=1 / u.cm / u.s)

    directions = [source.random_photon(tracker).direction.to_value(u.deg) for _ in range(30)]

    # An isotropic source should not always throw photons in the same
    # direction
    assert len(set(np.round(directions, 3))) > 1
def _radial_offset(detector, position, offaxis_angle):
    """Distance from the surrounding-circle centre to `position`, projected onto
    the direction the source sits in. Every photon a source throws must start on
    the plane tangent to the surrounding circle, so this equals the radius."""

    centre = detector.surrounding_circle_center

    return ((position.x - centre.x) * np.sin(offaxis_angle) +
            (position.y - centre.y) * np.cos(offaxis_angle))


def test_injection_plane_follows_a_changed_offaxis_angle(tracker):
    # Regression test: the throwing plane was cached on the detector alone, so
    # moving the source on the sky kept injecting photons from the plane
    # belonging to its original off-axis angle.
    source = PointSource(offaxis_angle=0 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV))

    source.random_injection_position(tracker)

    source.offaxis_angle = 180 * u.deg

    for _ in range(20):
        position = source.random_injection_position(tracker)

        assert _radial_offset(tracker, position, 180 * u.deg).to_value(u.cm) == pytest.approx(
            tracker.surrounding_circle_radius.to_value(u.cm))


def test_isotropic_photons_start_on_the_plane_matching_their_direction(tracker):
    # IsotropicSource reuses one PointSource across photons; position and
    # direction must stay consistent photon by photon.
    source = IsotropicSource(spectrum=MonoenergeticSpectrum(1 * u.MeV))

    radius = tracker.surrounding_circle_radius.to_value(u.cm)

    for _ in range(200):
        photon = source.random_photon(tracker)

        offaxis_angle = 270 * u.deg - photon.direction

        assert _radial_offset(tracker, photon.position, offaxis_angle).to_value(u.cm) == pytest.approx(radius)


def test_isotropic_directions_cover_the_full_circle(tracker):
    source = IsotropicSource(spectrum=MonoenergeticSpectrum(1 * u.MeV))

    directions = u.Quantity([source.random_photon(tracker).direction for _ in range(2000)])

    counts, _ = np.histogram(directions.to_value(u.deg), bins=8, range=(0, 360))

    # 2000 draws over 8 equal bins: 250 +- 16 expected, so 5-sigma is ~+-80
    assert np.all(np.abs(counts - 250) < 80)
