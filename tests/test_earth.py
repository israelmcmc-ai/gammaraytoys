import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pytest
import astropy.units as u
from astropy.constants import R_earth

from gammaraytoys.sims import Earth


def test_default_radius_matches_astropy_R_earth():
    earth = Earth()

    assert u.isclose(earth.radius, R_earth.to(u.km))


def test_custom_radius_is_kept():
    earth = Earth(radius=1000 * u.km)

    assert earth.radius == 1000 * u.km


@pytest.mark.parametrize('orbit_radius_km', [6500.0, 7000.0, 8000.0, 42164.0])
def test_angular_radius_matches_arcsin_formula(orbit_radius_km):
    # rho = arcsin(R_E / r), Section 3.2/4.5. Computed independently here
    # from the same physical constant the implementation uses (astropy's
    # R_earth), not from anything the implementation prints.
    earth = Earth()
    r_e = earth.radius.to_value(u.km)

    expected_rho_deg = np.degrees(np.arcsin(r_e / orbit_radius_km))

    rho = earth.angular_radius(orbit_radius_km * u.km)

    assert rho.to_value(u.deg) == pytest.approx(expected_rho_deg, rel=1e-10)
    assert rho < 90 * u.deg


def test_angular_radius_raises_at_or_below_surface():
    earth = Earth()

    with pytest.raises(ValueError):
        earth.angular_radius(earth.radius)

    with pytest.raises(ValueError):
        earth.angular_radius(earth.radius / 2)


def test_is_occulted_raises_at_or_below_surface():
    earth = Earth()

    with pytest.raises(ValueError):
        earth.is_occulted(0 * u.deg, 0 * u.deg, earth.radius)


def test_is_occulted_trivial_nadir_and_zenith():
    # A source exactly at nadir (looking straight down, through the planet)
    # must be occulted; one at zenith (straight up, opposite the planet)
    # must never be.
    earth = Earth()
    orbit_angle = 40 * u.deg
    nadir = orbit_angle + 180 * u.deg
    zenith = orbit_angle

    assert earth.is_occulted(nadir, orbit_angle, 7000 * u.km)
    assert not earth.is_occulted(zenith, orbit_angle, 7000 * u.km)


def test_is_occulted_boundary_from_both_sides():
    # Boundary at nadir +/- rho, tested a hair inside and a hair outside on
    # both sides of nadir. rho is computed independently from arcsin(R_E/r),
    # not read off any code output.
    earth = Earth()
    orbit_radius = 7000 * u.km
    orbit_angle = 0 * u.deg
    nadir_deg = 180.0
    rho_deg = np.degrees(np.arcsin(earth.radius.to_value(u.km) / 7000.0))
    eps = 1e-4

    # Just inside the disc on the increasing side.
    assert earth.is_occulted((nadir_deg + rho_deg - eps) * u.deg, orbit_angle, orbit_radius)
    # Just outside the disc on the increasing side.
    assert not earth.is_occulted((nadir_deg + rho_deg + eps) * u.deg, orbit_angle, orbit_radius)
    # Just inside the disc on the decreasing side.
    assert earth.is_occulted((nadir_deg - rho_deg + eps) * u.deg, orbit_angle, orbit_radius)
    # Just outside the disc on the decreasing side.
    assert not earth.is_occulted((nadir_deg - rho_deg - eps) * u.deg, orbit_angle, orbit_radius)


def test_is_occulted_agrees_on_both_representations_of_the_same_direction():
    # 180 deg and -180 deg name the same sky direction; the occultation
    # test must agree regardless of which representation is used, i.e. the
    # +/-180 deg wrap in the implementation's angle difference must not
    # leak into the answer.
    earth = Earth()
    orbit_angle = 0 * u.deg
    orbit_radius = 7000 * u.km

    assert earth.is_occulted(180 * u.deg, orbit_angle, orbit_radius)
    assert earth.is_occulted(-180 * u.deg, orbit_angle, orbit_radius)


def test_is_occulted_boundary_across_the_360_wrap():
    # Choose orbit_angle so that nadir + rho exceeds 360 deg, forcing the
    # occultation boundary itself to sit right on the wrap-around point.
    # rho and the boundary sky angle are both computed independently of the
    # implementation, from the arcsin formula and plain modular arithmetic.
    earth = Earth()
    orbit_radius = 8000 * u.km
    orbit_angle = 170 * u.deg
    nadir_deg = (orbit_angle.to_value(u.deg) + 180.0) % 360.0
    rho_deg = np.degrees(np.arcsin(earth.radius.to_value(u.km) / 8000.0))

    assert nadir_deg + rho_deg > 360.0, "test setup should actually cross the wrap"

    boundary_deg = (nadir_deg + rho_deg) % 360.0
    eps = 1e-4

    just_inside = (boundary_deg - eps) % 360.0
    just_outside = (boundary_deg + eps) % 360.0

    assert earth.is_occulted(just_inside * u.deg, orbit_angle, orbit_radius)
    assert not earth.is_occulted(just_outside * u.deg, orbit_angle, orbit_radius)


def test_is_occulted_array_inputs():
    earth = Earth()
    orbit_angle = 0 * u.deg
    orbit_radius = 7000 * u.km
    sky_angles = np.array([180.0, 0.0, 90.0, -90.0]) * u.deg

    occulted = earth.is_occulted(sky_angles, orbit_angle, orbit_radius)

    np.testing.assert_array_equal(occulted, [True, False, False, False])


# --------------------------------------------------------------------------
# _is_occulted() -- the plain-float fast path
# --------------------------------------------------------------------------

def test_private_is_occulted_accepts_numpy_arrays():
    earth = Earth()
    sky_angle_rad = np.array([np.pi, 0.0, np.pi / 2, -np.pi / 2])
    orbit_angle_rad = 0.0
    orbit_radius_km = 7000.0

    occulted = earth._is_occulted(sky_angle_rad, orbit_angle_rad, orbit_radius_km)

    assert isinstance(occulted, np.ndarray)
    np.testing.assert_array_equal(occulted, [True, False, False, False])


def test_private_is_occulted_does_not_validate_orbit_radius():
    # Documented, deliberate choice: the hot-path float method skips the
    # orbit_radius > radius check for speed. At or below the surface it
    # must not raise; the boundary case below is expected to run through
    # to a (nonsensical, but non-raising) answer rather than error out.
    earth = Earth()
    r_e = earth.radius.value

    # Should not raise, unlike the public path. (arcsin(>1) is undefined
    # and numpy warns and returns nan for it -- that's the documented
    # "caller's job to validate" behaviour, not a crash, so it's silenced
    # here rather than asserted on.)
    with np.errstate(invalid='ignore'):
        earth._is_occulted(0.0, 0.0, r_e)
        earth._is_occulted(0.0, 0.0, r_e / 2)


def test_public_and_private_paths_agree_exactly_across_the_full_circle():
    # Not "close" -- identical booleans, over a wide sweep of sky angles
    # spanning +/-2pi (including both wedge boundaries approached from
    # either side) at several orbital radii.
    earth = Earth()
    orbit_angle = 25 * u.deg
    orbit_angle_rad = orbit_angle.to_value(u.rad)

    rng = np.random.default_rng(0)
    sky_angles_deg = rng.uniform(-720.0, 720.0, size=500)

    orbit_radii_km = [6500.0, 6800.0, 7000.0, 8000.0, 15000.0, 42164.0]

    for orbit_radius_km in orbit_radii_km:
        rho_deg = np.degrees(np.arcsin(earth.radius.to_value(u.km) / orbit_radius_km))
        nadir_deg = orbit_angle.to_value(u.deg) + 180.0
        eps = 1e-4

        # Both wedge boundaries, approached from either side, on top of
        # the random sweep.
        boundary_deg = np.concatenate([
            sky_angles_deg,
            nadir_deg + rho_deg + np.array([-eps, eps]),
            nadir_deg - rho_deg + np.array([-eps, eps]),
        ])

        sky_angle = boundary_deg * u.deg
        orbit_radius = orbit_radius_km * u.km

        public = earth.is_occulted(sky_angle, orbit_angle, orbit_radius)
        private = earth._is_occulted(
            sky_angle.to_value(u.rad), orbit_angle_rad, orbit_radius_km)

        np.testing.assert_array_equal(public, private)


# --------------------------------------------------------------------------
# plot()
# --------------------------------------------------------------------------

def test_plot_returns_axes_and_draws_a_filled_disc():
    earth = Earth(radius=6371 * u.km)

    ax = earth.plot()

    assert ax is not None
    # fill() adds a filled Polygon patch; a bare, un-plotted Axes has none.
    assert len(ax.patches) >= 1

    # Independent sanity bound on the axis limits: nothing this method draws
    # should extend past the Earth's own radius by more than a small margin
    # (matplotlib's default autoscale padding is 5%).
    r = earth.radius.to_value(u.km)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert max(abs(v) for v in xlim + ylim) < 1.5 * r

    plt.close(ax.figure)


def test_plot_draws_into_the_given_axes_and_returns_it():
    fig, ax = plt.subplots()
    earth = Earth(radius=2000 * u.km)

    returned = earth.plot(ax=ax)

    assert returned is ax
    assert len(ax.patches) >= 1

    plt.close(fig)
