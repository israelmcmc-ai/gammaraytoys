import numpy as np
import pytest
import astropy.units as u

from gammaraytoys.sims import (ZenithPointing, NadirPointing, InertialPointing,
                                SpinPointing, TargetedPointing, Earth)


def test_zenith_pointing_gives_attitude_equal_to_orbit_angle():
    strategy = ZenithPointing()
    time = np.array([0.0, 100.0, 3600.0]) * u.s
    orbit_radius = np.array([7000.0, 7000.0, 7000.0]) * u.km
    orbit_angle = np.array([0.0, 37.5, 271.3]) * u.deg

    attitude = strategy(time, orbit_radius, orbit_angle)

    assert attitude.to_value(u.deg) == pytest.approx(orbit_angle.to_value(u.deg))


def test_nadir_pointing_gives_attitude_180_deg_from_orbit_angle():
    strategy = NadirPointing()
    time = np.array([0.0, 100.0]) * u.s
    orbit_radius = np.array([7000.0, 7000.0]) * u.km
    orbit_angle = np.array([0.0, 271.3]) * u.deg

    attitude = strategy(time, orbit_radius, orbit_angle)

    assert attitude.to_value(u.deg) == pytest.approx(orbit_angle.to_value(u.deg) + 180.0)


def test_inertial_pointing_is_constant_regardless_of_pose():
    fixed_attitude = 42 * u.deg
    strategy = InertialPointing(fixed_attitude)

    a1 = strategy(0 * u.s, 7000 * u.km, 0 * u.deg)
    a2 = strategy(9999 * u.s, 50000 * u.km, 271 * u.deg)

    assert a1 == fixed_attitude
    assert a2 == fixed_attitude


def test_spin_pointing_is_linear_in_time():
    rate = 0.5 * u.deg / u.s
    initial_attitude = 10 * u.deg
    strategy = SpinPointing(rate, initial_attitude=initial_attitude)

    time = np.array([0.0, 20.0, 200.0]) * u.s
    orbit_radius = np.array([7000.0, 7000.0, 7000.0]) * u.km
    orbit_angle = np.array([0.0, 5.0, 50.0]) * u.deg  # SpinPointing must ignore this

    attitude = strategy(time, orbit_radius, orbit_angle)

    expected_deg = (initial_attitude.to_value(u.deg)
                     + rate.to_value(u.deg / u.s) * time.to_value(u.s))
    assert attitude.to_value(u.deg) == pytest.approx(expected_deg)


def test_spin_pointing_default_initial_attitude_is_zero():
    rate = 1 * u.deg / u.s
    strategy = SpinPointing(rate)

    attitude = strategy(0 * u.s, 7000 * u.km, 0 * u.deg)

    assert attitude == 0 * u.deg


def test_zenith_pointing_wired_through_orbit_generation():
    # Integration check: ZenithPointing is the default observation_strategy
    # for from_elliptical_orbit, so the generated attitude column must equal
    # the orbit_angle column exactly, row for row.
    from gammaraytoys.sims import SpacecraftHistory

    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=7000 * u.km, eccentricity=0.0, time_step=200 * u.s)

    for interval in history:
        assert interval.attitude == interval.orbit_angle


# --- Shape guard, parametrized across every strategy -----------------------
#
# InertialPointing.__call__ used to return a bare scalar regardless of input
# shape, while ZenithPointing, NadirPointing and SpinPointing returned arrays
# matching their input; that mismatch has been fixed. This test pins the
# broadcasting behaviour of all five strategies together so that class of bug
# (any one strategy silently dropping back to a bare scalar for array input)
# cannot come back unnoticed for any of them.

def _make_strategy(name):
    if name == 'zenith':
        return ZenithPointing()
    if name == 'nadir':
        return NadirPointing()
    if name == 'inertial':
        return InertialPointing(30 * u.deg)
    if name == 'spin':
        return SpinPointing(1 * u.deg / u.s)
    if name == 'targeted':
        return TargetedPointing(40 * u.deg, Earth())
    raise ValueError(name)


@pytest.mark.parametrize('name', ['zenith', 'nadir', 'inertial', 'spin', 'targeted'])
def test_strategy_output_shape_matches_array_input(name):
    strategy = _make_strategy(name)
    time = np.array([0.0, 100.0, 3600.0]) * u.s
    orbit_radius = np.array([7000.0, 7000.0, 7000.0]) * u.km
    orbit_angle = np.array([0.0, 37.5, 271.3]) * u.deg

    attitude = strategy(time, orbit_radius, orbit_angle)

    assert attitude.shape == time.shape


@pytest.mark.parametrize('name', ['zenith', 'nadir', 'inertial', 'spin', 'targeted'])
def test_strategy_output_shape_matches_scalar_input(name):
    strategy = _make_strategy(name)

    attitude = strategy(0 * u.s, 7000 * u.km, 0 * u.deg)

    assert attitude.shape == ()


# --- TargetedPointing --------------------------------------------------

def test_targeted_pointing_visible_puts_source_on_axis():
    # While the target is visible, A == sky_angle, so Nu = A - sky_angle == 0
    # (Section 3.4): the source sits exactly on-axis.
    earth = Earth()
    sky_angle = 40 * u.deg
    strategy = TargetedPointing(sky_angle, earth)

    # orbit_angle = 0 deg puts nadir at 180 deg, far from sky_angle = 40 deg
    # (140 deg away, well outside rho ~ 67 deg at 550 km altitude): the
    # target is above the limb, not occulted.
    orbit_radius = earth.radius + 550 * u.km
    orbit_angle = 0 * u.deg

    attitude = strategy(0 * u.s, orbit_radius, orbit_angle)

    assert attitude.to_value(u.deg) == pytest.approx(sky_angle.to_value(u.deg))
    Nu = attitude - sky_angle
    assert Nu.to_value(u.deg) == pytest.approx(0.0)


def test_targeted_pointing_occulted_matches_zenith_pointing():
    # While the target is occulted, A == orbit_angle, exactly ZenithPointing.
    earth = Earth()
    sky_angle = 40 * u.deg
    strategy = TargetedPointing(sky_angle, earth)

    # orbit_angle = sky_angle - 180 deg puts nadir exactly on the target:
    # always occulted (rho > 0).
    orbit_radius = earth.radius + 550 * u.km
    orbit_angle = sky_angle - 180 * u.deg

    attitude = strategy(0 * u.s, orbit_radius, orbit_angle)

    assert attitude.to_value(u.deg) == pytest.approx(orbit_angle.to_value(u.deg))


def test_targeted_pointing_switches_exactly_at_occultation_boundary():
    # The switch must happen precisely at the occultation boundary, rho =
    # arcsin(R_E/r) from nadir -- check just inside and just outside it.
    earth = Earth()
    orbit_radius = earth.radius + 550 * u.km
    rho = earth.angular_radius(orbit_radius)

    orbit_angle = 0 * u.deg
    nadir = orbit_angle + 180 * u.deg

    eps = 1e-6 * u.deg

    # Just inside rho of nadir: occulted -> zenith fallback, A == orbit_angle.
    sky_angle_in = nadir + (rho - eps)
    strategy_in = TargetedPointing(sky_angle_in, earth)
    attitude_in = strategy_in(0 * u.s, orbit_radius, orbit_angle)
    assert attitude_in.to_value(u.deg) == pytest.approx(orbit_angle.to_value(u.deg))

    # Just outside rho of nadir: visible -> tracking, A == sky_angle.
    sky_angle_out = nadir + (rho + eps)
    strategy_out = TargetedPointing(sky_angle_out, earth)
    attitude_out = strategy_out(0 * u.s, orbit_radius, orbit_angle)
    assert attitude_out.to_value(u.deg) == pytest.approx(sky_angle_out.to_value(u.deg))


def test_targeted_pointing_array_input_returns_matching_shape():
    # Guard against repeating the InertialPointing wart: array input must
    # give an array output of the same shape, not a bare scalar.
    earth = Earth()
    sky_angle = 40 * u.deg
    strategy = TargetedPointing(sky_angle, earth)

    time = np.array([0.0, 100.0, 200.0, 300.0]) * u.s
    orbit_radius = np.full(4, (earth.radius + 550 * u.km).to_value(u.km)) * u.km
    # A mix of orbit_angle values that are, and are not, occulting the target.
    orbit_angle = np.array([180.0, sky_angle.to_value(u.deg) - 180.0, 90.0, 270.0]) * u.deg

    attitude = strategy(time, orbit_radius, orbit_angle)

    assert attitude.shape == time.shape
    assert isinstance(attitude, u.Quantity)


def test_targeted_pointing_alternates_regimes_through_orbit_generation():
    # End-to-end through from_elliptical_orbit: over one orbit, the target
    # must alternate between being tracked (A == sky_angle) and occulted
    # (A == orbit_angle, the ZenithPointing fallback).
    from gammaraytoys.sims import SpacecraftHistory

    earth = Earth()
    sky_angle = 40 * u.deg
    semi_major_axis = earth.radius + 550 * u.km
    strategy = TargetedPointing(sky_angle, earth)

    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=semi_major_axis, eccentricity=0.0, time_step=30 * u.s,
        observation_strategy=strategy, earth=earth)

    tracked = []
    for interval in history:
        occulted = earth.is_occulted(sky_angle, interval.orbit_angle, interval.orbit_radius)
        if occulted:
            assert interval.attitude.to_value(u.deg) == pytest.approx(
                interval.orbit_angle.to_value(u.deg))
        else:
            assert interval.attitude.to_value(u.deg) == pytest.approx(
                sky_angle.to_value(u.deg))
        tracked.append(not occulted)

    # Both regimes must actually occur over a full orbit (otherwise this
    # test would pass vacuously).
    assert any(tracked)
    assert not all(tracked)
