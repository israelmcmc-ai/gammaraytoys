import numpy as np
import pytest
import astropy.units as u

from gammaraytoys.sims import ZenithPointing, NadirPointing, InertialPointing, SpinPointing


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
