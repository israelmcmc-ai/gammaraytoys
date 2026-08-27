import numpy as np
import pytest
import astropy.units as u
from astropy.constants import G, M_earth

from gammaraytoys.sims import SpacecraftHistory, Earth

MU = (G * M_earth).to_value(u.km**3 / u.s**2)  # gravitational parameter, independent of the code under test


def _small_history():
    """A hand-built, valid 4-row history: 3 real intervals plus a terminator."""
    return SpacecraftHistory(
        time=np.array([0.0, 10.0, 30.0, 50.0]) * u.s,
        orbit_radius=np.array([7000.0, 7200.0, 7500.0, 7600.0]) * u.km,
        orbit_angle=np.array([0.0, 10.0, 20.0, 30.0]) * u.deg,
        attitude=np.array([90.0, 100.0, 110.0, 120.0]) * u.deg,
        uptime=np.array([8.0, 5.0, 20.0, 3.0]) * u.s,
    )


# --------------------------------------------------------------------------
# .ori round trip
# --------------------------------------------------------------------------

def test_ori_round_trip(tmp_path):
    original = _small_history()
    path = tmp_path / 'history.ori'

    original.write(path)
    reloaded = SpacecraftHistory.open(path)

    assert reloaded.nintervals == original.nintervals
    assert reloaded.total_livetime == original.total_livetime

    for a, b in zip(original, reloaded):
        assert a.start_time == b.start_time
        assert a.stop_time == b.stop_time
        assert a.livetime == b.livetime
        assert a.orbit_radius == b.orbit_radius
        assert a.orbit_angle == b.orbit_angle
        assert a.attitude == b.attitude


def test_ori_round_trip_preserves_full_precision_values(tmp_path):
    # Non-round numbers, to make sure write/read isn't merely correct for
    # values that happen to format exactly in decimal.
    rng = np.random.default_rng(1234)
    n = 6
    time = np.sort(rng.uniform(0, 1000, size=n))
    time[0] = 0.0
    radius = rng.uniform(7000.0, 7100.0, size=n)
    angle = rng.uniform(0.0, 360.0, size=n)
    attitude = rng.uniform(0.0, 360.0, size=n)
    dt = np.diff(time)
    uptime = np.zeros(n)
    uptime[:-1] = rng.uniform(0.0, 1.0, size=n - 1) * dt

    original = SpacecraftHistory(
        time=time * u.s, orbit_radius=radius * u.km,
        orbit_angle=angle * u.deg, attitude=attitude * u.deg,
        uptime=uptime * u.s)

    path = tmp_path / 'precise.ori'
    original.write(path)
    reloaded = SpacecraftHistory.open(path)

    for a, b in zip(original, reloaded):
        assert a.orbit_radius.to_value(u.km) == pytest.approx(b.orbit_radius.to_value(u.km), rel=1e-12)
        assert a.livetime.to_value(u.s) == pytest.approx(b.livetime.to_value(u.s), rel=1e-12)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def test_requires_strictly_increasing_timestamps():
    with pytest.raises(ValueError):
        SpacecraftHistory(
            time=np.array([0.0, 10.0, 10.0, 30.0]) * u.s,
            orbit_radius=np.array([7000.0] * 4) * u.km,
            orbit_angle=np.array([0.0, 10.0, 20.0, 30.0]) * u.deg,
            attitude=np.array([0.0, 10.0, 20.0, 30.0]) * u.deg,
            uptime=np.array([1.0, 1.0, 1.0, 0.0]) * u.s)

    with pytest.raises(ValueError):
        SpacecraftHistory(
            time=np.array([0.0, 10.0, 5.0, 30.0]) * u.s,  # non-monotonic
            orbit_radius=np.array([7000.0] * 4) * u.km,
            orbit_angle=np.array([0.0, 10.0, 20.0, 30.0]) * u.deg,
            attitude=np.array([0.0, 10.0, 20.0, 30.0]) * u.deg,
            uptime=np.array([1.0, 1.0, 1.0, 0.0]) * u.s)


def test_requires_at_least_two_rows():
    with pytest.raises(ValueError):
        SpacecraftHistory(
            time=np.array([0.0]) * u.s,
            orbit_radius=np.array([7000.0]) * u.km,
            orbit_angle=np.array([0.0]) * u.deg,
            attitude=np.array([0.0]) * u.deg,
            uptime=np.array([0.0]) * u.s)


def test_livetime_must_be_nonnegative():
    with pytest.raises(ValueError):
        SpacecraftHistory(
            time=np.array([0.0, 10.0, 20.0]) * u.s,
            orbit_radius=np.array([7000.0, 7000.0, 7000.0]) * u.km,
            orbit_angle=np.array([0.0, 0.0, 0.0]) * u.deg,
            attitude=np.array([0.0, 0.0, 0.0]) * u.deg,
            uptime=np.array([-1.0, 5.0, 0.0]) * u.s)


def test_livetime_cannot_exceed_interval_span():
    with pytest.raises(ValueError):
        SpacecraftHistory(
            time=np.array([0.0, 10.0, 20.0]) * u.s,
            orbit_radius=np.array([7000.0, 7000.0, 7000.0]) * u.km,
            orbit_angle=np.array([0.0, 0.0, 0.0]) * u.deg,
            attitude=np.array([0.0, 0.0, 0.0]) * u.deg,
            uptime=np.array([10.1, 5.0, 0.0]) * u.s)  # 10.1 > dt=10


def test_livetime_boundary_values_are_allowed():
    # 0 <= L_i <= dt_i is a closed interval: exactly 0 and exactly dt must
    # both be accepted.
    history = SpacecraftHistory(
        time=np.array([0.0, 10.0, 20.0]) * u.s,
        orbit_radius=np.array([7000.0, 7000.0, 7000.0]) * u.km,
        orbit_angle=np.array([0.0, 0.0, 0.0]) * u.deg,
        attitude=np.array([0.0, 0.0, 0.0]) * u.deg,
        uptime=np.array([0.0, 10.0, 0.0]) * u.s)

    assert history.total_livetime == 10 * u.s


def test_orbit_radius_must_exceed_earth_radius():
    earth = Earth()  # radius ~ 6378.1 km

    with pytest.raises(ValueError):
        SpacecraftHistory(
            time=np.array([0.0, 10.0, 20.0]) * u.s,
            orbit_radius=np.array([6000.0, 7000.0, 7000.0]) * u.km,  # first row sub-surface
            orbit_angle=np.array([0.0, 0.0, 0.0]) * u.deg,
            attitude=np.array([0.0, 0.0, 0.0]) * u.deg,
            uptime=np.array([5.0, 5.0, 0.0]) * u.s,
            earth=earth)

    with pytest.raises(ValueError):
        SpacecraftHistory(
            time=np.array([0.0, 10.0, 20.0]) * u.s,
            orbit_radius=np.array([earth.radius.to_value(u.km)] * 3) * u.km,  # exactly at surface
            orbit_angle=np.array([0.0, 0.0, 0.0]) * u.deg,
            attitude=np.array([0.0, 0.0, 0.0]) * u.deg,
            uptime=np.array([5.0, 5.0, 0.0]) * u.s,
            earth=earth)


def test_from_elliptical_orbit_raises_for_subsurface_perigee():
    # a(1-e) = 6000 km < R_E ~ 6378.1 km: the orbit passes through the planet.
    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=6000 * u.km, eccentricity=0.0)

    # a(1-e) = 10000 * (1-0.5) = 5000 km < R_E: also sub-surface, this time
    # via eccentricity rather than a small semi-major axis.
    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=10000 * u.km, eccentricity=0.5)


def test_from_elliptical_orbit_accepts_perigee_that_just_clears_the_surface():
    earth = Earth()
    r_e = earth.radius.to_value(u.km)
    a = (r_e + 100.0) * u.km  # circular orbit 100 km above the surface

    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=a, eccentricity=0.0, time_step=500 * u.s, earth=earth)

    assert history.nintervals > 0


# --------------------------------------------------------------------------
# Interval semantics (Section 4.2) -- the single most important test here.
# --------------------------------------------------------------------------

def test_interval_semantics_row_i_supplies_pose_and_livetime_for_interval_i(tmp_path):
    # Hand-built 4-row .ori file. Row 3 (the terminator) carries absurd pose
    # and uptime values that must never surface anywhere: not in any
    # interval, not in total_livetime.
    ori_text = (
        "time_s,orbit_radius_km,orbit_angle_deg,attitude_deg,uptime_s\n"
        "0.0,7000.0,0.0,90.0,8.0\n"
        "10.0,7200.0,10.0,100.0,5.0\n"
        "30.0,7500.0,20.0,110.0,20.0\n"
        "50.0,999999.0,999999.0,999999.0,999999.0\n"
    )
    path = tmp_path / 'terminator.ori'
    path.write_text(ori_text)

    history = SpacecraftHistory.open(path)

    # nintervals == nrows - 1
    assert history.nintervals == 3

    # total_livetime excludes the terminator's uptime (known-good: 8+5+20=33 s,
    # not 8+5+20+999999).
    assert history.total_livetime == 33 * u.s

    intervals = list(history)
    assert len(intervals) == 3

    expected = [
        dict(start=0.0, stop=10.0, live=8.0, radius=7000.0, angle=0.0, attitude=90.0),
        dict(start=10.0, stop=30.0, live=5.0, radius=7200.0, angle=10.0, attitude=100.0),
        dict(start=30.0, stop=50.0, live=20.0, radius=7500.0, angle=20.0, attitude=110.0),
    ]

    for interval, exp in zip(intervals, expected):
        assert interval.start_time == exp['start'] * u.s
        assert interval.stop_time == exp['stop'] * u.s
        assert interval.livetime == exp['live'] * u.s
        assert interval.orbit_radius == exp['radius'] * u.km
        assert interval.orbit_angle == exp['angle'] * u.deg
        assert interval.attitude == exp['attitude'] * u.deg

        # None of the terminator's absurd values leak into any interval.
        assert interval.orbit_radius != 999999.0 * u.km
        assert interval.orbit_angle != 999999.0 * u.deg
        assert interval.attitude != 999999.0 * u.deg
        assert interval.livetime != 999999.0 * u.s

    # The last interval's stop_time comes from the terminator row's
    # timestamp, which *is* meaningful (only its pose and uptime are not).
    assert intervals[-1].stop_time == 50.0 * u.s


def test_terminator_row_pose_is_never_read_even_if_physically_absurd(tmp_path):
    # A terminator radius below the Earth's surface must not raise --
    # Section 4.2 explicitly says the reader must not require the
    # terminator's pose to be meaningful.
    ori_text = (
        "time_s,orbit_radius_km,orbit_angle_deg,attitude_deg,uptime_s\n"
        "0.0,7000.0,0.0,0.0,1.0\n"
        "10.0,1.0,0.0,0.0,1.0\n"  # terminator radius well below Earth's surface
    )
    path = tmp_path / 'sub_surface_terminator.ori'
    path.write_text(ori_text)

    history = SpacecraftHistory.open(path)  # must not raise

    assert history.nintervals == 1
    assert history.total_livetime == 1 * u.s


# --------------------------------------------------------------------------
# Circular orbit
# --------------------------------------------------------------------------

def test_circular_orbit_has_constant_radius_and_uniform_angular_rate():
    a = 7000.0 * u.km
    period = 2 * np.pi * np.sqrt(a.to_value(u.km)**3 / MU) * u.s
    n = 200

    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=a, eccentricity=0.0, time_step=(period / n))

    radii = np.array([iv.orbit_radius.to_value(u.km) for iv in history])
    assert radii == pytest.approx(a.to_value(u.km), rel=1e-10)

    # Recompute angular rate directly from the public row data (start_time,
    # orbit_angle of each interval, plus the terminator's stop_time) rather
    # than poking at private attributes.
    start_times = [iv.start_time.to_value(u.s) for iv in history]
    orbit_angles = [iv.orbit_angle.to_value(u.deg) for iv in history]
    stop_time = list(history)[-1].stop_time.to_value(u.s)

    all_times = np.array(start_times + [stop_time])
    # For a circular orbit, theta advances by exactly 360 deg over the full
    # period, so append the known angle at the terminator (t = period) by
    # continuity: theta(period) = theta(0) + 360 deg.
    all_angles = np.array(orbit_angles + [orbit_angles[0] + 360.0])

    dtheta_dt = np.diff(all_angles) / np.diff(all_times)  # deg/s
    expected_rate_deg_s = np.degrees(np.sqrt(MU / a.to_value(u.km)**3))

    assert dtheta_dt == pytest.approx(expected_rate_deg_s, rel=1e-8)
    assert np.std(dtheta_dt) / np.mean(dtheta_dt) < 1e-8


# --------------------------------------------------------------------------
# Eccentric orbit
# --------------------------------------------------------------------------

def test_eccentric_orbit_radius_spans_periapsis_to_apoapsis():
    a_km = 30000.0
    e = 0.7
    a = a_km * u.km
    period = 2 * np.pi * np.sqrt(a_km**3 / MU) * u.s
    n = 1000  # even, so a row lands exactly at t = period / 2 (apoapsis)

    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=a, eccentricity=e, time_step=(period / n))

    radii = np.array([iv.orbit_radius.to_value(u.km) for iv in history])

    expected_periapsis = a_km * (1 - e)
    expected_apoapsis = a_km * (1 + e)

    # Row 0 is exactly at periapsis passage (t = 0 by construction).
    assert radii[0] == pytest.approx(expected_periapsis, rel=1e-9)
    # Row n/2 is exactly at t = period/2, i.e. apoapsis.
    assert radii[n // 2] == pytest.approx(expected_apoapsis, rel=1e-9)

    assert radii.min() == pytest.approx(expected_periapsis, rel=1e-9)
    assert radii.max() == pytest.approx(expected_apoapsis, rel=1e-9)
    assert np.all(radii >= expected_periapsis - 1e-6)
    assert np.all(radii <= expected_apoapsis + 1e-6)


def test_eccentric_orbit_conserves_specific_angular_momentum():
    # r^2 dtheta/dt (Kepler's second law) must be constant and equal to
    # sqrt(mu a (1-e^2)) -- an independent closed form from orbital
    # mechanics, not derived from anything the implementation prints.
    a_km = 30000.0
    e = 0.7
    a = a_km * u.km
    period = 2 * np.pi * np.sqrt(a_km**3 / MU) * u.s
    n = 4000

    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=a, eccentricity=e, time_step=(period / n))

    start_times = np.array([iv.start_time.to_value(u.s) for iv in history])
    radii = np.array([iv.orbit_radius.to_value(u.km) for iv in history])
    angles = np.array([iv.orbit_angle.to_value(u.deg) for iv in history])
    stop_time = list(history)[-1].stop_time.to_value(u.s)

    all_times = np.append(start_times, stop_time)
    all_radii = np.append(radii, radii[0])       # r(period) == r(0)
    all_angles = np.append(angles, angles[0] + 360.0)  # theta(period) == theta(0) + 360 deg

    dt = np.diff(all_times)
    dtheta = np.radians(np.diff(all_angles))
    r_mid = 0.5 * (all_radii[:-1] + all_radii[1:])

    specific_angular_momentum = r_mid**2 * (dtheta / dt)  # km^2/s
    expected = np.sqrt(MU * a_km * (1 - e**2))

    assert np.max(np.abs(specific_angular_momentum - expected)) / expected < 1e-4
    assert np.std(specific_angular_momentum) / np.mean(specific_angular_momentum) < 1e-4


# --------------------------------------------------------------------------
# Period
# --------------------------------------------------------------------------

def test_default_duration_is_one_orbital_period(tmp_path):
    a_km = 30000.0
    e = 0.7
    a = a_km * u.km
    expected_period_s = 2 * np.pi * np.sqrt(a_km**3 / MU)

    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=a, eccentricity=e, time_step=100 * u.s)

    intervals = list(history)
    span_s = intervals[-1].stop_time.to_value(u.s) - intervals[0].start_time.to_value(u.s)

    assert span_s == pytest.approx(expected_period_s, rel=1e-9)

    # Physical self-consistency check, independent of the mu value used
    # internally: over exactly one period the orbital angle must advance by
    # exactly 360 deg, regardless of eccentricity. The terminator row's pose
    # is never read through the public interval API, so recover it from the
    # written .ori file (a public artifact) instead of a private attribute.
    path = tmp_path / 'one_period.ori'
    history.write(path)
    last_line = [line for line in path.read_text().splitlines() if line and not line.startswith('#')][-1]
    terminator_orbit_angle_deg = float(last_line.split(',')[2])

    angle_start_deg = intervals[0].orbit_angle.to_value(u.deg)
    assert terminator_orbit_angle_deg == pytest.approx(angle_start_deg + 360.0, abs=1e-6)
