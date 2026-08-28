import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pytest
import astropy.units as u
from astropy.constants import G, M_earth

from gammaraytoys.sims import SpacecraftHistory, SpacecraftInterval, Earth
from gammaraytoys.sims.spacecraft_history import _solve_kepler_equation

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


def test_open_raises_on_header_only_file(tmp_path):
    # 0 data rows: even the "at least 2 rows" message must fire, not some
    # unrelated crash inside pandas/astropy.
    path = tmp_path / 'header_only.ori'
    path.write_text('time_s,orbit_radius_km,orbit_angle_deg,attitude_deg,uptime_s\n')

    with pytest.raises(ValueError, match='row'):
        SpacecraftHistory.open(path)


def test_open_raises_naming_a_missing_column(tmp_path):
    path = tmp_path / 'missing_column.ori'
    path.write_text(
        "time_s,orbit_radius_km,orbit_angle_deg,attitude_deg\n"  # uptime_s absent
        "0.0,7000.0,0.0,90.0\n"
        "10.0,7000.0,10.0,100.0\n")

    with pytest.raises(ValueError, match='uptime_s'):
        SpacecraftHistory.open(path)


def test_open_raises_naming_a_misspelled_column(tmp_path):
    path = tmp_path / 'misspelled_column.ori'
    path.write_text(
        # attitude_deg misspelled as atttitude_deg
        "time_s,orbit_radius_km,orbit_angle_deg,atttitude_deg,uptime_s\n"
        "0.0,7000.0,0.0,90.0,5.0\n"
        "10.0,7000.0,10.0,100.0,5.0\n")

    with pytest.raises(ValueError, match='attitude_deg'):
        SpacecraftHistory.open(path)


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


def test_requires_all_row_arrays_to_have_matching_length():
    with pytest.raises(ValueError, match='same length'):
        SpacecraftHistory(
            time=np.array([0.0, 10.0, 20.0]) * u.s,
            orbit_radius=np.array([7000.0, 7000.0]) * u.km,  # one row short
            orbit_angle=np.array([0.0, 10.0, 20.0]) * u.deg,
            attitude=np.array([0.0, 10.0, 20.0]) * u.deg,
            uptime=np.array([5.0, 5.0, 0.0]) * u.s)


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


@pytest.mark.parametrize('bad_column, ori_column', [
    ('orbit_radius', 'orbit_radius_km'),
    ('orbit_angle', 'orbit_angle_deg'),
    ('attitude', 'attitude_deg'),
    ('uptime', 'uptime_s'),
])
def test_validate_raises_on_non_finite_interval_value(bad_column, ori_column):
    # Each of the four interval columns is checked independently (Section
    # 4.2's NaN note); put a NaN in interval row 0 (not the terminator) of
    # each column in turn and check it is caught, and named, on its own.
    kwargs = dict(
        time=np.array([0.0, 10.0, 20.0]) * u.s,
        orbit_radius=np.array([7000.0, 7000.0, 7000.0]) * u.km,
        orbit_angle=np.array([0.0, 10.0, 20.0]) * u.deg,
        attitude=np.array([0.0, 10.0, 20.0]) * u.deg,
        uptime=np.array([5.0, 5.0, 0.0]) * u.s,
    )
    values = kwargs[bad_column]
    tainted = values.value.copy()
    tainted[0] = np.nan
    kwargs[bad_column] = tainted * values.unit

    with pytest.raises(ValueError, match=ori_column):
        SpacecraftHistory(**kwargs)


def test_terminator_row_may_contain_non_finite_values():
    # Section 4.2: the terminator's pose and uptime are never read, so a
    # NaN there must not raise -- this is the documented exemption from the
    # non-finite check above.
    history = SpacecraftHistory(
        time=np.array([0.0, 10.0, 20.0]) * u.s,
        orbit_radius=np.array([7000.0, 7000.0, np.nan]) * u.km,
        orbit_angle=np.array([0.0, 10.0, np.nan]) * u.deg,
        attitude=np.array([0.0, 10.0, np.nan]) * u.deg,
        uptime=np.array([5.0, 5.0, np.nan]) * u.s)  # must not raise

    assert history.nintervals == 2
    assert history.total_livetime == 10 * u.s


def test_terminator_row_may_contain_the_999999_placeholder():
    # The 999999 sentinel used by the example .ori file in Section 4.1 must
    # be just as acceptable in the terminator row as a NaN.
    history = SpacecraftHistory(
        time=np.array([0.0, 10.0, 20.0]) * u.s,
        orbit_radius=np.array([7000.0, 7000.0, 999999.0]) * u.km,
        orbit_angle=np.array([0.0, 10.0, 999999.0]) * u.deg,
        attitude=np.array([0.0, 10.0, 999999.0]) * u.deg,
        uptime=np.array([5.0, 5.0, 999999.0]) * u.s)

    assert history.nintervals == 2
    assert history.total_livetime == 10 * u.s


def test_from_elliptical_orbit_raises_for_subsurface_perigee():
    # a(1-e) = 6000 km < R_E ~ 6378.1 km: the orbit passes through the planet.
    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=6000 * u.km, eccentricity=0.0)

    # a(1-e) = 10000 * (1-0.5) = 5000 km < R_E: also sub-surface, this time
    # via eccentricity rather than a small semi-major axis.
    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=10000 * u.km, eccentricity=0.5)


def test_from_elliptical_orbit_rejects_eccentricity_out_of_range():
    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=7000 * u.km, eccentricity=-0.1)

    with pytest.raises(ValueError):  # e == 1 is parabolic, excluded by the half-open [0, 1)
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=7000 * u.km, eccentricity=1.0)


def test_from_elliptical_orbit_rejects_livetime_fraction_out_of_range():
    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=7000 * u.km, livetime_fraction=-0.1)

    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=7000 * u.km, livetime_fraction=1.1)


def test_from_elliptical_orbit_rejects_non_positive_duration():
    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=7000 * u.km, duration=0 * u.s)

    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=7000 * u.km, duration=-10 * u.s)


def test_from_elliptical_orbit_rejects_non_positive_time_step():
    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=7000 * u.km, time_step=0 * u.s)

    with pytest.raises(ValueError):
        SpacecraftHistory.from_elliptical_orbit(semi_major_axis=7000 * u.km, time_step=-1 * u.s)


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


# --------------------------------------------------------------------------
# Multi-orbit duration / revolution unwrapping
# --------------------------------------------------------------------------

def test_multi_orbit_duration_unwraps_orbit_angle_monotonically():
    # The highest-risk code in this module: nu must keep advancing past 360
    # deg on every subsequent orbit instead of wrapping back to its
    # atan2-principal range. Check both the physical invariant (orbit_angle
    # strictly increasing throughout, since theta only ever advances for a
    # bound orbit) and the closed-form invariant (each period boundary is
    # exactly 360 deg * k past the start, by definition of the orbital
    # period), neither of which comes from anything the implementation
    # prints.
    a = 40000.0 * u.km  # perigee a(1-e) = 8000 km, clear of the ~6378 km Earth
    e = 0.8
    period_s = 2 * np.pi * np.sqrt(a.to_value(u.km)**3 / MU)
    n_orbits = 3
    n_per_orbit = 40  # coarse: only period boundaries and monotonicity matter here

    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=a, eccentricity=e,
        duration=n_orbits * period_s * u.s,
        time_step=(period_s / n_per_orbit) * u.s)

    start_times = np.array([iv.start_time.to_value(u.s) for iv in history])
    orbit_angles = np.array([iv.orbit_angle.to_value(u.deg) for iv in history])
    stop_time = list(history)[-1].stop_time.to_value(u.s)

    all_times = np.append(start_times, stop_time)
    # theta(t=0) == theta at the end of orbit n_orbits, plus n_orbits full
    # turns, by periodicity.
    all_angles = np.append(orbit_angles, orbit_angles[0] + 360.0 * n_orbits)

    assert np.all(np.diff(all_angles) > 0), "orbit_angle must be strictly increasing"

    for k in range(n_orbits + 1):
        t_k = k * period_s
        idx = int(np.argmin(np.abs(all_times - t_k)))
        assert all_times[idx] == pytest.approx(t_k, abs=1e-6)
        assert all_angles[idx] == pytest.approx(360.0 * k, abs=1e-6)


# --------------------------------------------------------------------------
# Kepler solver convergence
# --------------------------------------------------------------------------

def test_kepler_solver_raises_runtime_error_at_extreme_eccentricity():
    # Newton from E0 = M converges to machine precision through e = 0.95 and
    # diverges above roughly 0.96 (stated behaviour of the solver); at
    # e = 0.999 it must raise rather than silently hand back a
    # half-converged E.
    mean_anomaly = np.linspace(0.0, 2 * np.pi, 500, endpoint=False)

    with pytest.raises(RuntimeError):
        _solve_kepler_equation(mean_anomaly, eccentricity=0.999)


# --------------------------------------------------------------------------
# Untested parameter paths
# --------------------------------------------------------------------------

def test_livetime_fraction_scales_uptime_by_the_requested_fraction():
    fraction = 0.4

    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=7000 * u.km, eccentricity=0.0,
        time_step=500 * u.s, livetime_fraction=fraction)

    for interval in history:
        dt_s = (interval.stop_time - interval.start_time).to_value(u.s)
        assert interval.livetime.to_value(u.s) == pytest.approx(fraction * dt_s, rel=1e-9)


def test_initial_time_shifts_the_absolute_clock_not_the_orbit():
    # t_periapsis = 0 on the absolute clock, so a history generated with
    # initial_time = period/4 must start at exactly the pose a
    # initial_time = 0 history has at its own t = period/4 row.
    a = 10000.0 * u.km  # perigee a(1-e) = 7000 km, clear of the ~6378 km Earth
    e = 0.3
    period_s = 2 * np.pi * np.sqrt(a.to_value(u.km)**3 / MU)
    n = 40  # divisible by 4, so period/4 lands exactly on a row
    step = (period_s / n) * u.s

    baseline = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=a, eccentricity=e, time_step=step, duration=period_s * u.s)
    shifted = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=a, eccentricity=e, time_step=step, duration=step,
        initial_time=(period_s / 4) * u.s)

    quarter_row = list(baseline)[n // 4]
    shifted_row = list(shifted)[0]

    assert shifted_row.start_time.to_value(u.s) == pytest.approx(period_s / 4, rel=1e-9)
    assert shifted_row.orbit_radius.to_value(u.km) == pytest.approx(
        quarter_row.orbit_radius.to_value(u.km), rel=1e-9)
    assert shifted_row.orbit_angle.to_value(u.deg) == pytest.approx(
        quarter_row.orbit_angle.to_value(u.deg), rel=1e-9)


def test_argument_of_periapsis_shifts_orbit_angle_by_a_constant():
    # theta = nu + omega: omega is pose-independent of the true anomaly, so
    # rotating it must shift orbit_angle by exactly omega at every row,
    # while leaving orbit_radius (a function of E alone) untouched.
    a = 15000.0 * u.km  # perigee a(1-e) = 7500 km, clear of the ~6378 km Earth
    e = 0.5
    omega = 45 * u.deg
    step = 300 * u.s

    unrotated = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=a, eccentricity=e, time_step=step)
    rotated = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=a, eccentricity=e, time_step=step, argument_of_periapsis=omega)

    for iv0, iv1 in zip(unrotated, rotated):
        assert iv0.orbit_radius == iv1.orbit_radius
        diff_deg = (iv1.orbit_angle - iv0.orbit_angle).to_value(u.deg) % 360.0
        assert diff_deg == pytest.approx(omega.to_value(u.deg), abs=1e-9)


# --------------------------------------------------------------------------
# SpacecraftInterval.mid_time
# --------------------------------------------------------------------------

def test_interval_mid_time_is_the_average_of_start_and_stop():
    interval = SpacecraftInterval(
        start_time=10 * u.s, stop_time=30 * u.s, livetime=5 * u.s,
        orbit_radius=7000 * u.km, orbit_angle=0 * u.deg, attitude=0 * u.deg)

    assert interval.mid_time == 20 * u.s


def test_history_iteration_mid_time_matches_start_stop_average():
    history = _small_history()

    for interval in history:
        assert interval.mid_time == 0.5 * (interval.start_time + interval.stop_time)


# --------------------------------------------------------------------------
# earth property
# --------------------------------------------------------------------------

def test_earth_property_returns_the_earth_passed_to_init():
    earth = Earth(radius=6371 * u.km)
    history = SpacecraftHistory(
        time=np.array([0.0, 10.0, 20.0]) * u.s,
        orbit_radius=np.array([7000.0, 7000.0, 7000.0]) * u.km,
        orbit_angle=np.array([0.0, 10.0, 20.0]) * u.deg,
        attitude=np.array([0.0, 10.0, 20.0]) * u.deg,
        uptime=np.array([5.0, 5.0, 0.0]) * u.s,
        earth=earth)

    assert history.earth is earth
    assert history.earth.radius == 6371 * u.km


def test_earth_property_has_no_setter():
    history = _small_history()

    with pytest.raises(AttributeError):
        history.earth = Earth()


def test_plot_defaults_to_the_stored_earth(monkeypatch):
    earth = Earth(radius=6371 * u.km)
    history = SpacecraftHistory(
        time=np.array([0.0, 10.0, 20.0]) * u.s,
        orbit_radius=np.array([7000.0, 7000.0, 7000.0]) * u.km,
        orbit_angle=np.array([0.0, 10.0, 20.0]) * u.deg,
        attitude=np.array([0.0, 10.0, 20.0]) * u.deg,
        uptime=np.array([5.0, 5.0, 0.0]) * u.s,
        earth=earth)

    seen = []
    original_plot = Earth.plot

    def spy(self, ax=None):
        seen.append(self)
        return original_plot(self, ax=ax)

    monkeypatch.setattr(Earth, 'plot', spy)

    ax = history.plot()

    assert seen == [earth]
    plt.close(ax.figure)


# --------------------------------------------------------------------------
# plot()
# --------------------------------------------------------------------------

def test_history_plot_returns_axes_and_draws_orbit_and_earth():
    history = _small_history()

    ax = history.plot()

    assert ax is not None
    assert len(ax.lines) >= 1     # the orbit path
    assert len(ax.patches) >= 1   # the Earth disc

    plt.close(ax.figure)


def test_history_plot_nposes_greater_than_nintervals_does_not_crash():
    history = _small_history()  # nintervals == 3
    assert history.nintervals == 3

    ax = history.plot(nposes=100)

    assert ax is not None
    plt.close(ax.figure)


def test_plot_ignores_the_terminator_pose_when_it_is_a_placeholder():
    # Before the fix, the terminator's 999999 placeholder leaked into the
    # plotted data and blew out the axis limits by more than an order of
    # magnitude. The independent sanity bound here -- twice the orbit
    # radius -- is far below that failure mode and comfortably covers the
    # correct extent (orbit radius, plus attitude arrows of at most 8% of
    # the max radius, plus matplotlib's autoscale padding).
    orbit_radius_km = 7000.0
    history = SpacecraftHistory(
        time=np.array([0.0, 1000.0, 2000.0, 3000.0]) * u.s,
        orbit_radius=np.array([orbit_radius_km] * 3 + [999999.0]) * u.km,
        orbit_angle=np.array([0.0, 90.0, 180.0, 999999.0]) * u.deg,
        attitude=np.array([0.0, 90.0, 180.0, 999999.0]) * u.deg,
        uptime=np.array([500.0, 500.0, 500.0, 999999.0]) * u.s)

    ax = history.plot()

    bound = 2 * orbit_radius_km
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    assert max(abs(v) for v in xlim) < bound
    assert max(abs(v) for v in ylim) < bound

    plt.close(ax.figure)
