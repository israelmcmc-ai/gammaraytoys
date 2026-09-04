"""Tests for `EarthAlbedoSource` (plan Section 5.6, PR 5 entry in Section 7,
traps 1 and 6 in Section 8).

Sizing note, matching the house style of `tests/test_inertial_simulator.py`
and `tests/test_near_and_extended_sources.py`. Two kinds of statistical
assertion appear below, and each states which it is:

  - Count/binomial comparisons (an observed photon count, or an observed
    per-bin fraction, against a formula-derived expectation) state the sigma
    they are built on and assert at **4 sigma**.
  - KS goodness-of-fit checks (a sample against a reference distribution)
    assert `pvalue > _KS_PVALUE_FLOOR`, the two-sided 4-sigma-equivalent
    p-value (`2 * (1 - Phi(4)) = 6.3e-5`) -- NOT the much looser `p > 0.01`
    (only ~2.5 sigma).

Every expected value below is derived from Section 5.6's formulas or from
plain geometry, independently of the implementation:

    beta_max  = arccos(R_E / r)
    rho       = arcsin(R_E / r)
    s(beta)   = sqrt(r^2 + R_E^2 - 2 r R_E cos beta)
    cos_theta = (r cos beta - R_E) / s(beta)
    lam(beta) = arctan2(R_E sin beta, r - R_E cos beta)

    lambertian rate:  N = 2 a E arcsin(R_E / r)                          (closed form)
    isotropic rate:   N = (2 a E / pi) integral[ dlam / cos_theta(lam) ]  (SKY-ANGLE route,
                      with sin_theta = (r/R_E) sin(lam) -- deliberately NOT the beta route
                      `(2 a E R_E / pi) integral[dbeta / s(beta)]` the implementation itself
                      uses to compute this number, so the rate test below is not just
                      re-running the implementation's own integral.)

None of these numbers was obtained by running the implementation and reading
its output back; see
`/tmp/claude-0/-home-user-gammaraytoys/4a1d53f1-f083-5ad3-bf87-1c801ee0fb3a/scratchpad/pr5/TEST_NOTES.md`
for the first-principles numbers (rates, ratios, sampling tables) this file
is checked against, computed before any of this code existed.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import G, M_earth
from scipy import stats
from scipy.integrate import cumulative_trapezoid, quad

from gammaraytoys import ToyTracker2D
from gammaraytoys.coordinates import offaxis_to_sky_angle, sky_angle_to_offaxis
from gammaraytoys.sims import (Earth, EarthAlbedoSource, InertialSimulator,
                               MonoenergeticSpectrum, PointSource,
                               SimpleTraditionalReconstructor, SpacecraftHistory,
                               SpacecraftInterval, ZenithPointing)


# Two-sided 4-sigma-equivalent p-value: 2 * (1 - Phi(4)) = 6.3e-5. The floor
# every KS assertion in this file uses -- see the module docstring.
_KS_PVALUE_FLOOR = 6.3e-5

# Shared Earth/orbit geometry, matching the sampling numbers worked out in
# TEST_NOTES.md ("Sampling, at r = 6771 km (400 km altitude)"):
#   beta_max = 19.793 deg, rho = 70.207 deg.
EARTH_RADIUS = 6371.0 * u.km
ORBIT_RADIUS = 6771.0 * u.km

SPECTRUM = MonoenergeticSpectrum(1 * u.MeV)


def _make_tracker():
    return ToyTracker2D(material='Ge',
                        layer_length=16 * u.cm,
                        layer_positions=[0, 5, 10, 20, 25, 30] * u.mm,
                        layer_thickness=5 * u.mm,
                        energy_resolution=0.01,
                        energy_threshold=20 * u.keV)


def _make_earth():
    return Earth(radius=EARTH_RADIUS)


def _wrap180(angle_deg):
    """Wrap a plain-float angle (or array), in degrees, to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _pose(orbit_radius, orbit_angle=65 * u.deg, attitude=40 * u.deg):
    """A single-interval pose at a chosen orbital radius.

    `orbit_angle` and `attitude` default to deliberately non-trivial values
    (not 0 deg) so that a test recovering the drawn sky angle by inverting
    `offaxis_to_sky_angle` cannot pass by accident through a degenerate sign
    flip -- the same reasoning `test_near_and_extended_sources.py`'s GAP A
    block spells out for `ExtendedSource`.
    """
    return SpacecraftInterval(start_time=0 * u.s, stop_time=1 * u.s, livetime=1 * u.s,
                              orbit_radius=orbit_radius, orbit_angle=orbit_angle,
                              attitude=attitude)


def _drawn_sky_offsets_deg(source, detector, pose, earth, n):
    """Draw `n` photons and recover each one's signed sky-angle offset from
    nadir, in degrees.

    Recovers the drawn direction from `source._point_source.offaxis_angle`
    (the off-axis angle the underlying `PointSource` was actually re-aimed
    to and threw from) converted back to an inertial sky angle with
    `offaxis_to_sky_angle(offaxis_angle, pose.attitude)` -- the exact inverse
    of the `Nu = A - lambda` transform `random_photon` itself applies -- and
    then measured relative to `nadir = orbit_angle + 180 deg`. Same technique
    `test_near_and_extended_sources.py`'s `_drawn_sky_angles_deg` uses for
    `ExtendedSource`.

    Every draw is asserted non-`None`: `EarthAlbedoSource.occultable` is
    `False`, so a `None` here would itself be a bug (trap 1).
    """
    nadir_deg = (pose.orbit_angle + 180 * u.deg).to_value(u.deg)
    offsets = np.empty(n)
    for i in range(n):
        photon = source.random_photon(detector, pose=pose, earth=earth)
        assert photon is not None, "EarthAlbedoSource must never return None (trap 1)"
        lam_deg = offaxis_to_sky_angle(source._point_source.offaxis_angle,
                                       pose.attitude).to_value(u.deg)
        offsets[i] = _wrap180(lam_deg - nadir_deg)
    return offsets


def _orbital_period(semi_major_axis):
    """`2 pi sqrt(a^3 / mu)`, straight from astropy's constants -- matching
    `tests/test_inertial_simulator.py`."""
    return (2 * np.pi * np.sqrt(semi_major_axis**3 / (G * M_earth))).to(u.s)


def _make_history(duration, n_intervals, earth):
    """A circular orbit at `ORBIT_RADIUS`, tiled into `n_intervals` equal
    intervals over `duration` -- matching `tests/test_inertial_simulator.py`."""
    return SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=ORBIT_RADIUS,
        eccentricity=0.0,
        earth=earth,
        observation_strategy=ZenithPointing(),
        time_step=duration / n_intervals,
        duration=duration,
        livetime_fraction=1.0)


def _flux_for_expected_counts(mu, detector, livetime):
    """Invert `mu = flux * throwing_plane_size * livetime`."""
    return (mu / (detector.throwing_plane_size * livetime)).to(1 / u.cm / u.s)


# --- 1. Lambertian rate matches the closed form at several altitudes ------
# --- and 15. simulated_rate genuinely depends on orbit_radius -------------

def test_lambertian_rate_matches_closed_form_across_altitudes():
    # Section 5.6: N = 2 a E arcsin(R_E / r), a deterministic formula check
    # (no MC) at five altitudes spanning 100 km to 100 000 km, reproducing
    # TEST_NOTES.md's "Lambertian closed form vs direct surface integration"
    # table.
    detector = _make_tracker()
    earth = _make_earth()
    a_cm = detector.surrounding_circle_radius.to_value(u.cm)

    emissivity = 3.7 / u.cm / u.s
    E = emissivity.to_value(1 / u.cm / u.s)
    RE = EARTH_RADIUS.to_value(u.km)

    altitudes_km = [100.0, 400.0, 1000.0, 10000.0, 100000.0]
    rates_hz = []

    for altitude_km in altitudes_km:
        r_km = RE + altitude_km
        pose = _pose(r_km * u.km)

        source = EarthAlbedoSource(emissivity=emissivity, spectrum=SPECTRUM,
                                   law='lambertian', earth=earth)

        expected_hz = 2 * a_cm * E * np.arcsin(RE / r_km)

        actual = source.simulated_rate(detector, pose)
        assert actual.unit.is_equivalent(u.Hz)
        assert actual.to_value(u.Hz) == pytest.approx(expected_hz, rel=1e-9)

        rates_hz.append(actual.to_value(u.Hz))

    # NOTE on item 15 (simulated_rate genuinely depends on orbit_radius):
    # asserting that `rates_hz` varies across the loop above is tautological
    # here and proves nothing extra -- each `rates_hz[i]` was ALREADY pinned
    # to its own `expected_hz` (built from the SAME `RE + altitude_km`) at
    # rel=1e-9 just above, and every `expected_hz` in the list is trivially
    # different by construction, so "the rates differ" follows from the
    # preceding asserts with no additional test power: a source that ignored
    # `pose.orbit_radius` would already have failed the per-altitude
    # `expected_hz` comparison above, at the FIRST altitude that isn't its
    # (wrong) fixed one. The test that actually earns item 15 is
    # `test_earth_albedo_rate_cache_invalidates_when_orbit_radius_changes`
    # below: it reuses ONE source object across interleaved radii, which is
    # the only way to distinguish "uses pose.orbit_radius" from "cached the
    # first orbit_radius it ever saw and never updated" -- a fresh source
    # per altitude, as built in this loop, cannot tell those apart.


# --- 2. isotropic rate matches the independent SKY-ANGLE integral ---------
# --- and 16. the two laws are distinguishable at low altitude -------------

def test_isotropic_rate_matches_independent_sky_angle_integral():
    # Section 5.6 / TEST_NOTES.md: the isotropic total rate has no
    # elementary closed form, so the implementation computes it via the BETA
    # route, `(2 a E R_E / pi) integral[dbeta / s(beta)]`. This test computes
    # the SKY-ANGLE route instead, `(2 a E / pi) integral[dlam / cos_theta(lam)]`
    # with `sin_theta = (r/R_E) sin(lam)` -- an independent parametrisation
    # of the exact same physical integral (specific intensity conserved
    # along a ray), so agreement is a real cross-check and not the
    # implementation re-confirming its own arithmetic. `cos_theta` has an
    # integrable `eps^-1/2` divergence at `lam = +-rho` (Section 5.6); scipy's
    # adaptive `quad` handles that endpoint singularity to ~1e-12 relative
    # accuracy without any special substitution (checked by hand against a
    # sqrt-substituted version of this same integral before writing this
    # test), so no `points=` or transform is needed here.
    detector = _make_tracker()
    earth = _make_earth()
    a_cm = detector.surrounding_circle_radius.to_value(u.cm)

    emissivity = 3.7 / u.cm / u.s
    E = emissivity.to_value(1 / u.cm / u.s)
    RE = EARTH_RADIUS.to_value(u.km)

    altitudes_km = [100.0, 400.0, 1000.0, 10000.0, 100000.0]
    # TEST_NOTES.md's independently-verified isotropic/lambertian ratios.
    expected_ratios = [1.4134, 1.2289, 1.1341, 1.0140, 1.0003]

    for altitude_km, expected_ratio in zip(altitudes_km, expected_ratios):
        r_km = RE + altitude_km
        rho = np.arcsin(RE / r_km)

        def cos_theta(lam, r_km=r_km):
            sin_theta = (r_km / RE) * np.sin(lam)
            return np.sqrt(np.clip(1.0 - sin_theta**2, 0.0, None))

        integral, quad_err = quad(lambda lam: 1.0 / cos_theta(lam), -rho, rho, limit=200)
        expected_hz = (2 * a_cm * E / np.pi) * integral

        pose = _pose(r_km * u.km)
        source = EarthAlbedoSource(emissivity=emissivity, spectrum=SPECTRUM,
                                   law='isotropic', earth=earth)

        actual = source.simulated_rate(detector, pose)
        assert actual.to_value(u.Hz) == pytest.approx(expected_hz, rel=1e-6)

        lambertian_hz = 2 * a_cm * E * rho
        # Item 16: at 100 000 km the two laws are within 0.03% of each other
        # (a test there alone could not tell them apart); at 100 km they
        # differ by 41%. Checking the actual ratio at every altitude,
        # against TEST_NOTES.md's independently-verified numbers, is what
        # makes this a real distinguishing test rather than a single
        # high-altitude check where any bug that only mildly perturbs the
        # rate would slip through unnoticed.
        assert actual.to_value(u.Hz) / lambertian_hz == pytest.approx(expected_ratio, abs=2e-4)


# --- GAP 1: the geometry cache must invalidate on every orbit_radius, ----
# --- not just the first one seen ------------------------------------------
#
# Section 5.6 / CONTRACT.md: "constant for a circular orbit and changes
# only per interval otherwise". Every test above that checks a rate builds
# a FRESH `EarthAlbedoSource` per altitude, so none of them can tell "reads
# pose.orbit_radius correctly" apart from "cached whatever radius it saw on
# its first call and never updated" -- a source that freezes its geometry
# at construction-time's first radius would pass every one of them. These
# two tests reuse ONE source object across multiple, interleaved radii.

def test_earth_albedo_rate_cache_invalidates_when_orbit_radius_changes():
    # Alternates between a low (200 km) and a high (20 000 km) altitude,
    # several times each way, on a single long-lived source per law, and
    # checks `simulated_rate` against the closed/independent form for
    # WHICHEVER radius was just visited. A cache keyed on `orbit_radius`
    # only somewhat -- e.g. one that updates going up but not coming back
    # down -- would still fail the low-altitude checks after the first
    # high-altitude visit; interleaving both directions repeatedly is what
    # catches that, rather than a single low-then-high (or high-then-low)
    # pass.
    detector = _make_tracker()
    earth = _make_earth()
    a_cm = detector.surrounding_circle_radius.to_value(u.cm)
    RE = EARTH_RADIUS.to_value(u.km)

    emissivity = 2.3 / u.cm / u.s
    E = emissivity.to_value(1 / u.cm / u.s)

    r_lo_km = RE + 200.0
    r_hi_km = RE + 20000.0
    pose_lo = _pose(r_lo_km * u.km)
    pose_hi = _pose(r_hi_km * u.km)

    def expected_lambertian_hz(r_km):
        return 2 * a_cm * E * np.arcsin(RE / r_km)

    def expected_isotropic_hz(r_km):
        rho = np.arcsin(RE / r_km)

        def cos_theta(lam, r_km=r_km):
            sin_theta = (r_km / RE) * np.sin(lam)
            return np.sqrt(np.clip(1.0 - sin_theta**2, 0.0, None))

        integral, _ = quad(lambda lam: 1.0 / cos_theta(lam), -rho, rho, limit=200)
        return (2 * a_cm * E / np.pi) * integral

    expected_fns = {'lambertian': expected_lambertian_hz, 'isotropic': expected_isotropic_hz}

    for law, expected_fn in expected_fns.items():
        source = EarthAlbedoSource(emissivity=emissivity, spectrum=SPECTRUM,
                                   law=law, earth=earth)

        # lo, hi, lo, hi, lo, hi -- three full round trips.
        sequence = [(pose_lo, r_lo_km), (pose_hi, r_hi_km)] * 3

        for step, (pose, r_km) in enumerate(sequence):
            actual_hz = source.simulated_rate(detector, pose).to_value(u.Hz)
            expected_hz = expected_fn(r_km)
            assert actual_hz == pytest.approx(expected_hz, rel=1e-6), (
                f"{law}, step {step}: after visiting r = {r_km} km, "
                f"simulated_rate returned {actual_hz} Hz, expected "
                f"{expected_hz} Hz -- looks like a stale geometry cache")


def test_earth_albedo_sampling_cache_invalidates_when_orbit_radius_changes():
    # The companion check on the SAMPLING side of the same cache
    # (`_update_geometry` also rebuilds rho, the beta grid and the beta CDF,
    # not just the rate's `flux_factor`): after switching to a new radius,
    # every drawn sky angle must respect the NEW rho, not a stale one
    # inherited from whichever radius the cache last (really) updated to.
    # Same interleaved lo/hi/lo/hi/lo/hi pattern as the rate test above.
    detector = _make_tracker()
    earth = _make_earth()
    RE = EARTH_RADIUS.to_value(u.km)

    r_lo_km = RE + 200.0
    r_hi_km = RE + 20000.0
    rho_lo_deg = np.degrees(np.arcsin(RE / r_lo_km))
    rho_hi_deg = np.degrees(np.arcsin(RE / r_hi_km))
    # Sanity: the two rho values must be comfortably different, or a stale
    # cache could pass this test by accident.
    assert rho_hi_deg < 0.5 * rho_lo_deg

    pose_lo = _pose(r_lo_km * u.km)
    pose_hi = _pose(r_hi_km * u.km)

    n_per_step = 150

    for law in ('lambertian', 'isotropic'):
        source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM,
                                   law=law, earth=earth)

        sequence = [(pose_lo, rho_lo_deg), (pose_hi, rho_hi_deg)] * 3

        for step, (pose, rho_deg) in enumerate(sequence):
            offsets_deg = _drawn_sky_offsets_deg(source, detector, pose, earth, n_per_step)
            max_abs = np.abs(offsets_deg).max()
            assert max_abs <= rho_deg + 1e-6, (
                f"{law}, step {step}: max |offset| = {max_abs} deg exceeds "
                f"rho = {rho_deg} deg for this pose -- looks like a stale "
                "geometry cache")

            # And, for the high-altitude steps specifically, a cache stuck
            # on the low-altitude rho (much wider) would still pass the
            # bound above vacuously if it happened to sample only small
            # angles -- so also require the draws to actually use most of
            # the CURRENT rho's range, not just a stale wider one's centre.
            if rho_deg == rho_hi_deg:
                assert max_abs > 0.5 * rho_hi_deg, (
                    f"{law}, step {step}: draws stayed within "
                    f"{max_abs} deg, well inside the current rho = "
                    f"{rho_hi_deg} deg -- looks like a stale, wider cached rho")


def test_earth_albedo_expected_count_matches_independent_per_interval_sum_on_an_elliptical_orbit():
    # The second prong of GAP 1: a real elliptical orbit (a = 14371 km,
    # e = 0.35, matching the review's own measurement), whose orbital radius
    # varies continuously interval to interval -- 200 intervals, radius
    # ranging over roughly 9341-19401 km for this orbit. `InertialSimulator`
    # queries the SAME long-lived `EarthAlbedoSource` object once per
    # interval via `simulated_rate(detector, pose)`
    # (`InertialSimulator._expected_counts`, and identically inside
    # `run_events`'s own Poisson-mean computation) -- exactly the "one
    # source, many different radii" situation a frozen-at-first-radius cache
    # cannot survive. The comparison total is summed here per interval by
    # hand, from the SAME closed/independent forms used elsewhere in this
    # file, never by calling `source.simulated_rate`.
    #
    # `_expected_counts()` is used rather than a full stochastic
    # `run_events()` because it is deterministic (no Poisson noise to
    # budget sigma for) and exercises exactly the code path
    # (`simulated_rate(detector, pose)` once per interval, on one shared
    # source) the review's own reproduction measured: a stale cache showed
    # up there as "the reused source reports a total expected count +52.8%
    # (lambertian) / +56.8% (isotropic) above the truth" -- the same
    # quantity, and the same rough magnitude of error, this test's `rel`
    # tolerance is nowhere near wide enough to hide.
    earth = _make_earth()
    detector = _make_tracker()
    a_cm = detector.surrounding_circle_radius.to_value(u.cm)
    RE = EARTH_RADIUS.to_value(u.km)

    semi_major_axis = 14371.0 * u.km
    eccentricity = 0.35
    period = _orbital_period(semi_major_axis)
    n_intervals = 200

    history = SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=semi_major_axis, eccentricity=eccentricity, earth=earth,
        observation_strategy=ZenithPointing(), time_step=period / n_intervals,
        duration=period, livetime_fraction=1.0)

    radii_km = [interval.orbit_radius.to_value(u.km) for interval in history]
    # Sanity: the orbit really does visit a wide range of radii, or this
    # test would not distinguish a correct cache from a frozen one.
    assert max(radii_km) > 1.5 * min(radii_km)

    emissivity = 1.7 / u.cm / u.s
    E = emissivity.to_value(1 / u.cm / u.s)

    for law in ('lambertian', 'isotropic'):
        source = EarthAlbedoSource(emissivity=emissivity, spectrum=SPECTRUM,
                                   law=law, earth=earth)

        independent_total = 0.0
        for interval in history:
            r_km = interval.orbit_radius.to_value(u.km)
            livetime_s = interval.livetime.to_value(u.s)

            if law == 'lambertian':
                rate_hz = 2 * a_cm * E * np.arcsin(RE / r_km)
            else:
                rho = np.arcsin(RE / r_km)

                def cos_theta(lam, r_km=r_km):
                    sin_theta = (r_km / RE) * np.sin(lam)
                    return np.sqrt(np.clip(1.0 - sin_theta**2, 0.0, None))

                integral, _ = quad(lambda lam: 1.0 / cos_theta(lam), -rho, rho, limit=200)
                rate_hz = (2 * a_cm * E / np.pi) * integral

            independent_total += rate_hz * livetime_s

        simulator = InertialSimulator(detector=detector, sources=[source],
                                      reconstructor=SimpleTraditionalReconstructor(),
                                      spacecraft_history=history, earth=earth)

        actual_total = simulator._expected_counts()

        assert actual_total == pytest.approx(independent_total, rel=1e-4), (
            f"{law}: InertialSimulator._expected_counts() = {actual_total} "
            f"disagrees with the independently per-interval-summed total "
            f"{independent_total} on an elliptical orbit")


# --- 3. sampled sky angles all fall within rho of nadir -------------------

def test_sampled_sky_angles_all_fall_within_rho_of_nadir():
    # Section 5.6: every drawn direction comes from a surface point actually
    # visible from the spacecraft, |beta| < beta_max, whose sky angle is
    # bounded by lam(beta_max) = rho identically. Checked for EVERY draw
    # (100%, not a statistical fraction) at both laws.
    detector = _make_tracker()
    earth = _make_earth()
    RE = EARTH_RADIUS.to_value(u.km)
    r = ORBIT_RADIUS.to_value(u.km)
    rho_deg = np.degrees(np.arcsin(RE / r))

    pose = _pose(ORBIT_RADIUS)
    n = 800

    for law in ('lambertian', 'isotropic'):
        source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM,
                                   law=law, earth=earth)
        offsets_deg = _drawn_sky_offsets_deg(source, detector, pose, earth, n)

        assert np.all(np.abs(offsets_deg) <= rho_deg + 1e-6), (
            f"{law}: a sampled sky angle fell outside rho = {rho_deg} deg of nadir "
            f"(max |offset| = {np.abs(offsets_deg).max()} deg)")


# --- TQ3: a check made directly on photon.direction, not through the -----
# --- inverse of the transform random_photon just applied -----------------

def test_random_photon_direction_matches_a_hand_computed_value_at_a_known_pose():
    # `_drawn_sky_offsets_deg` (used by most tests above) recovers the drawn
    # sky angle with `offaxis_to_sky_angle(offaxis, pose.attitude)`, the
    # EXACT inverse of the `Nu = A - lambda` transform `random_photon` just
    # applied via `sky_angle_to_offaxis`. Composing a transform with its own
    # inverse cancels any error in how `attitude` (or `orbit_angle`) is
    # used identically on both sides -- the PR 4 lesson recorded in
    # `.claude/cosimita-progress.md` ("a test helper must not invert the
    # transform under test"). This test instead reads `photon.direction`
    # directly and compares it to a value computed by hand from Section
    # 3.4's stated conventions ("a source at off-axis angle Nu lies along
    # (sin Nu, cos Nu) while the photon it emits flies along 270 deg - Nu"),
    # composed with `Nu = A - lambda` and `lambda = nadir = orbit_angle +
    # 180 deg` -- NOT by calling `sky_angle_to_offaxis` / `offaxis_to_sky_angle`.
    #
    # To make this a single-draw, deterministic check (no KS test, no
    # statistics) rather than a check on a distribution's mean, the orbital
    # radius is pushed absurdly high (1e7 km) so rho collapses to
    # ~0.0365 deg -- test 3 above already establishes every draw lands
    # within rho of nadir, so at this radius EVERY draw, for EITHER law, is
    # nadir to within 0.0365 deg, and `photon.direction` must match the
    # hand-computed nadir direction to that same tight tolerance.
    #
    # `orbit_angle = 50 deg` and `attitude = 125 deg` are both deliberately
    # non-zero and mutually distinct, so a bug that swaps them, drops one,
    # or uses the wrong sign cannot pass by coincidence the way it could at
    # the degenerate attitude = orbit_angle = 0 deg.
    detector = _make_tracker()
    earth = _make_earth()

    orbit_angle = 50 * u.deg
    attitude = 125 * u.deg
    r = 1.0e7 * u.km
    pose = SpacecraftInterval(start_time=0 * u.s, stop_time=1 * u.s, livetime=1 * u.s,
                              orbit_radius=r, orbit_angle=orbit_angle, attitude=attitude)

    rho_deg = np.degrees(np.arcsin(EARTH_RADIUS.to_value(u.km) / r.to_value(u.km)))
    assert rho_deg == pytest.approx(0.0365, abs=0.0005)

    nadir_deg = (orbit_angle + 180 * u.deg).to_value(u.deg)
    expected_nu_deg = _wrap180(attitude.to_value(u.deg) - nadir_deg)
    expected_direction_deg = (270.0 - expected_nu_deg) % 360.0
    assert expected_nu_deg == pytest.approx(-105.0)
    assert expected_direction_deg == pytest.approx(15.0)

    for law in ('lambertian', 'isotropic'):
        source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM,
                                   law=law, earth=earth)

        n = 60
        for _ in range(n):
            photon = source.random_photon(detector, pose=pose, earth=earth)
            assert photon is not None

            direction_deg = photon.direction.to_value(u.deg)
            diff = ((direction_deg - expected_direction_deg + 180.0) % 360.0) - 180.0

            assert abs(diff) <= rho_deg + 1e-6, (
                f"{law}: photon.direction = {direction_deg} deg is "
                f"{diff} deg from the hand-computed nadir direction "
                f"{expected_direction_deg} deg (tolerance {rho_deg} deg)")


# --- 4. the sampled beta distribution passes a KS test against pdf ~ 1/s --

def test_isotropic_beta_distribution_passes_ks_against_pdf_over_s():
    # Section 5.6 / trap 8.6: the isotropic law must be sampled in beta, from
    # pdf(beta) ~ 1/s(beta), never directly in sky angle. This test recovers
    # each draw's sky-angle offset from nadir (an external observable, via
    # `_drawn_sky_offsets_deg`) and inverts the plan's own forward mapping
    # `lam(beta) = arctan2(R_E sin beta, r - R_E cos beta)` -- built fresh
    # here on an independent grid, never touching the implementation's own
    # cached beta table -- to recover each draw's beta. The recovered betas
    # are then KS-tested against an independently tabulated CDF of
    # pdf(beta) ~ 1/s(beta) (`cumulative_trapezoid`, the same "about six
    # readable lines" technique Section 5.6 prescribes, computed here from
    # scratch as an oracle, exactly like `test_near_and_extended_sources.py`
    # uses `scipy.stats.vonmises` as an oracle for `ExtendedSource`).
    detector = _make_tracker()
    earth = _make_earth()
    RE = EARTH_RADIUS.to_value(u.km)
    r = ORBIT_RADIUS.to_value(u.km)
    beta_max = np.arccos(RE / r)

    pose = _pose(ORBIT_RADIUS)

    beta_grid = np.linspace(-beta_max, beta_max, 4001)
    # s(beta), written to avoid the same catastrophic-cancellation trap the
    # plan warns about (Section 5.6): s^2 = (r-R_E)^2 + 4 r R_E sin^2(beta/2).
    s_grid = np.sqrt((r - RE)**2 + 4 * r * RE * np.sin(beta_grid / 2)**2)
    pdf_unnorm = 1.0 / s_grid
    lam_grid = np.arctan2(RE * np.sin(beta_grid), r - RE * np.cos(beta_grid))

    assert np.all(np.diff(lam_grid) > 0)  # lam(beta) must be monotonic to invert by np.interp

    # Consistency check on this test's OWN grid (Section 5.6:
    # "lam(beta_max) == rho identically"), independent of the
    # implementation's own internal check of the same identity.
    rho_rad = np.arcsin(RE / r)
    assert lam_grid[-1] == pytest.approx(rho_rad, abs=1e-9)

    beta_cdf_grid = cumulative_trapezoid(pdf_unnorm, beta_grid, initial=0)
    beta_cdf_grid /= beta_cdf_grid[-1]

    def beta_cdf(x):
        return np.interp(x, beta_grid, beta_cdf_grid)

    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM,
                               law='isotropic', earth=earth)

    n = 4000
    offset_rad = np.deg2rad(_drawn_sky_offsets_deg(source, detector, pose, earth, n))

    # Invert the forward beta -> lam map by interpolation (lam_grid is
    # monotonic, checked above).
    beta_samples = np.interp(offset_rad, lam_grid, beta_grid)

    result = stats.kstest(beta_samples, beta_cdf)
    assert result.pvalue > _KS_PVALUE_FLOOR


# --- 5. the Lambertian sky angle is uniform over [nadir-rho, nadir+rho] ---

def test_lambertian_sky_angle_is_uniform_over_nadir_pm_rho():
    # Section 5.6: the Lambertian radiance k = E/2 is independent of angle,
    # so the sky angle is drawn uniformly on [-rho, rho] around nadir, with
    # no surface sampling at all. KS test against Uniform(-rho, rho) deg.
    #
    # Alternative hypothesis rejected: the sampler routes Lambertian through
    # the beta machinery instead of drawing uniformly (or draws uniformly in
    # beta rather than in sky angle) -- either would visibly concentrate
    # mass away from the limb, which this KS test has power against.
    detector = _make_tracker()
    earth = _make_earth()
    RE = EARTH_RADIUS.to_value(u.km)
    r = ORBIT_RADIUS.to_value(u.km)
    rho_deg = np.degrees(np.arcsin(RE / r))

    pose = _pose(ORBIT_RADIUS)

    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM,
                               law='lambertian', earth=earth)

    n = 4000
    offsets_deg = _drawn_sky_offsets_deg(source, detector, pose, earth, n)

    assert np.abs(offsets_deg).max() <= rho_deg + 1e-6

    result = stats.kstest(offsets_deg, stats.uniform(loc=-rho_deg, scale=2 * rho_deg).cdf)
    assert result.pvalue > _KS_PVALUE_FLOOR


# --- 6. the isotropic law is limb-brightened relative to the Lambertian ---

def test_isotropic_is_limb_brightened_relative_to_lambertian():
    # Section 5.6: isotropic brightness in sky angle goes as 1/cos_theta(lam),
    # so unlike the Lambertian's flat 20% per equal-width |lam|/rho bin, the
    # isotropic distribution must pile up toward the limb. Predicted per-bin
    # fractions are computed here by integrating 1/cos_theta(lam) (the SAME
    # independent sky-angle route as the rate test above, never the
    # implementation's beta-route sampler) over each of five equal
    # |lam|/rho bins and normalising by the total.
    detector = _make_tracker()
    earth = _make_earth()
    RE = EARTH_RADIUS.to_value(u.km)
    r = ORBIT_RADIUS.to_value(u.km)
    rho_rad = np.arcsin(RE / r)
    rho_deg = np.degrees(rho_rad)

    pose = _pose(ORBIT_RADIUS)

    def cos_theta(lam):
        sin_theta = (r / RE) * np.sin(lam)
        return np.sqrt(np.clip(1.0 - sin_theta**2, 0.0, None))

    total, _ = quad(lambda lam: 1.0 / cos_theta(lam), 0.0, rho_rad, limit=200)

    edges_rad = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) * rho_rad
    predicted_fracs = np.array([
        quad(lambda lam: 1.0 / cos_theta(lam), lo, hi, limit=200)[0] / total
        for lo, hi in zip(edges_rad[:-1], edges_rad[1:])
    ])
    assert predicted_fracs.sum() == pytest.approx(1.0, abs=1e-6)

    # Orientation check against TEST_NOTES.md's independently-verified
    # (200000-draw MC) per-bin ratios -- 0.527, 0.560, 0.661, 0.859, 2.420 --
    # loose tolerance since that reference is itself a Monte Carlo estimate,
    # not the exact quad integral computed here.
    reference_ratios = np.array([0.527, 0.560, 0.661, 0.859, 2.420])
    assert predicted_fracs / 0.2 == pytest.approx(reference_ratios, abs=0.06)

    # Monotone rise toward the limb -- the qualitative signature (Section
    # 5.6, TEST_NOTES.md) that distinguishes isotropic from a flat 20% in
    # every bin.
    assert np.all(np.diff(predicted_fracs) > 0)

    n = 6000
    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM,
                               law='isotropic', earth=earth)
    offsets_deg = _drawn_sky_offsets_deg(source, detector, pose, earth, n)

    ratio = np.abs(offsets_deg) / rho_deg
    bin_edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    observed_fracs = np.array([
        np.mean((ratio >= lo) & (ratio <= hi if hi == 1.0 else ratio < hi))
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:])
    ])
    assert observed_fracs.sum() == pytest.approx(1.0, abs=1e-9)

    for k, (obs, pred) in enumerate(zip(observed_fracs, predicted_fracs)):
        sigma = np.sqrt(pred * (1 - pred) / n)
        assert abs(obs - pred) < 4 * sigma, (
            f"bin {k}: observed fraction {obs} vs predicted {pred} "
            f"(sigma = {sigma})")

    # The strong, unambiguous end-to-end signature: the outermost bin must
    # hold several times as many draws as the innermost -- predicted 4.52x,
    # so a 2x margin here comfortably tolerates sampling noise while still
    # ruling out a Lambertian-like (flat, ~1x) distribution.
    assert observed_fracs[-1] > 2 * observed_fracs[0]


# --- 7. the albedo is not suppressed by occultation (trap 1) --------------

def test_earth_albedo_is_not_suppressed_by_occultation():
    # Trap 1 / Section 8.1. Precisely what this test does and does NOT show:
    # today, `EarthAlbedoSource.random_photon` never calls `self._occulted`
    # at all (unlike `PointSource`/`IsotropicSource`/`ExtendedSource`, which
    # each do) -- its immunity to occultation is structural, coming from
    # only ever sampling |beta| < beta_max (surface points actually visible
    # from the spacecraft), not from consulting `occultable`. So this test
    # is NOT independent confirmation that `occultable` is wired into a live
    # rejection path -- flipping `occultable` to `True` alone leaves this
    # test passing (only the direct property test,
    # `test_occultable_is_false_for_both_laws`, catches that). What this
    # test DOES guard is the regression that would matter most: if a future
    # change added an `_occulted(...)` check to `random_photon` (the pattern
    # every other far-field source uses), it would occult essentially ALL of
    # the albedo's own photons, not merely a fraction -- every drawn sky
    # angle sits within rho of the CURRENT nadir by construction (test 3
    # above), and the albedo re-aims to that current nadir every single
    # draw, so it is never on the "unoccluted" side the way a fixed-sky-
    # angle source sometimes is over an orbit (checked directly: applying
    # `Earth._is_occulted` to 2000 of this test's own draws, post hoc,
    # occults 2000/2000 = 100% of them). n_albedo would collapse from
    # ~1200 to ~0, astronomically far from the assertion below -- this test
    # has ample power against that regression. Mixing an ordinary
    # PointSource into the SAME run, which the geometry predicts loses only
    # a fraction rho/pi = 0.3900 of ITS photons over a full orbit, is the
    # contrast that shows the albedo behaving differently in practice today.
    earth = _make_earth()
    detector = _make_tracker()

    period = _orbital_period(ORBIT_RADIUS)
    history = _make_history(period, 360, earth)

    mu = 1200.0

    point_flux = _flux_for_expected_counts(mu, detector, period)
    point_source = PointSource(sky_angle=0 * u.deg, spectrum=SPECTRUM, flux=point_flux)

    rho_rad = np.arcsin(EARTH_RADIUS.to_value(u.km) / ORBIT_RADIUS.to_value(u.km))
    a_cm = detector.throwing_plane_size.to_value(u.cm) / 2.0
    period_s = period.to_value(u.s)
    emissivity_value = mu / (2 * a_cm * rho_rad * period_s)  # invert N = 2 a E rho
    albedo_source = EarthAlbedoSource(emissivity=emissivity_value / u.cm / u.s,
                                      spectrum=SPECTRUM, law='lambertian', earth=earth)

    simulator = InertialSimulator(detector=detector,
                                  sources=[point_source, albedo_source],
                                  reconstructor=SimpleTraditionalReconstructor(),
                                  spacecraft_history=history,
                                  earth=earth)

    events = list(simulator.run_events())

    n_point = sum(1 for _, source, _, _ in events if source is point_source)
    n_albedo = sum(1 for _, source, _, _ in events if source is albedo_source)

    expected_point = mu * (1.0 - rho_rad / np.pi)
    sigma_point = np.sqrt(expected_point)
    assert abs(n_point - expected_point) < 4 * sigma_point

    sigma_albedo = np.sqrt(mu)
    assert abs(n_albedo - mu) < 4 * sigma_albedo

    # Not merely "both survived some photons" -- the albedo's survival, in
    # the very same run and against the very same Earth, must be measurably
    # more complete than the ordinary far-field source's.
    assert n_albedo > n_point


# --- 8. law outside {'lambertian', 'isotropic'} raises ---------------------

def test_law_other_than_lambertian_or_isotropic_raises():
    with pytest.raises(ValueError):
        EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM, law='specular')


# --- 9. zero and negative emissivity raise ---------------------------------

def test_zero_or_negative_emissivity_raises():
    with pytest.raises(ValueError):
        EarthAlbedoSource(emissivity=0 / u.cm / u.s, spectrum=SPECTRUM)

    with pytest.raises(ValueError):
        EarthAlbedoSource(emissivity=-1 / u.cm / u.s, spectrum=SPECTRUM)


# --- 10. simulated_rate and random_photon raise when pose is None ---------

def test_simulated_rate_and_random_photon_require_a_pose():
    # Unlike every other far-field source, this one has no pose-free mode at
    # all (Section 5.6): its normalization depends on how much sky the Earth
    # fills, which depends on orbit_radius.
    detector = _make_tracker()
    earth = _make_earth()
    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM, earth=earth)

    with pytest.raises(ValueError):
        source.simulated_rate(detector)

    with pytest.raises(ValueError):
        source.random_photon(detector, earth=earth)


# --- 11. random_photon raises when earth is None ---------------------------

def test_random_photon_requires_earth():
    # Not for occultation testing (occultable is False) -- but the Earth's
    # radius is what the source samples FROM, and random_photon has no other
    # way to learn it at draw time.
    detector = _make_tracker()
    earth = _make_earth()
    pose = _pose(ORBIT_RADIUS)
    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM, earth=earth)

    with pytest.raises(ValueError):
        source.random_photon(detector, pose=pose)


# --- 12. an orbit radius at or below the Earth's surface raises -----------

def test_orbit_radius_at_or_below_earth_surface_raises():
    detector = _make_tracker()
    earth = _make_earth()
    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM, earth=earth)

    pose_at_surface = _pose(EARTH_RADIUS)
    with pytest.raises(ValueError):
        source.simulated_rate(detector, pose_at_surface)

    pose_below_surface = _pose(EARTH_RADIUS - 100 * u.km)
    with pytest.raises(ValueError):
        source.random_photon(detector, pose=pose_below_surface, earth=earth)


# --- 13. an earth at draw time that disagrees with the source's own -------

def test_random_photon_rejects_a_mismatched_earth():
    detector = _make_tracker()
    earth = _make_earth()
    pose = _pose(ORBIT_RADIUS)
    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM, earth=earth)

    wrong_earth = Earth(radius=EARTH_RADIUS + 500 * u.km)
    with pytest.raises(ValueError):
        source.random_photon(detector, pose=pose, earth=wrong_earth)

    # The source's own Earth is always accepted (sanity: the check above
    # isn't rejecting every Earth indiscriminately).
    photon = source.random_photon(detector, pose=pose, earth=earth)
    assert photon is not None


# --- 14. occultable is False for both laws ---------------------------------

def test_occultable_is_false_for_both_laws():
    for law in ('lambertian', 'isotropic'):
        source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM, law=law)
        assert source.occultable is False


# ===========================================================================
# GAP 2: plot()
# ===========================================================================

def test_plot_raises_before_any_photon_has_been_drawn():
    # `EarthAlbedoSource.plot` has no pose of its own -- it only knows
    # where nadir is, and rho, once a photon has actually been drawn at
    # some pose. Mirrors the same requirement `PointSource(sky_angle=...)`
    # and `ExtendedSource` already have.
    detector = _make_tracker()
    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    detector.plot(ax=ax)

    with pytest.raises(RuntimeError):
        source.plot(ax, detector)

    plt.close(fig)


def test_plot_arc_is_centred_on_nadir_with_the_full_earth_diameter_as_extent():
    # After a draw, `plot` must show an arc of angular width `2 rho` (the
    # Earth's full apparent diameter) centred on nadir in the detector
    # frame -- exactly the patch of sky every sampled photon can come from
    # (test 3 above). `expected_center` uses `sky_angle_to_offaxis`, the
    # SAME helper `ExtendedSource`'s own analogous plot test
    # (`test_extended_source_plot_arc_is_centred_on_the_sky_angle_not_last_draw`
    # in `test_near_and_extended_sources.py`) uses -- this is checking which
    # POSE `plot` reflects (current vs. stale), a different property from
    # whether the coordinate transform itself is correct, which is pinned
    # directly on `photon.direction`, independent of this helper, by
    # `test_random_photon_direction_matches_a_hand_computed_value_at_a_known_pose`
    # above (the TQ3 fix).
    detector = _make_tracker()
    earth = _make_earth()
    RE = EARTH_RADIUS.to_value(u.km)
    r = ORBIT_RADIUS.to_value(u.km)
    rho_deg = np.degrees(np.arcsin(RE / r))

    orbit_angle = 65 * u.deg
    attitude = 40 * u.deg
    pose = _pose(ORBIT_RADIUS, orbit_angle=orbit_angle, attitude=attitude)

    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM,
                               law='lambertian', earth=earth)

    photon = source.random_photon(detector, pose=pose, earth=earth)
    assert photon is not None

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    detector.plot(ax=ax)
    source.plot(ax, detector)

    line = ax.get_lines()[-1]
    xdata, ydata = line.get_xdata(), line.get_ydata()

    center = detector.surrounding_circle_center
    cx = center.x.to(u.cm).value
    cy = center.y.to(u.cm).value

    mid_x = xdata[len(xdata) // 2] - cx
    mid_y = ydata[len(ydata) // 2] - cy
    plotted_center_deg = np.degrees(np.arctan2(mid_x, mid_y))

    angle0_deg = np.degrees(np.arctan2(xdata[0] - cx, ydata[0] - cy))
    angle1_deg = np.degrees(np.arctan2(xdata[-1] - cx, ydata[-1] - cy))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    plt.close(fig)

    nadir = orbit_angle + 180 * u.deg
    expected_center = sky_angle_to_offaxis(nadir, attitude)

    diff = ((plotted_center_deg - expected_center.to_value(u.deg) + 180) % 360) - 180
    assert abs(diff) < 1.0

    extent_deg = abs(((angle1_deg - angle0_deg + 180) % 360) - 180)
    assert extent_deg == pytest.approx(2 * rho_deg, abs=0.5)

    # PR 4's off-screen trap (Section 8, item 5's sibling regressions in
    # `test_near_and_extended_sources.py`): the arc must actually be inside
    # the visible axes, not clipped out of view.
    assert xlim[0] <= xdata.min() and xdata.max() <= xlim[1]
    assert ylim[0] <= ydata.min() and ydata.max() <= ylim[1]


# ===========================================================================
# GAP 3: normalization
# ===========================================================================

def test_normalization_returns_the_emissivity_not_the_pose_dependent_flux():
    # `normalization` (used polymorphically by `diff_flux`, `integrate_flux`,
    # `discretize_spectrum` and `plot_spectrum`) must be the pose-FREE
    # emissivity, not `flux(pose)` -- `flux` needs a pose and, for the
    # Lambertian law (Section 5.6), returns `emissivity * rho`, not the bare
    # emissivity. Pinned by checking the two actually differ numerically
    # (rho != 1 rad at this altitude, so they can't coincide) and that
    # `flux(pose) == normalization * rho` exactly -- stronger than merely
    # checking `normalization` is "some positive 1/cm/s number", which a
    # `normalization` that fell through to `flux(pose)`'s value could also
    # satisfy by accident.
    earth = _make_earth()
    emissivity = 4.2 / u.cm / u.s
    source = EarthAlbedoSource(emissivity=emissivity, spectrum=SPECTRUM,
                               law='lambertian', earth=earth)

    normalization_value = source.normalization.to_value(1 / u.cm / u.s)
    assert normalization_value == pytest.approx(emissivity.to_value(1 / u.cm / u.s), rel=1e-12)

    pose = _pose(ORBIT_RADIUS)
    flux_value = source.flux(pose).to_value(1 / u.cm / u.s)

    rho_rad = np.arcsin(EARTH_RADIUS.to_value(u.km) / ORBIT_RADIUS.to_value(u.km))
    assert rho_rad == pytest.approx(1.2254, abs=0.001)  # comfortably != 1 rad

    assert normalization_value != pytest.approx(flux_value, rel=1e-3)
    assert flux_value == pytest.approx(normalization_value * rho_rad, rel=1e-9)


# ===========================================================================
# Regression: InertialSimulator rejects a mismatched EarthAlbedoSource Earth
# at CONSTRUCTION, not only per photon (fix commit 641ab3a)
# ===========================================================================

def test_inertial_simulator_rejects_a_mismatched_albedo_earth_at_construction():
    # `EarthAlbedoSource` carries its own `Earth` (it needs R_E for its own
    # geometry); if that disagrees with the simulator's own `Earth`, the run
    # silently mixes two planets -- caught previously only per photon
    # (`EarthAlbedoSource._check_earth`, inside `random_photon`), which
    # means a run whose Poisson draw happens to come up empty for that
    # source finishes with no error at all, having computed its expected
    # count from the wrong Earth the whole time. `InertialSimulator` must
    # now catch this eagerly, at construction.
    #
    # The realistic trap this guards, called out explicitly in the fix
    # commit: astropy's default `Earth()` uses R_E = 6378.1 km, while this
    # project's own tests and notebooks use 6371 km -- so the plain
    # `EarthAlbedoSource(emissivity, spectrum)` form (no explicit `earth=`)
    # is exactly the mismatching one against an explicit
    # `Earth(radius = 6371 km)` simulator, which is what this test uses.
    detector = _make_tracker()
    simulator_earth = _make_earth()  # EARTH_RADIUS = 6371 km, explicit

    duration = 1000 * u.s
    history = _make_history(duration, 1, simulator_earth)

    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM)
    # Sanity: this really is the mismatching, no-`earth=` form the fix
    # commit is about, and the two radii really do differ.
    assert source.earth.radius != simulator_earth.radius

    with pytest.raises(ValueError):
        InertialSimulator(detector=detector, sources=[source],
                          reconstructor=SimpleTraditionalReconstructor(),
                          spacecraft_history=history, earth=simulator_earth)

    # The matching form -- explicit earth = the simulator's own -- must
    # construct cleanly.
    matching_source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM,
                                        earth=simulator_earth)
    InertialSimulator(detector=detector, sources=[matching_source],
                      reconstructor=SimpleTraditionalReconstructor(),
                      spacecraft_history=history, earth=simulator_earth)
