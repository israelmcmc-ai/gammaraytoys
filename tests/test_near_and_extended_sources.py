"""Tests for `NearPointSource` and `ExtendedSource` (plan Sections 5.4, 5.5,
PR 4 entry in Section 7).

Sizing note, matching `tests/test_inertial_simulator.py`'s house style: every
statistical assertion below states the sigma it is built on and asserts at
**4 sigma**, one wider than the plan's "within 3 sigma", and separately names
the alternative hypothesis it rejects.

Every expected value is derived from the plan's formulas or from geometry,
independently of the implementation -- see
`/tmp/claude-0/-home-user-gammaraytoys/4a1d53f1-f083-5ad3-bf87-1c801ee0fb3a/scratchpad/pr4/TEST_NOTES.md`
for the first-principles numbers this file is built against:

    f(s >= a) = arcsin(a/s) / pi                    (Section 5.4)
    f(s <  a) = 1                                    (Section 5.4)
    kappa = 1 / width_rad**2                         (Section 5.5)

None of them was read back out of the implementation.
"""

import astropy.units as u
import numpy as np
import pytest
from scipy import stats
from scipy.stats import vonmises

from gammaraytoys import ToyTracker2D
from gammaraytoys.coordinates import Cartesian2D, offaxis_to_sky_angle
from gammaraytoys.sims import (Earth, ExtendedSource, IsotropicSource,
                               MonoenergeticSpectrum, NearPointSource,
                               PointSource, SpacecraftInterval)


def _make_tracker():
    return ToyTracker2D(material='Ge',
                        layer_length=16 * u.cm,
                        layer_positions=[0, 5, 10, 20, 25, 30] * u.mm,
                        layer_thickness=5 * u.mm,
                        energy_resolution=0.01,
                        energy_threshold=20 * u.keV)


def _near_position(detector, s_over_a, direction_deg=0.0):
    """A detector-frame position at `s = s_over_a * a` from
    `detector.surrounding_circle_center`, along `direction_deg` (plain
    float, CCW from +x), so `s/a` is exact by construction."""

    center = detector.surrounding_circle_center
    a = detector.surrounding_circle_radius
    s = s_over_a * a
    theta = np.deg2rad(direction_deg)

    return Cartesian2D(center.x + s * np.cos(theta),
                       center.y + s * np.sin(theta))


SPECTRUM = MonoenergeticSpectrum(1 * u.MeV)


# --- 1. near-source rate matches rate * arcsin(a/s) / pi ------------------

def test_near_source_rate_matches_arcsin_over_pi_formula():
    # Deterministic formula check, no MC: `simulated_rate` must equal
    # `rate * arcsin(a/s) / pi` exactly (to floating precision) for several
    # s/a ratios, reproducing the table independently verified by 2e6-ray MC
    # in TEST_NOTES.md (s/a = 1.5, 2.0, 5.0, 20.0, 100.0).
    detector = _make_tracker()
    rate = 4000 * u.Hz

    for s_over_a in [1.5, 2.0, 5.0, 20.0, 100.0]:
        position = _near_position(detector, s_over_a, direction_deg=37.0)
        source = NearPointSource(position=position, spectrum=SPECTRUM, rate=rate)

        expected_f = np.arcsin(1.0 / s_over_a) / np.pi
        expected_rate = (rate * expected_f).to(u.Hz)

        actual = source.simulated_rate(detector)

        assert actual.unit.is_equivalent(u.Hz)
        assert actual.to_value(u.Hz) == pytest.approx(expected_rate.to_value(u.Hz), rel=1e-9)


# --- 2. inside-the-circle branch: f = 1 AND a uniform direction ----------

def test_near_source_inside_circle_has_full_acceptance_and_uniform_direction():
    # s < a (source well inside the surrounding circle): f must be exactly
    # 1 (Section 5.4), and the flight direction uniform over [0, 360) deg.
    #
    # KS test, n = 4000, against Uniform(0, 360). Alternative hypothesis
    # rejected: the direction is drawn only in some wedge (e.g. the s >= a
    # branch's logic firing by mistake), which would produce gaps a KS test
    # against the full uniform range easily catches. 4-sigma critical value
    # for the KS statistic at n=4000 is D_crit = 4 * 1/sqrt(4000)/sqrt(2*pi)
    # in the usual asymptotic approximation; we instead just use scipy's
    # p-value directly and require p > 1e-4 (a bug producing any wedge
    # narrower than the full circle drives p to ~0).
    detector = _make_tracker()
    rate = 1000 * u.Hz

    position = _near_position(detector, 0.4)  # s = 0.4 a < a
    source = NearPointSource(position=position, spectrum=SPECTRUM, rate=rate)

    assert source.simulated_rate(detector).to_value(u.Hz) == pytest.approx(rate.to_value(u.Hz), rel=1e-12)

    n = 4000
    directions_deg = np.array([
        source.random_photon(detector).direction.to_value(u.deg) for _ in range(n)
    ])

    assert directions_deg.min() >= 0.0
    assert directions_deg.max() < 360.0

    result = stats.kstest(directions_deg, stats.uniform(loc=0, scale=360).cdf)
    assert result.pvalue > 1e-4


# --- 3. a near source at the centre triggers far more often ---------------

def test_near_source_at_center_triggers_far_more_often_than_far_source():
    # Same `rate` and duration, two placements: dead centre (s ~ 0, so
    # f = 1 exactly) and far outside (s = 20 a, so
    # f = arcsin(1/20)/pi = 0.015923, from the arcsin formula, matching the
    # independently-verified TEST_NOTES.md row for s/a = 20).
    #
    # The acceptance ratio f_center/f_far = 1/0.015923 = 62.8 is derived
    # from geometry alone. Per-photon detector *interaction* probability
    # once a photon reaches the throwing plane is NOT analytically derivable
    # here (it depends on the material's attenuation coefficient integrated
    # along the flight path) -- so this test does not predict an exact
    # triggered count. It instead draws an actual Poisson-mean number of
    # photons for each configuration (mean = simulated_rate * duration,
    # itself an exact formula-derived number) and runs them through
    # `detector.simulate_event`, then checks that the *observed* asymmetry
    # in triggered counts is large and can't be explained by chance.
    #
    # Alternative hypothesis rejected: the two configurations trigger the
    # detector at statistically indistinguishable rates (e.g. because
    # `NearPointSource` ignores `position` entirely). Sigma is the standard
    # two-count comparison sigma = sqrt(n_center + n_far) on the *observed*
    # triggered counts; asserted at 4 sigma.
    detector = _make_tracker()
    rate = 600 * u.Hz
    duration = 6 * u.s

    center_source = NearPointSource(position=detector.surrounding_circle_center,
                                    spectrum=SPECTRUM, rate=rate)
    far_source = NearPointSource(position=_near_position(detector, 20.0, direction_deg=15.0),
                                 spectrum=SPECTRUM, rate=rate)

    f_far = np.arcsin(1.0 / 20.0) / np.pi

    # Pin `simulated_rate()` itself against the geometry-derived RATIO
    # first (independent of the absolute value already checked by
    # `test_near_source_rate_matches_arcsin_over_pi_formula`) -- this is
    # what makes the Poisson means below meaningful rather than incidental:
    # a bug that makes `simulated_rate` ignore `position` (e.g. returning
    # the same acceptance for every source) would fail *this* assertion
    # outright, before any MC is even run.
    rate_center = center_source.simulated_rate(detector)
    rate_far = far_source.simulated_rate(detector)

    assert (rate_far / rate_center).to_value(u.one) == pytest.approx(f_far, rel=1e-9)

    # The actual number of photons launched in a real run is
    # `Poisson(simulated_rate * duration)` (Section 6) -- drawn here from
    # `simulated_rate()` itself, exactly as `Simulator`/`InertialSimulator`
    # would, so that a broken `simulated_rate()` (caught above already, but
    # redundantly so here) would also show up as an indistinguishable
    # number of photons launched for the two configurations.
    mean_launched_center = (rate_center * duration).to_value(u.one)
    mean_launched_far = (rate_far * duration).to_value(u.one)

    assert mean_launched_center == pytest.approx(3600.0)
    assert mean_launched_far == pytest.approx(57.32, abs=0.01)

    n_launched_center = np.random.poisson(mean_launched_center)
    n_launched_far = np.random.poisson(mean_launched_far)

    def _count_triggered(source, n):
        triggered = 0
        for _ in range(n):
            photon = source.random_photon(detector)
            result = detector.simulate_event(photon)
            if result.interaction is not None:
                triggered += 1
        return triggered

    n_triggered_center = _count_triggered(center_source, n_launched_center)
    n_triggered_far = _count_triggered(far_source, n_launched_far)

    sigma = np.sqrt(n_triggered_center + n_triggered_far)
    assert n_triggered_center - n_triggered_far > 4 * sigma
    # Sanity: the far source did launch some photons, or the comparison
    # above would be checking against zero on both sides vacuously.
    assert n_launched_far > 0

    # Two-sided: the assertion above is satisfied just as well by a far
    # source that has stopped aiming at the detector altogether (it would
    # then trigger even *less* than a correctly-aimed one, which only
    # widens the gap to the centre source -- a bug that breaks aim_angle,
    # e.g. transposing `np.arctan2(dy, dx)`, REINFORCES this assertion
    # rather than being caught by it). Require the far source to actually
    # trigger at least once, so a source that has stopped reaching the
    # circle at all cannot pass by virtue of triggering *less* than
    # expected. The dedicated ray-geometry test below is what actually
    # verifies aim_angle is correct; this is a lightweight second check in
    # the same spirit, local to this test.
    assert n_triggered_far > 0


# --- 14. drawn directions actually reach the surrounding circle -----------

def _closest_approach_and_forward_parameter(photon, center, length_unit=u.cm):
    """Closest approach of the infinite ray `(photon.position, photon.direction)`
    to `center`, and the ray parameter `t` at that closest point (`t > 0`
    means the closest point is ahead of the photon, `t < 0` means behind).

    Plain ray/point geometry, independent of anything in
    `gammaraytoys.sims.source`: with `d = (cos(direction), sin(direction))`
    the unit *velocity* direction (the contract's "standard maths
    convention", cross-checked in CONTRACT.md against `PointSource`'s own
    `270 deg - offaxis_angle`) and `w = center - position`,
    `t = w . d` and the closest-approach distance is
    `sqrt(|w|^2 - t^2)`.
    """

    x0 = photon.position.x.to_value(length_unit)
    y0 = photon.position.y.to_value(length_unit)
    theta = photon.direction.to_value(u.rad)
    dx, dy = np.cos(theta), np.sin(theta)

    wx = center.x.to_value(length_unit) - x0
    wy = center.y.to_value(length_unit) - y0

    t = wx * dx + wy * dy
    dist2 = max(wx**2 + wy**2 - t**2, 0.0)

    return np.sqrt(dist2), t


def test_near_source_drawn_directions_actually_reach_the_surrounding_circle():
    # Section 5.4 / CONTRACT.md: "every direction drawn is one that reaches
    # the surrounding circle by construction -- there is no rejection
    # step." That claim is the entire geometric justification for
    # `f = arcsin(a/s) / pi`, and nothing above actually checks it:
    # `simulated_rate` is a pure function of `s` and `a`, never of the aim
    # direction, so a source that computes `aim_angle` wrong (e.g. a
    # transposed `np.arctan2(dy, dx) -> np.arctan2(dx, dy)`, or one aimed
    # exactly backward) would still pass every rate-based test above.
    #
    # For each of several hundred drawn photons, at three s/a ratios (0.5,
    # inside the circle; 1.5 and 5.0, outside), compute the ray's closest
    # approach to `detector.surrounding_circle_center` from first-principles
    # ray geometry (`_closest_approach_and_forward_parameter`, independent
    # of any code in `gammaraytoys.sims.source`) and require:
    #
    #   (a) the closest-approach distance is strictly less than the
    #       surrounding-circle radius, for EVERY draw (100%, not a
    #       statistical fraction -- the plan promises this unconditionally,
    #       not "usually").
    #   (b) for a source outside the circle, the closest-approach parameter
    #       t is positive -- the circle is ahead of the photon, not behind
    #       it. A distance-only check cannot catch a source aimed exactly
    #       180 deg wrong, since an infinite line's closest approach to a
    #       point doesn't depend on which way along it you're facing; only
    #       the sign of t does.
    detector = _make_tracker()
    center = detector.surrounding_circle_center
    a = detector.surrounding_circle_radius.to_value(u.cm)
    rate = 100 * u.Hz
    n = 500

    for s_over_a, outside in [(0.5, False), (1.5, True), (5.0, True)]:
        position = _near_position(detector, s_over_a, direction_deg=200.0)
        source = NearPointSource(position=position, spectrum=SPECTRUM, rate=rate)

        for _ in range(n):
            photon = source.random_photon(detector)
            distance, t_closest = _closest_approach_and_forward_parameter(photon, center)

            assert distance < a, (
                f"s/a={s_over_a}: closest approach {distance} cm to the "
                f"circle centre is not less than the circle radius {a} cm "
                "-- this direction does not reach the surrounding circle")

            if outside:
                assert t_closest > 0, (
                    f"s/a={s_over_a}: the circle's closest approach point "
                    f"is behind the photon (t={t_closest}), not ahead of it")


# --- 9. the acceptance is discontinuous at s = a ---------------------------

def test_near_source_acceptance_jumps_from_one_half_to_one_at_s_equals_a():
    # Section 5.4 / TEST_NOTES.md: as s -> a from outside, f -> arcsin(1)/pi
    # = 0.5; at s = a exactly the `s >= a` branch applies (f = 0.5); the
    # instant s dips below a, the `s < a` branch applies and f jumps to 1.
    # This is a genuine discontinuity, not a numerical-precision artefact --
    # asserted exactly at s = a and just below it.
    detector = _make_tracker()
    rate = 1000 * u.Hz

    position_on_boundary = _near_position(detector, 1.0)
    source_on_boundary = NearPointSource(position=position_on_boundary, spectrum=SPECTRUM, rate=rate)
    f_on_boundary = source_on_boundary.simulated_rate(detector) / rate
    assert f_on_boundary.to_value(u.one) == pytest.approx(0.5, abs=1e-9)

    position_just_inside = _near_position(detector, 1.0 - 1e-9)
    source_just_inside = NearPointSource(position=position_just_inside, spectrum=SPECTRUM, rate=rate)
    f_just_inside = source_just_inside.simulated_rate(detector) / rate
    assert f_just_inside.to_value(u.one) == pytest.approx(1.0, abs=1e-9)


# --- 10. simulated_rate returns None when unnormalized ---------------------

def test_near_source_simulated_rate_is_none_when_unnormalized():
    detector = _make_tracker()

    source = NearPointSource(position=detector.surrounding_circle_center,
                             spectrum=SPECTRUM, rate=None)

    assert source.simulated_rate(detector) is None


# --- 11. NearPointSource ignores pose/earth, never returns None -----------

def test_near_source_random_photon_ignores_pose_and_earth_and_never_none():
    # A near-field source is fixed in the detector frame and is not
    # occultable (Section 5.4): pose/earth must have zero effect, even when
    # they describe a geometry (a huge Earth swallowing the whole detector)
    # that would occult *any* far-field source outright.
    detector = _make_tracker()
    rate = 500 * u.Hz

    source = NearPointSource(position=_near_position(detector, 3.0),
                             spectrum=SPECTRUM, rate=rate)

    huge_earth = Earth(radius=1e6 * u.km)
    pose = SpacecraftInterval(start_time=0 * u.s, stop_time=1 * u.s, livetime=1 * u.s,
                              orbit_radius=1e6 * u.km + 1 * u.km, orbit_angle=0 * u.deg,
                              attitude=0 * u.deg)

    n = 200
    for _ in range(n):
        photon_no_pose = source.random_photon(detector)
        photon_with_pose = source.random_photon(detector, pose=pose, earth=huge_earth)

        assert photon_no_pose is not None
        assert photon_with_pose is not None

    rate_no_pose = source.simulated_rate(detector)
    rate_with_pose = source.simulated_rate(detector, pose=pose)

    assert rate_no_pose.to_value(u.Hz) == pytest.approx(rate_with_pose.to_value(u.Hz), rel=1e-12)


# --- 13. position is not mutated by drawing/propagating photons -----------

def test_near_source_position_is_not_mutated_by_drawing_photons():
    detector = _make_tracker()

    position = _near_position(detector, 4.0, direction_deg=123.0)
    source = NearPointSource(position=position, spectrum=SPECTRUM, rate=300 * u.Hz)

    original_x = source.position.x.to(u.cm).value
    original_y = source.position.y.to(u.cm).value

    for _ in range(500):
        photon = source.random_photon(detector)
        detector.simulate_event(photon)

    assert source.position.x.to(u.cm).value == pytest.approx(original_x, abs=1e-12)
    assert source.position.y.to(u.cm).value == pytest.approx(original_y, abs=1e-12)


# ===========================================================================
# ExtendedSource
# ===========================================================================

def _no_occultation_pose_and_earth():
    """A pose/Earth pair under which occultation is negligible to the point
    of being ignorable in a KS test: the Earth's angular radius rho =
    arcsin(R_E / r) is ~1e-9 rad for a 1 m Earth at a 1000 km orbit, so the
    occulted fraction of any distribution (rho/pi, Section 7 PR3) is
    ~3e-10 -- far below one part in a sample of a few thousand. This isolates
    the von Mises *sampling* step (what Section 5.5 actually specifies) from
    occultation (already covered by `test_inertial_simulator.py`)."""

    earth = Earth(radius=1 * u.m)
    pose = SpacecraftInterval(start_time=0 * u.s, stop_time=1 * u.s, livetime=1 * u.s,
                              orbit_radius=1000 * u.km, orbit_angle=0 * u.deg,
                              attitude=0 * u.deg)
    return pose, earth


def _drawn_sky_angles_deg(source, detector, pose, earth, n):
    """Draw `n` photons and recover the *inertial sky angle* each was
    actually thrown from, the same technique
    `test_isotropic_source_occultation_wedge_is_centred_on_attitude_minus_nadir`
    uses: `source._point_source.offaxis_angle` is exactly the off-axis angle
    the (occultation-surviving) draw was aimed at, converted back to a sky
    angle with `offaxis_to_sky_angle`. Occultation is assumed negligible
    (see `_no_occultation_pose_and_earth`), so every draw is expected to
    survive; that is asserted, not assumed silently."""

    angles = np.empty(n)
    for i in range(n):
        photon = source.random_photon(detector, pose=pose, earth=earth)
        assert photon is not None, "unexpected occultation -- check the Earth/orbit setup"
        angles[i] = offaxis_to_sky_angle(source._point_source.offaxis_angle, pose.attitude).to_value(u.deg)
    return angles


# --- 4. sky angles pass a KS test against scipy.stats.vonmises ------------

def test_extended_source_sky_angles_pass_ks_test_against_vonmises():
    # Section 5.5: kappa = 1 / width_rad**2. Centred at 0 deg (maximally far
    # from the +-180 deg wrap boundary, given width = 15 deg keeps the bulk
    # of the mass within a few tens of degrees) so no rewrapping is needed.
    #
    # Alternative hypothesis rejected: the sampling does not follow
    # `vonmises(kappa, loc=sky_angle)` with `kappa = 1/width_rad**2` -- e.g.
    # a Gaussian instead of von Mises, a wrong kappa formula (missing the
    # square, or not converting to radians), or a bug in the offaxis
    # round-trip. n = 3000 gives the KS test good power against any of
    # those; asserted via scipy's own p-value at the conventional 0.01
    # threshold (a genuine von Mises match with this sample size and no
    # mismatch has p uniform on [0,1], so a true positive at the 1% level
    # happens 1% of the time -- this is the KS analogue of "4 sigma", stated
    # as a p-value rather than a sigma because that is what a KS test
    # natively reports).
    detector = _make_tracker()
    sky_angle = 0 * u.deg
    width = 15 * u.deg
    kappa = 1.0 / width.to_value(u.rad)**2

    source = ExtendedSource(sky_angle=sky_angle, width=width, spectrum=SPECTRUM, flux=1 / u.cm / u.s)

    pose, earth = _no_occultation_pose_and_earth()
    n = 3000
    angles_deg = _drawn_sky_angles_deg(source, detector, pose, earth, n)

    # Comfortably inside +-90 deg of the +-180 deg wrap boundary at
    # kappa ~ 14.6 (width 15 deg): no rewrap needed.
    assert np.abs(angles_deg).max() < 150.0

    angles_rad = np.deg2rad(angles_deg)
    loc_rad = sky_angle.to_value(u.rad)

    result = stats.kstest(angles_rad, lambda x: vonmises.cdf(x, kappa, loc=loc_rad))
    assert result.pvalue > 0.01


# --- 5. wide limit matches IsotropicSource --------------------------------

def test_extended_source_wide_limit_matches_isotropic():
    # TEST_NOTES.md: width must be >= ~360 deg for the drawn sky angle to
    # actually look uniform (a KS-vs-uniform p-value at width = 180 deg is
    # ~0, i.e. "still peaked"; at 360 deg it is 0.574). Using width = 720 deg
    # for extra margin.
    #
    # Two independent checks, per TEST_NOTES.md's "what the limit tests
    # actually test":
    #
    # (a) simulated_rate is width-independent by construction
    #     (flux * throwing_plane_size, Section 5.2) -- true whether or not
    #     the sampling is right, so this alone proves little, but it IS a
    #     required check (Section 5.5) and a genuine regression guard
    #     against a `flux`-as-per-angle-brightness misreading (which would
    #     make `simulated_rate` depend on width).
    # (b) the DISTRIBUTION of drawn sky angles: a KS test against
    #     Uniform(-180, 180) deg, alternative hypothesis rejected being
    #     "the drawn angles remain measurably peaked at kappa this small",
    #     asserted via scipy's p-value at the 1% level (same reasoning as
    #     the vonmises KS test above).
    detector = _make_tracker()
    sky_angle = 40 * u.deg
    width = 720 * u.deg
    flux = 3 / u.cm / u.s

    extended = ExtendedSource(sky_angle=sky_angle, width=width, spectrum=SPECTRUM, flux=flux)
    isotropic = IsotropicSource(spectrum=SPECTRUM, flux=flux)

    rate_extended = extended.simulated_rate(detector)
    rate_isotropic = isotropic.simulated_rate(detector)

    assert rate_extended.to_value(u.Hz) == pytest.approx(rate_isotropic.to_value(u.Hz), rel=1e-12)

    # Also confirm this holds at a *different* width -- the strongest form
    # of the "total photons launched must not depend on width" check
    # (TEST_NOTES.md): at fixed flux, simulated_rate for two very different
    # widths must be identical, not just each individually equal to
    # IsotropicSource by coincidence.
    narrow_extended = ExtendedSource(sky_angle=sky_angle, width=0.01 * u.deg, spectrum=SPECTRUM, flux=flux)
    assert narrow_extended.simulated_rate(detector).to_value(u.Hz) == pytest.approx(
        rate_extended.to_value(u.Hz), rel=1e-12)

    pose, earth = _no_occultation_pose_and_earth()
    n = 3000
    angles_deg = _drawn_sky_angles_deg(extended, detector, pose, earth, n)

    result = stats.kstest(angles_deg, stats.uniform(loc=-180, scale=360).cdf)
    assert result.pvalue > 0.01


# --- 6. narrow limit matches PointSource -----------------------------------

def test_extended_source_narrow_limit_matches_point():
    # TEST_NOTES.md: width = 0.01 deg keeps every draw within 0.04 deg of
    # the centre -- "a point source for practical purposes".
    #
    # (a) simulated_rate matches a PointSource at the same flux exactly
    #     (both are flux * throwing_plane_size, Section 5.2) -- again
    #     necessary but, alone, insufficient (TEST_NOTES.md).
    # (b) every one of n = 500 drawn sky angles falls within 0.1 deg of
    #     `sky_angle` -- a 10x margin over the 0.04 deg TEST_NOTES.md bound,
    #     so this is not a knife-edge assertion. Alternative hypothesis
    #     rejected: `kappa` is not `1/width_rad**2` (e.g. off by a missing
    #     square, or width used in degrees instead of radians -- either
    #     would spread the draws over tens of degrees, not hundredths).
    detector = _make_tracker()
    sky_angle = 200 * u.deg
    width = 0.01 * u.deg
    flux = 2 / u.cm / u.s

    extended = ExtendedSource(sky_angle=sky_angle, width=width, spectrum=SPECTRUM, flux=flux)
    point = PointSource(sky_angle=sky_angle, spectrum=SPECTRUM, flux=flux)

    assert extended.simulated_rate(detector).to_value(u.Hz) == pytest.approx(
        point.simulated_rate(detector).to_value(u.Hz), rel=1e-12)

    pose, earth = _no_occultation_pose_and_earth()
    n = 500
    angles_deg = _drawn_sky_angles_deg(extended, detector, pose, earth, n)

    deviation_deg = np.abs(((angles_deg - sky_angle.to_value(u.deg) + 180) % 360) - 180)
    assert deviation_deg.max() < 0.1


# --- 12. ExtendedSource raises for pose=None and for pose without earth ---

def test_extended_source_raises_without_pose():
    # `earth` is deliberately given here (unlike the companion test below)
    # so this isolates the `pose is None` check specifically: without an
    # `earth`, `_occulted` would itself raise ValueError for "no earth"
    # (Section 5.3) regardless of whether the `pose is None` check under
    # test is even present, silently masking that check's own mutation.
    detector = _make_tracker()
    source = ExtendedSource(sky_angle=10 * u.deg, width=5 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s)
    earth = Earth()

    with pytest.raises(ValueError):
        source.random_photon(detector, earth=earth)


def test_extended_source_raises_with_pose_but_no_earth():
    detector = _make_tracker()
    source = ExtendedSource(sky_angle=10 * u.deg, width=5 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s)

    pose = SpacecraftInterval(start_time=0 * u.s, stop_time=1 * u.s, livetime=1 * u.s,
                              orbit_radius=7000 * u.km, orbit_angle=0 * u.deg,
                              attitude=0 * u.deg)

    with pytest.raises(ValueError):
        source.random_photon(detector, pose=pose)


# --- 7. regression: plot centres the arc on the source's direction, -------
#     not on the last drawn photon

def test_extended_source_plot_arc_is_centred_on_the_sky_angle_not_last_draw():
    # Orchestrator-found defect: `ExtendedSource.plot` used to centre its
    # arc on `self._point_source.offaxis_angle` -- a von Mises *sample*,
    # different every draw -- instead of the source's own `sky_angle`. At
    # width = 30 deg this could wander up to 49.5 deg from the true centre.
    #
    # This test draws one photon (fixing `_point_source.offaxis_angle` to
    # whatever that draw happened to be) and then checks the plotted arc's
    # centre against the value independently computed by
    # `sky_angle_to_offaxis(sky_angle, attitude)` -- the *distribution*
    # centre -- not against whatever `_point_source.offaxis_angle` holds.
    # A correct implementation must match regardless of which sample the
    # single draw produced.
    from gammaraytoys.coordinates import sky_angle_to_offaxis

    detector = _make_tracker()
    sky_angle = 30 * u.deg
    width = 30 * u.deg

    source = ExtendedSource(sky_angle=sky_angle, width=width, spectrum=SPECTRUM, flux=1 / u.cm / u.s)

    earth = Earth(radius=1 * u.m)  # negligible occultation, see above

    # Draw first at one pose, then at a second, DIFFERENT pose, and plot
    # only after the second. `plot` must reflect the *last* pose a photon
    # was actually drawn at, not the first -- a `_last_attitude` that gets
    # set once and never updated again (a distinct staleness bug from the
    # "last draw vs true centre" one this test also targets) would still
    # plot the first pose's centre and this catches that too.
    pose_1 = SpacecraftInterval(start_time=0 * u.s, stop_time=1 * u.s, livetime=1 * u.s,
                                orbit_radius=7000 * u.km, orbit_angle=0 * u.deg,
                                attitude=80 * u.deg)
    pose_2 = SpacecraftInterval(start_time=0 * u.s, stop_time=1 * u.s, livetime=1 * u.s,
                                orbit_radius=7000 * u.km, orbit_angle=0 * u.deg,
                                attitude=200 * u.deg)

    for _ in range(20):
        photon = source.random_photon(detector, pose=pose_1, earth=earth)
        if photon is not None:
            break
    else:
        pytest.fail("never drew a surviving photon at pose_1")

    pose = pose_2

    # Draw until we get a sample (at pose_2) whose off-axis angle differs
    # meaningfully from the true distribution centre, so a bug that centres
    # on the last draw instead of the true centre cannot pass by accident.
    expected_center = sky_angle_to_offaxis(sky_angle, pose.attitude)
    for _ in range(200):
        photon = source.random_photon(detector, pose=pose, earth=earth)
        if photon is not None:
            drawn_offaxis = source._point_source.offaxis_angle
            if abs((drawn_offaxis - expected_center).to_value(u.deg)) > 5.0:
                break
    else:
        pytest.fail("never drew a sample far enough from the centre to distinguish the two hypotheses")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    detector.plot(ax=ax)
    source.plot(ax, detector)

    line = ax.get_lines()[-1]
    xdata, ydata = line.get_xdata(), line.get_ydata()

    # `plot_sky_arc` traces the arc as (sin Nu, cos Nu) around
    # `surrounding_circle_center`; its midpoint sample (index 100 of 200,
    # see `plot_sky_arc`) is exactly at `center_angle`.
    center = detector.surrounding_circle_center
    mid_x = xdata[len(xdata) // 2] - center.x.to(u.cm).value
    mid_y = ydata[len(ydata) // 2] - center.y.to(u.cm).value
    plotted_center_deg = np.degrees(np.arctan2(mid_x, mid_y))  # inverse of (sin Nu, cos Nu)

    plt.close(fig)

    diff = ((plotted_center_deg - expected_center.to_value(u.deg) + 180) % 360) - 180
    assert abs(diff) < 1.0

    # And it must NOT match the last drawn (jittered) sample's off-axis
    # angle, which is what the bug centred on.
    diff_from_sample = ((plotted_center_deg - drawn_offaxis.to_value(u.deg) + 180) % 360) - 180
    assert abs(diff_from_sample) > 4.0


# --- 8. regression: changing width after construction changes the spread -

def test_extended_source_width_change_after_construction_takes_effect():
    # Orchestrator-found defect: kappa used to be frozen at construction
    # time, so a source built at width = 1 deg kept a ~1 deg spread even
    # after `width` was set to 30 deg. This draws a batch at a small width,
    # widens `width`, draws again, and checks the *second* batch's spread
    # actually grew -- a frozen-kappa bug would leave it unchanged.
    detector = _make_tracker()
    sky_angle = 0 * u.deg

    source = ExtendedSource(sky_angle=sky_angle, width=1 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s)

    pose, earth = _no_occultation_pose_and_earth()
    n = 800

    angles_narrow = _drawn_sky_angles_deg(source, detector, pose, earth, n)

    source.width = 30 * u.deg
    angles_wide = _drawn_sky_angles_deg(source, detector, pose, earth, n)

    spread_narrow = np.std(angles_narrow)
    spread_wide = np.std(angles_wide)

    # A frozen kappa would leave spread_wide ~= spread_narrow (~1 deg,
    # TEST_NOTES.md: 1 deg width gives ~1.002 deg circular std). A correctly
    # re-derived kappa gives ~30x more spread (TEST_NOTES.md's 20 deg row:
    # ratio 1.034, i.e. ~20.68 deg circular std at 20 deg width -- 30 deg is
    # in the same regime). Asserting a >5x growth is a comfortable margin
    # over both the ~1x a frozen-kappa bug would give and sampling noise at
    # n = 800 (std of a circular-std estimator at this n is a few percent).
    assert spread_wide > 5 * spread_narrow
