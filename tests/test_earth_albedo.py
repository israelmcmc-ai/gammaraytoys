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
from gammaraytoys.coordinates import offaxis_to_sky_angle
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

    # Item 15: a source that ignored `pose.orbit_radius` (e.g. used a fixed
    # altitude) would give the same rate at every altitude above. Instead,
    # the rate must be strictly decreasing with altitude, and the 100 km /
    # 100 000 km ratio must match the closed form's OWN ratio exactly (not
    # merely "be different") -- ruling out a source that depends on `r`
    # through some other, wrong channel that happens to still vary.
    assert np.all(np.diff(rates_hz) < 0)

    expected_ratio = (np.arcsin(RE / (RE + 100.0)) / np.arcsin(RE / (RE + 100000.0)))
    assert expected_ratio == pytest.approx(23.278, abs=0.01)  # TEST_NOTES.md's two rows
    assert rates_hz[0] / rates_hz[-1] == pytest.approx(expected_ratio, rel=1e-9)


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
    # Trap 1 / Section 8.1: occultable = False must survive a REAL
    # InertialSimulator run over a full circular orbit, not just the bare
    # property. Same geometry as
    # `tests/test_inertial_simulator.py::test_circular_orbit_occults_a_point_source_for_the_analytic_rho_over_pi`
    # (EARTH_RADIUS = 6371 km, ORBIT_RADIUS = 6771 km, rho/pi = 0.3900): over
    # a full orbit an ordinary far-field PointSource is occulted for a
    # fraction rho/pi of its photons, while the EarthAlbedoSource, mixed
    # into the SAME run, must lose none of its mu = 1200 mean at all.
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
