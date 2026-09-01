"""Tests for `InertialSimulator` (plan Section 6) and for occultation.

Sizing note. Every statistical assertion below states the Poisson (or
binomial) sigma it is built on and asserts at **4 sigma**, one wider than the
plan's "within 3 sigma", so that a passing implementation cannot fail by
chance on a different random realisation. In each case the alternative
hypothesis the test is meant to reject (no occultation, a deterministic
count, livetime ignored, ...) is separately noted and sits at 15 sigma or
more.

Every expected value is derived from the plan's formulas or from geometry:

    mu           = flux * throwing_plane_size * livetime        (Sections 5.2, 6)
    rho          = arcsin(R_E / r)                              (Section 4.5)
    occulted     <=> |wrap(lambda - (theta + 180 deg))| < rho   (Section 4.5)
    occulted fraction over a full circular orbit = rho / pi     (Section 7, PR 3)

None of them was read back out of the implementation.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import G, M_earth
from scipy import stats

from gammaraytoys import ToyTracker2D
from gammaraytoys.sims import (Earth, InertialSimulator, IsotropicSource,
                               MonoenergeticSpectrum, PointSource,
                               SimpleTraditionalReconstructor, SpacecraftHistory,
                               SpacecraftInterval, SpinPointing, ZenithPointing)


# --- geometry constants, fixed here rather than taken from astropy -------
#
# An explicit Earth radius keeps rho hand-computable:
#     rho = arcsin(6371 / 6771) = 1.226118 rad = 70.2513 deg
#     rho / pi = 0.390281
EARTH_RADIUS = 6371.0 * u.km
ORBIT_RADIUS = 6771.0 * u.km

RHO_RAD = np.arcsin(EARTH_RADIUS.to_value(u.km) / ORBIT_RADIUS.to_value(u.km))
OCCULTED_FRACTION = RHO_RAD / np.pi


def _wrap180(angle_deg):
    """Wrap a plain-float angle in degrees to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _orbital_period(semi_major_axis):
    """`2 pi sqrt(a^3 / mu)`, straight from astropy's constants."""
    return (2 * np.pi * np.sqrt(semi_major_axis**3 / (G * M_earth))).to(u.s)


def _make_earth():
    return Earth(radius=EARTH_RADIUS)


def _make_tracker():
    """Same detector as the `tracker` fixture in conftest, built locally so
    module-scoped fixtures (which pytest sets up before function-scoped ones)
    can use it too."""

    return ToyTracker2D(material='Ge',
                        layer_length=16 * u.cm,
                        layer_positions=[0, 5, 10, 20, 25, 30] * u.mm,
                        layer_thickness=5 * u.mm,
                        energy_resolution=0.01,
                        energy_threshold=20 * u.keV)


def _make_history(duration, n_intervals, earth,
                  observation_strategy=None, livetime_fraction=1.0):
    """A circular orbit at `ORBIT_RADIUS`, tiled into `n_intervals` equal
    intervals over `duration`.

    With `eccentricity = 0` and `argument_of_periapsis = 0` the orbit angle
    is exactly `theta(t) = 360 deg * t / period`, starting at 0, which is
    what every geometric expectation in this file is worked out from.
    """

    if observation_strategy is None:
        observation_strategy = ZenithPointing()

    return SpacecraftHistory.from_elliptical_orbit(
        semi_major_axis=ORBIT_RADIUS,
        eccentricity=0.0,
        earth=earth,
        observation_strategy=observation_strategy,
        time_step=duration / n_intervals,
        duration=duration,
        livetime_fraction=livetime_fraction)


def _flux_for_expected_counts(mu, detector, livetime):
    """Invert `mu = flux * throwing_plane_size * livetime` (Sections 5.2, 6)."""

    return (mu / (detector.throwing_plane_size * livetime)).to(1 / u.cm / u.s)


def _visible_livetime_fraction(history, sky_angle):
    """Fraction of the history's livetime during which a far-field source at
    `sky_angle` is *not* behind the Earth.

    Computed here directly from Section 4.5's geometry -- nadir at
    `theta + 180 deg`, blocked when within `rho = arcsin(R_E/r)` of it --
    over the history's own frozen per-interval poses, so it accounts exactly
    for the discretisation the simulator sees. It never calls `Earth` or the
    simulator.
    """

    sky_angle_deg = sky_angle.to_value(u.deg)

    visible = 0.0
    total = 0.0

    for interval in history:
        radius_km = interval.orbit_radius.to_value(u.km)
        rho_deg = np.degrees(np.arcsin(EARTH_RADIUS.to_value(u.km) / radius_km))

        nadir_deg = interval.orbit_angle.to_value(u.deg) + 180.0

        livetime = interval.livetime.to_value(u.s)
        total += livetime

        if abs(_wrap180(sky_angle_deg - nadir_deg)) >= rho_deg:
            visible += livetime

    return visible / total


def _run(detector, sources, history, earth):
    """Run an `InertialSimulator` to completion and return the event list."""

    simulator = InertialSimulator(detector=detector,
                                  sources=sources,
                                  reconstructor=SimpleTraditionalReconstructor(),
                                  spacecraft_history=history,
                                  earth=earth)

    return list(simulator.run_events())


class _NonOccultablePointSource(PointSource):
    """A `PointSource` that opts out of occultation.

    Stands in for PR 5's `EarthAlbedoSource`, which does not exist yet:
    Section 8.1 requires `occultable = False` to switch the per-photon
    rejection off entirely.
    """

    @property
    def occultable(self):
        return False

    @occultable.setter
    def occultable(self, value):
        # A no-op setter, so this subclass works whether the base declares
        # `occultable` as a read-only property or assigns it per instance.
        pass


# --- the shared "clean" run ----------------------------------------------
#
# One arc of a circular orbit, chosen so that the source is provably never
# occulted, so the expected count is the bare `flux * 2a * livetime` with no
# geometric correction. Half an orbit starting at theta = 0 puts nadir in
# [180, 360) deg while the source sits at lambda = 90 deg, so the smallest
# source-to-nadir separation anywhere in the run is 90 deg -- comfortably
# outside rho = 70.25 deg.

CLEAN_MU = 1500.0
CLEAN_NINTERVALS = 250
CLEAN_SKY_ANGLE = 90 * u.deg


@pytest.fixture(scope='module')
def clean_run():
    # Seeded here explicitly: pytest sets module-scoped fixtures up *before*
    # the function-scoped autouse `_seed_random`, so this fixture would
    # otherwise be the one unseeded thing in the suite.
    np.random.seed(31337)

    earth = _make_earth()
    detector = _make_tracker()

    duration = _orbital_period(ORBIT_RADIUS) / 2
    history = _make_history(duration, CLEAN_NINTERVALS, earth)

    flux = _flux_for_expected_counts(CLEAN_MU, detector, duration)

    source = PointSource(sky_angle=CLEAN_SKY_ANGLE,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=flux)

    events = _run(detector, source, history, earth)

    return dict(events=events, history=history, detector=detector,
                source=source, earth=earth, duration=duration, flux=flux)


def test_clean_run_source_is_never_occulted(clean_run):
    # The premise of the two tests below, asserted from geometry alone so
    # that a failure there cannot be blamed on occultation.
    fraction = _visible_livetime_fraction(clean_run['history'], CLEAN_SKY_ANGLE)

    assert fraction == 1.0


def test_total_counts_match_flux_times_throwing_plane_size_times_livetime(clean_run):
    # Section 6: mu = simulated_rate * livetime, summed over intervals, and
    # simulated_rate = flux * throwing_plane_size (Section 5.2). With a
    # constant flux and no occultation that is just
    #     mu = flux * 2a * total_livetime
    # which is how `flux` was chosen, so the expectation is CLEAN_MU exactly.
    #
    # sigma = sqrt(1500) = 38.7; the 4-sigma window is +-155 (10%).
    # A missing throwing-plane factor, or livetime instead of span, would
    # move the answer by hundreds of sigma.
    detector = clean_run['detector']
    duration = clean_run['duration']

    expected = (clean_run['flux'] * detector.throwing_plane_size * duration).to_value('')

    assert expected == pytest.approx(CLEAN_MU, rel=1e-9)

    n_events = len(clean_run['events'])
    sigma = np.sqrt(expected)

    assert abs(n_events - expected) < 4 * sigma


def test_per_interval_counts_are_poisson_distributed(clean_run):
    # Section 6 insists the per-(source, interval) count is *always* a
    # Poisson draw, never a rounded expectation. With 250 identical
    # intervals the counts are i.i.d. Poisson(6), so the dispersion
    # statistic  D = sum (n_i - nbar)^2 / nbar  follows chi^2 with 249
    # degrees of freedom: mean 249, sigma sqrt(2*249) = 22.3.
    # The 4-sigma window is [160, 338]. A deterministic round(mu) per
    # interval would give D = 0.
    history = clean_run['history']
    events = clean_run['events']

    edges = [interval.start_time.to_value(u.s) for interval in history]
    edges.append(list(history)[-1].stop_time.to_value(u.s))
    edges = np.array(edges)

    times = np.array([event[0].to_value(u.s) for event in events])

    assert times.min() >= edges[0]
    assert times.max() < edges[-1]

    counts, _ = np.histogram(times, bins=edges)

    assert counts.size == CLEAN_NINTERVALS

    mean = counts.mean()
    assert mean == pytest.approx(CLEAN_MU / CLEAN_NINTERVALS, rel=0.15)

    dispersion = np.sum((counts - mean)**2) / mean

    dof = CLEAN_NINTERVALS - 1
    sigma = np.sqrt(2 * dof)

    assert abs(dispersion - dof) < 4 * sigma


# --- occultation ---------------------------------------------------------

def test_circular_orbit_occults_a_point_source_for_the_analytic_rho_over_pi():
    # Section 7, PR 3: "one in the orbital plane is occulted for a fraction
    # rho/pi of the orbit". In flatland every far-field source lies in the
    # orbital plane, so this is the generic case: over a full circular orbit
    # nadir sweeps all 360 deg uniformly in time, and the source is blocked
    # whenever nadir passes within rho of it -- an arc of 2 rho out of 2 pi,
    # i.e. a fraction rho/pi = 0.3903.
    earth = _make_earth()
    detector = _make_tracker()

    period = _orbital_period(ORBIT_RADIUS)
    history = _make_history(period, 360, earth)

    sky_angle = 0 * u.deg

    # (i) pure geometry, no simulator: the history's own visible fraction
    #     must reproduce the analytic 1 - rho/pi. The frozen-pose
    #     discretisation costs at most one interval (1 deg of orbit) at each
    #     of the two limb crossings, i.e. 2/360 = 0.006 in fraction.
    visible_fraction = _visible_livetime_fraction(history, sky_angle)

    assert visible_fraction == pytest.approx(1 - OCCULTED_FRACTION, abs=0.01)

    # (ii) the simulator must reject exactly that fraction of its photons.
    #      mu = 1500 photons drawn from the *unocculted* mean; expected
    #      survivors 1500 * 0.6097 = 914.6, sigma = 30.2 (thinned Poisson is
    #      still Poisson), 4 sigma = +-121.  No occultation at all would sit
    #      at 1500, which is 19 sigma away.
    mu = 1500.0
    flux = _flux_for_expected_counts(mu, detector, period)

    source = PointSource(sky_angle=sky_angle,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=flux)

    events = _run(detector, source, history, earth)

    expected = mu * visible_fraction
    sigma = np.sqrt(expected)

    assert abs(len(events) - expected) < 4 * sigma


def test_isotropic_source_loses_the_earth_disc_fraction_of_its_photons():
    # An isotropic sky is uniform over 2 pi, and the Earth covers 2 rho of
    # it, so the same rho/pi fraction is lost -- but for a completely
    # different reason (the source spans the sky rather than the pose
    # sweeping past it), and at a single frozen pose rather than over an
    # orbit. `IsotropicSource` "comes along for free" per PR 3.
    #
    # mu = 800, expected survivors 800 * 0.6097 = 487.8, sigma = 22.1,
    # 4 sigma = +-88. No occultation would sit at 800, i.e. 14 sigma away.
    earth = _make_earth()
    detector = _make_tracker()

    duration = 1000 * u.s
    history = _make_history(duration, 1, earth)

    mu = 800.0
    flux = _flux_for_expected_counts(mu, detector, duration)

    source = IsotropicSource(spectrum=MonoenergeticSpectrum(1 * u.MeV), flux=flux)

    events = _run(detector, source, history, earth)

    expected = mu * (1 - OCCULTED_FRACTION)
    sigma = np.sqrt(expected)

    assert abs(len(events) - expected) < 4 * sigma


def test_isotropic_source_occultation_wedge_is_centred_on_attitude_minus_nadir():
    # F2 (PR3 review): the test above only checks *how many* photons an
    # IsotropicSource loses to the Earth, at a pose (theta = 0,
    # ZenithPointing => A = 0) where the occulted wedge happens to be
    # symmetric about Nu = 0. A source that forgot to convert its
    # detector-frame draw to an inertial sky angle before testing occultation
    # -- i.e. tested `offaxis_angle` itself against the Earth instead of
    # `offaxis_to_sky_angle(offaxis_angle, attitude)` -- would produce a
    # wedge that is *also* symmetric about Nu = 0 at that same pose, so the
    # count would come out identical and the bug would go unnoticed. This
    # test instead pins a non-trivial attitude and checks *where* the wedge
    # falls, sample by sample.
    #
    # Geometry, worked out by hand (never read back from the implementation):
    # a pose at orbit_angle = 0 deg and attitude A = 50 deg gives
    #     nadir = orbit_angle + 180 deg = 180 deg
    #     rho   = arcsin(R_E / r) = RHO_RAD (module constant) = 70.2074 deg
    # A source is occulted iff its inertial sky angle lambda falls within rho
    # of nadir. Since lambda = A - Nu (mod 360), that sky-angle wedge maps to
    # an off-axis-angle wedge centred at
    #     Nu = A - nadir = 50 - 180 = -130 deg
    # of half-width rho (full width 2*rho = 140.4 deg): the map
    # lambda -> Nu = A - lambda is a reflection, which preserves interval
    # widths and just relabels the centre.
    earth = _make_earth()
    detector = _make_tracker()

    orbit_angle = 0 * u.deg
    attitude = 50 * u.deg

    pose = SpacecraftInterval(start_time=0 * u.s, stop_time=1 * u.s, livetime=1 * u.s,
                              orbit_radius=ORBIT_RADIUS, orbit_angle=orbit_angle,
                              attitude=attitude)

    expected_center_deg = _wrap180(attitude.to_value(u.deg) - orbit_angle.to_value(u.deg) - 180.0)
    expected_halfwidth_deg = np.degrees(RHO_RAD)

    assert expected_center_deg == pytest.approx(-130.0)
    assert 2 * expected_halfwidth_deg == pytest.approx(140.41480693771166)

    source = IsotropicSource(spectrum=MonoenergeticSpectrum(1 * u.MeV), flux=1 / u.cm / u.s)

    n_samples = 3000
    offaxis_samples = np.empty(n_samples)
    occluded_samples = np.empty(n_samples, dtype=bool)

    for i in range(n_samples):
        photon = source.random_photon(detector, pose, earth)

        # IsotropicSource re-aims its single reusable PointSource to the
        # off-axis angle it just drew, whether or not that draw survived
        # occultation (see IsotropicSource.random_photon), so this is
        # exactly the Nu that was tested.
        offaxis_samples[i] = source._point_source.offaxis_angle.to_value(u.deg)
        occluded_samples[i] = photon is None

    # At least some photons must have been thrown and some occulted, or the
    # comparison below would be vacuous.
    assert 0 < np.count_nonzero(occluded_samples) < n_samples

    nu_relative_to_wedge = _wrap180(offaxis_samples - expected_center_deg)
    predicted_occluded = np.abs(nu_relative_to_wedge) < expected_halfwidth_deg

    # Exact, per-sample agreement -- not a statistical count -- between the
    # hand-worked wedge and what the real code actually rejected.
    assert np.array_equal(occluded_samples, predicted_occluded)


# --- occultable = False --------------------------------------------------
#
# A sixth of an orbit sweeps theta from 0 to 60 deg, so nadir stays in
# [180, 240) deg. A source at lambda = 180 deg is then never further than
# 60 deg from nadir, and rho = 70.25 deg, so it is behind the Earth for the
# whole run.

FULLY_OCCULTED_SKY_ANGLE = 180 * u.deg
FULLY_OCCULTED_MU = 300.0


def _fully_occulted_setup():
    earth = _make_earth()
    detector = _make_tracker()

    duration = _orbital_period(ORBIT_RADIUS) / 6
    history = _make_history(duration, 40, earth)

    flux = _flux_for_expected_counts(FULLY_OCCULTED_MU, detector, duration)

    return earth, detector, history, flux


def test_a_source_behind_the_earth_for_the_whole_run_yields_nothing():
    earth, detector, history, flux = _fully_occulted_setup()

    # Premise, from geometry alone.
    assert _visible_livetime_fraction(history, FULLY_OCCULTED_SKY_ANGLE) == 0.0

    source = PointSource(sky_angle=FULLY_OCCULTED_SKY_ANGLE,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=flux)

    events = _run(detector, source, history, earth)

    assert len(events) == 0


def test_occultable_false_disables_the_rejection_entirely():
    # Section 8.1: `occultable = False` must switch the per-photon rejection
    # off, otherwise PR 5's albedo (whose photons all arrive from the Earth's
    # direction) would be rejected wholesale.
    #
    # Same geometry, same flux, same livetime as the test above, which yields
    # exactly zero events. Here the full mu = 300 must come through:
    # sigma = 17.3, 4 sigma = +-69.
    earth, detector, history, flux = _fully_occulted_setup()

    source = _NonOccultablePointSource(sky_angle=FULLY_OCCULTED_SKY_ANGLE,
                                       spectrum=MonoenergeticSpectrum(1 * u.MeV),
                                       flux=flux)

    assert source.occultable is False

    events = _run(detector, source, history, earth)

    sigma = np.sqrt(FULLY_OCCULTED_MU)

    assert abs(len(events) - FULLY_OCCULTED_MU) < 4 * sigma


# --- the sky stands still ------------------------------------------------

def test_the_sky_stands_still_while_the_spacecraft_spins():
    # The headline test of PR 3. A source pinned at a fixed inertial
    # `sky_angle` must not drift on the sky as the spacecraft turns: for
    # every photon it throws,
    #     A(t) - Nu == lambda   (mod 360 deg)
    # where Nu is recovered from the thrown photon exactly as
    # `Simulator.run_binned` already does it, `Nu = 270 deg - direction`.
    #
    # This is an identity, not a statistical statement, so it is asserted to
    # 1e-6 deg on every single event.
    earth = _make_earth()
    detector = _make_tracker()

    period = _orbital_period(ORBIT_RADIUS)

    # 3 deg/s over a ~92 min orbit is ~16600 deg of attitude: the history
    # hands over wildly unwrapped attitudes, which the transform must wrap.
    history = _make_history(period, 200, earth,
                            observation_strategy=SpinPointing(rate=3 * u.deg / u.s))

    sky_angle = 137 * u.deg

    mu = 250.0
    flux = _flux_for_expected_counts(mu, detector, period)

    source = PointSource(sky_angle=sky_angle,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=flux)

    events = _run(detector, source, history, earth)

    intervals = list(history)

    # Not vacuous: with ~39% occulted we still expect ~150 events.
    assert len(events) > 60

    attitudes = []
    residuals = []

    for time, _source, sim_event, _reco in events:
        t_s = time.to_value(u.s)

        matching = [interval for interval in intervals
                    if interval.start_time.to_value(u.s) <= t_s < interval.stop_time.to_value(u.s)]

        assert len(matching) == 1, f"time {time} falls in {len(matching)} intervals"

        attitude_deg = matching[0].attitude.to_value(u.deg)
        offaxis_deg = 270.0 - sim_event.direction.to_value(u.deg)

        attitudes.append(_wrap180(attitude_deg))
        residuals.append(_wrap180(attitude_deg - offaxis_deg - sky_angle.to_value(u.deg)))

    residuals = np.array(residuals)

    assert np.max(np.abs(residuals)) < 1e-6

    # The attitude really did sweep the whole circle, so the invariance
    # above is not an accident of a nearly-constant pointing.
    attitudes = np.sort(np.array(attitudes))
    assert attitudes[-1] - attitudes[0] > 300.0


# --- livetime -------------------------------------------------------------

def test_livetime_scales_the_counts_but_not_the_timestamps():
    # Section 6: "Timestamps are uniform over the full interval span, not
    # over the live part. Livetime only scales the count."
    earth = _make_earth()
    detector = _make_tracker()

    duration = _orbital_period(ORBIT_RADIUS) / 2
    mu_full = 800.0

    flux = _flux_for_expected_counts(mu_full, detector, duration)

    def run(livetime_fraction):
        history = _make_history(duration, 200, earth,
                                livetime_fraction=livetime_fraction)

        source = PointSource(sky_angle=CLEAN_SKY_ANGLE,
                             spectrum=MonoenergeticSpectrum(1 * u.MeV),
                             flux=flux)

        assert _visible_livetime_fraction(history, CLEAN_SKY_ANGLE) == 1.0

        return history, _run(detector, source, history, earth)

    history_full, events_full = run(1.0)
    history_half, events_half = run(0.5)

    n_full = len(events_full)
    n_half = len(events_half)

    # Counts: halving the livetime halves mu. Var(2*N_half - N_full) =
    # 4*(mu/2) + mu = 3 mu = 2400, sigma = 49, 4 sigma = +-196.
    # If livetime were ignored entirely, 2*N_half - N_full would sit at
    # mu = 800, i.e. 16 sigma away.
    assert abs(2 * n_half - n_full) < 4 * np.sqrt(3 * mu_full)

    # ... and the run really is half as long in livetime while spanning the
    # same wall-clock time.
    assert history_half.total_livetime.to_value(u.s) == pytest.approx(
        0.5 * history_full.total_livetime.to_value(u.s))

    # Timestamps: still uniform over the *full* span of the history, not
    # bunched into the first half of each interval.
    span_start = 0.0
    span_stop = duration.to_value(u.s)

    for events in (events_full, events_half):
        times = np.array([event[0].to_value(u.s) for event in events])

        assert times.min() >= span_start
        assert times.max() < span_stop

        pvalue = stats.kstest(times, 'uniform',
                              args=(span_start, span_stop - span_start)).pvalue

        # A uniform p-value, so this threshold flakes once in 10000 runs;
        # timestamps drawn over the *live* part of each interval instead of
        # the full span would land at p ~ 1e-40.
        assert pvalue > 1e-4

    # A concrete version of the same statement: with a 50% livetime the
    # second half of every interval must still be populated. Deadtime parked
    # at the end of each interval would empty it.
    dt = span_stop / 200
    times_half = np.array([event[0].to_value(u.s) for event in events_half])
    phase = (times_half % dt) / dt

    late = np.count_nonzero(phase >= 0.5)

    # Binomial(n_half, 0.5): sigma = sqrt(n/4) ~ 10 for n ~ 400,
    # 4 sigma = +-40 around n/2 = 200.
    assert abs(late - n_half / 2) < 4 * np.sqrt(n_half / 4)


# --- multiple sources -----------------------------------------------------

def test_counts_split_between_sources_in_proportion_to_their_rates():
    # Section 6 runs the Poisson draw per (source, interval), so each
    # source's total is Poisson about its own flux * 2a * livetime,
    # independently of the others.
    earth = _make_earth()
    detector = _make_tracker()

    duration = _orbital_period(ORBIT_RADIUS) / 2
    history = _make_history(duration, 100, earth)

    mu_bright = 300.0
    mu_faint = 100.0

    spec = MonoenergeticSpectrum(1 * u.MeV)

    bright = PointSource(sky_angle=CLEAN_SKY_ANGLE, spectrum=spec,
                         flux=_flux_for_expected_counts(mu_bright, detector, duration))
    faint = PointSource(sky_angle=CLEAN_SKY_ANGLE, spectrum=spec,
                        flux=_flux_for_expected_counts(mu_faint, detector, duration))

    assert _visible_livetime_fraction(history, CLEAN_SKY_ANGLE) == 1.0

    events = _run(detector, [bright, faint], history, earth)

    n_bright = sum(1 for event in events if event[1] is bright)
    n_faint = sum(1 for event in events if event[1] is faint)

    assert n_bright + n_faint == len(events)

    # sigma = 17.3 and 10.0; 4 sigma = +-69 and +-40.
    assert abs(n_bright - mu_bright) < 4 * np.sqrt(mu_bright)
    assert abs(n_faint - mu_faint) < 4 * np.sqrt(mu_faint)


# --- tstart / tstop -------------------------------------------------------

def test_tstart_and_tstop_narrow_the_run_to_that_window():
    # Section 6: "Termination is the .ori range, optionally narrowed by
    # tstart/tstop." The window below falls exactly on row boundaries --
    # rows 50 and 150 of 200 equal intervals -- so it selects half the
    # livetime under any reasonable reading of a partially-covered interval.
    earth = _make_earth()
    detector = _make_tracker()

    duration = _orbital_period(ORBIT_RADIUS) / 2
    n_intervals = 200
    history = _make_history(duration, n_intervals, earth)

    mu_full = 600.0
    flux = _flux_for_expected_counts(mu_full, detector, duration)

    source = PointSource(sky_angle=CLEAN_SKY_ANGLE,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=flux)

    assert _visible_livetime_fraction(history, CLEAN_SKY_ANGLE) == 1.0

    tstart = duration / 4
    tstop = 3 * duration / 4

    simulator = InertialSimulator(detector=detector,
                                  sources=source,
                                  reconstructor=SimpleTraditionalReconstructor(),
                                  spacecraft_history=history,
                                  earth=earth)

    events = list(simulator.run_events(tstart=tstart, tstop=tstop))

    times = np.array([event[0].to_value(u.s) for event in events])

    assert np.all(times >= tstart.to_value(u.s))
    # `np.random.uniform` is half-open on the high end (see `_poses`'s
    # `np.random.uniform(start, stop)`), so a timestamp can never land on
    # tstop itself.
    assert np.all(times < tstop.to_value(u.s))

    # Half the livetime, so half the counts: mu = 300, sigma = 17.3,
    # 4 sigma = +-69. Ignoring the window would give 600, i.e. 17 sigma away.
    expected = mu_full / 2
    assert abs(len(events) - expected) < 4 * np.sqrt(expected)


def test_tstart_and_tstop_rescale_livetime_for_a_partially_clipped_interval():
    # F3 (PR3 review): the test above picks a window that falls exactly on
    # interval boundaries, so no interval is ever partly clipped and the
    # `live = interval.livetime * (hi - lo) / (stop - start)` rescale in
    # `_poses` (Section 6) never actually gets exercised. This test instead
    # picks a window that clips a single interval in half, so an
    # implementation that used the interval's *unrescaled* livetime for a
    # partial overlap would be caught.
    #
    # 10 intervals of 200 s each (2000 s total); tstart = 50 s, tstop = 150 s
    # falls entirely inside interval 0 = [0, 200) s, clipping it to a 100 s
    # overlap out of its 200 s span. No other interval overlaps the window at
    # all. With mu_full drawn against the *full* 2000 s duration, the
    # correctly rescaled expectation is mu_full * (100 / 2000) = mu_full / 20.
    earth = _make_earth()
    detector = _make_tracker()

    n_intervals = 10
    interval_span = 200 * u.s
    duration = n_intervals * interval_span

    mu_full = 4000.0
    flux = _flux_for_expected_counts(mu_full, detector, duration)

    history = _make_history(duration, n_intervals, earth)

    source = PointSource(sky_angle=CLEAN_SKY_ANGLE,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=flux)

    assert _visible_livetime_fraction(history, CLEAN_SKY_ANGLE) == 1.0

    tstart = 50 * u.s
    tstop = 150 * u.s

    simulator = InertialSimulator(detector=detector,
                                  sources=source,
                                  reconstructor=SimpleTraditionalReconstructor(),
                                  spacecraft_history=history,
                                  earth=earth)

    events = list(simulator.run_events(tstart=tstart, tstop=tstop))

    times = np.array([event[0].to_value(u.s) for event in events])

    assert np.all(times >= tstart.to_value(u.s))
    assert np.all(times < tstop.to_value(u.s))

    # Correct: mu = 4000 * 100/2000 = 200, sigma = sqrt(200) = 14.1,
    # 4 sigma = +-56.6. An unrescaled implementation, using interval 0's
    # full 200 s livetime instead of the 100 s overlap, would sit at
    # mu = 400 -- about 14 sigma away.
    expected = mu_full * 100.0 / duration.to_value(u.s)
    sigma = np.sqrt(expected)

    assert abs(len(events) - expected) < 4 * sigma


# --- what run_events actually yields -------------------------------------

def test_run_events_yields_time_source_sim_event_and_reco_event():
    earth = _make_earth()
    detector = _make_tracker()

    duration = 500 * u.s
    history = _make_history(duration, 5, earth)

    source = PointSource(sky_angle=CLEAN_SKY_ANGLE,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=_flux_for_expected_counts(30.0, detector, duration))

    events = _run(detector, source, history, earth)

    assert len(events) > 0

    for event in events:
        assert len(event) == 4

        time, event_source, sim_event, reco_event = event

        assert isinstance(time, u.Quantity)
        assert time.unit.is_equivalent(u.s)
        assert 0 * u.s <= time < duration

        assert event_source is source

        # The thrown photon, as `Simulator` already yields it.
        assert isinstance(sim_event.direction, u.Quantity)
        assert sim_event.energy == 1 * u.MeV

        # And a reconstruction, triggered or not.
        assert isinstance(reco_event.triggered, (bool, np.bool_))


# --- error cases ----------------------------------------------------------

def test_unnormalized_source_raises_at_construction():
    # Without a flux there is no rate, so there is no mu to draw a Poisson
    # from. Section 6 has no meaningful fallback, and silently simulating
    # nothing would be the worst possible outcome.
    earth = _make_earth()
    detector = _make_tracker()

    history = _make_history(500 * u.s, 5, earth)

    source = PointSource(sky_angle=10 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV))

    assert source.flux() is None

    with pytest.raises(Exception):
        InertialSimulator(detector=detector,
                          sources=source,
                          reconstructor=SimpleTraditionalReconstructor(),
                          spacecraft_history=history,
                          earth=earth)


def test_offaxis_angle_far_field_source_raises_at_construction():
    # A detector-frame-aimed far-field source is bolted to the spacecraft: it
    # would silently ignore the attitude and sit at a fixed off-axis angle
    # forever, which is exactly the bug the inertial simulator exists to
    # avoid. It has to be refused up front, not at the first photon.
    earth = _make_earth()
    detector = _make_tracker()

    history = _make_history(500 * u.s, 5, earth)

    source = PointSource(offaxis_angle=10 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=1e-3 / u.cm / u.s)

    with pytest.raises(Exception) as excinfo:
        InertialSimulator(detector=detector,
                          sources=source,
                          reconstructor=SimpleTraditionalReconstructor(),
                          spacecraft_history=history,
                          earth=earth)

    # However it spells them ('offaxis_angle' / 'off-axis angle',
    # 'sky_angle' / 'sky angle'), the message has to say which is wrong.
    message = str(excinfo.value).lower()

    assert 'axis' in message or 'sky' in message, str(excinfo.value)


def test_mismatched_earth_raises_at_construction():
    # F1 (PR3 review): `earth` is taken separately from `spacecraft_history`
    # precisely so the two can disagree with each other -- but nothing
    # checked that they didn't. If earth.radius > some interval's
    # orbit_radius, `Earth._is_occulted`'s `arcsin(R_E / r)` silently
    # produces `nan`, `abs(delta) < nan` is always `False`, and occultation
    # vanishes from the whole run with nothing louder than a numpy
    # RuntimeWarning. This must be caught at construction instead.
    earth_history = _make_earth()  # radius = EARTH_RADIUS = 6371 km
    detector = _make_tracker()

    history = _make_history(500 * u.s, 5, earth_history)  # orbit_radius = 6771 km

    source = PointSource(sky_angle=10 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=1e-3 / u.cm / u.s)

    # An Earth whose radius exceeds the history's orbit_radius: exactly the
    # inconsistency this check exists to catch.
    mismatched_earth = Earth(radius=50000 * u.km)

    with pytest.raises(ValueError) as excinfo:
        InertialSimulator(detector=detector,
                          sources=source,
                          reconstructor=SimpleTraditionalReconstructor(),
                          spacecraft_history=history,
                          earth=mismatched_earth)

    message = str(excinfo.value)
    assert '50000' in message or '50000.0' in message

    # A consistent Earth (the one the history was actually validated
    # against) must still construct without complaint.
    InertialSimulator(detector=detector,
                      sources=source,
                      reconstructor=SimpleTraditionalReconstructor(),
                      spacecraft_history=history,
                      earth=earth_history)


def test_sky_angle_source_raises_when_used_without_a_pose():
    # The counterpart at the source level: `pose = None` is detector-frame
    # mode, which a sky-angle source cannot honour.
    detector = _make_tracker()

    source = PointSource(sky_angle=10 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=1e-3 / u.cm / u.s)

    with pytest.raises(Exception) as excinfo:
        source.random_photon(detector)

    assert 'pose' in str(excinfo.value).lower()
