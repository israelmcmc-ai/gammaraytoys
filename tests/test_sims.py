import astropy.units as u
import numpy as np
import pytest
from histpy import Axis

from gammaraytoys.sims import (Simulator, PointSource,
                               MonoenergeticSpectrum,
                               SimpleTraditionalReconstructor)


@pytest.fixture
def source():
    return PointSource(offaxis_angle=0 * u.deg,
                       spectrum=MonoenergeticSpectrum(1 * u.MeV),
                       flux=1e-3 / u.cm / u.s)


@pytest.fixture
def simulator(tracker, source):
    return Simulator(detector=tracker, sources=source,
                     reconstructor=SimpleTraditionalReconstructor())


def test_nsources_single_source(simulator):
    assert simulator.nsources == 1


def test_nsources_multiple_sources(tracker, source):
    source2 = PointSource(offaxis_angle=10 * u.deg,
                          spectrum=MonoenergeticSpectrum(2 * u.MeV),
                          flux=1e-3 / u.cm / u.s)

    sim = Simulator(detector=tracker, sources=[source, source2],
                    reconstructor=SimpleTraditionalReconstructor())

    assert sim.nsources == 2
    assert sim.total_flux.to_value(source.flux.unit) == pytest.approx(
        (source.flux + source2.flux).to_value(source.flux.unit))


def test_standarize_termination_requires_exactly_one_condition(simulator):
    with pytest.raises(ValueError):
        simulator._standarize_termination()

    with pytest.raises(ValueError):
        simulator._standarize_termination(nsim=10, ntrig=5)


def test_run_events_by_nsim_count(simulator):
    events = list(simulator.run_events(nsim=25))

    assert len(events) == 25
    assert simulator.nsim == 25

    for sim_event, reco_event in events:
        assert sim_event.energy == 1 * u.MeV


def test_run_events_by_ntrig_count(simulator):
    events = list(simulator.run_events(ntrig=5))

    ntriggered = sum(1 for _, reco in events if reco.triggered)

    assert ntriggered == 5
    assert simulator.ntrig == 5


def test_run_binned_returns_histogram_on_requested_axis(simulator):
    simulator.measured_energy_axis = Axis(np.geomspace(.1, 10, 20) * u.MeV,
                                          label='Em', scale='log')

    h_data = simulator.run_binned(nsim=200, axes='Em')

    assert h_data.axes.labels == ['Em']
    assert np.sum(h_data.contents) <= 200


def test_run_binned_with_sim_hist_returns_pair(simulator):
    h_data, h_sim = simulator.run_binned(nsim=50, axes='Em', photon_axes=True)

    assert list(h_sim.axes.labels) == list(simulator.photon_axes.labels)
    assert np.sum(h_sim.contents) == 50


def test_measured_energy_axis_setter_preserves_scale(simulator):
    simulator.measured_energy_axis = np.geomspace(.1, 10, 5) * u.MeV

    assert simulator.measured_energy_axis.label == 'Em'
    assert simulator.measured_energy_axis.nbins == 4


# --- PR 1: rate-based nsim/duration must match the pre-refactor formula -----
#
# Plan section 5.2: Simulator used to compute
#     nsim = round(total_flux * duration * throwing_plane_size)
# and now computes nsim = round(total_rate * duration), with
# total_rate = simulated_rate() summed over sources, which for a far-field
# source is flux * throwing_plane_size. For a far-field-only run the two
# must be numerically identical. We compute the "old formula" side directly
# from the flux/duration inputs and detector.throwing_plane_size (never by
# calling the Simulator and reading its own output back), then compare it to
# what `_standarize_termination` -- the new rate-based path -- actually
# produces. `_standarize_termination` is O(1) regardless of how large nsim
# comes out, so a huge nsim costs nothing to check.

@pytest.mark.parametrize("flux_per_cm_s, duration_s", [
    (1e-9, 0.137),    # tiny flux, sub-second duration -> nsim == 0
    (1e-6, 100.0),    # small flux
    (2.5e-4, 137.5),  # non-round values on both sides
    (1e-3, 1000.0),   # a "typical" tutorial-scale flux
    (1.0, 100.0),     # -> nsim in the thousands
    (5.0, 1e7),       # top of the requested range on both axes -> huge nsim
], ids=["tiny-flux-fractional-duration",
        "small-flux",
        "fractional-values",
        "typical-flux",
        "large-nsim",
        "huge-flux-and-duration"])
def test_nsim_from_duration_matches_old_flux_formula(tracker, flux_per_cm_s, duration_s):
    flux = flux_per_cm_s / u.cm / u.s
    duration = duration_s * u.s

    source = PointSource(offaxis_angle=0 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=flux)
    sim = Simulator(detector=tracker, sources=source,
                    reconstructor=SimpleTraditionalReconstructor())

    old_formula_nsim = round((flux * duration * tracker.throwing_plane_size).to_value(''))

    nsim, _, _ = sim._standarize_termination(duration=duration)

    assert nsim == old_formula_nsim


def test_nsim_from_duration_matches_old_flux_formula_multi_source(tracker):
    # Same check, but with the total-rate-summed-over-sources path that the
    # multi-source branch of Simulator.__init__ takes, against the old
    # formula's flux summed over sources.
    flux1 = 3e-4 / u.cm / u.s
    flux2 = 7e-4 / u.cm / u.s
    duration = 500 * u.s

    spec = MonoenergeticSpectrum(1 * u.MeV)
    s1 = PointSource(offaxis_angle=0 * u.deg, spectrum=spec, flux=flux1)
    s2 = PointSource(offaxis_angle=90 * u.deg, spectrum=spec, flux=flux2)

    sim = Simulator(detector=tracker, sources=[s1, s2],
                    reconstructor=SimpleTraditionalReconstructor())

    old_formula_nsim = round(((flux1 + flux2) * duration * tracker.throwing_plane_size).to_value(''))

    nsim, _, _ = sim._standarize_termination(duration=duration)

    assert nsim == old_formula_nsim


@pytest.mark.parametrize("flux_per_cm_s, nsim", [
    (1e-9, 1),
    (1e-6, 50),
    (1e-3, 2000),        # -> nsim well above 1000
    (5.0, 5_000_000),    # top of the requested flux range, huge nsim
], ids=["tiny-flux", "small-flux", "large-nsim", "huge-flux-and-nsim"])
def test_duration_from_nsim_matches_old_flux_formula(tracker, flux_per_cm_s, nsim):
    # Inverse direction: given nsim, Simulator._standarize_termination must
    # recover duration = nsim / (flux * throwing_plane_size), i.e. the exact
    # algebraic inverse of the formula checked above.
    flux = flux_per_cm_s / u.cm / u.s

    source = PointSource(offaxis_angle=0 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=flux)
    sim = Simulator(detector=tracker, sources=source,
                    reconstructor=SimpleTraditionalReconstructor())

    expected_duration = (nsim / (flux * tracker.throwing_plane_size)).to(u.s)

    _, _, duration = sim._standarize_termination(nsim=nsim)

    assert duration.to_value(u.s) == pytest.approx(expected_duration.to_value(u.s), rel=1e-12)


# --- PR 1: multi-source selection weights match the rate ratios ------------

def test_multi_source_selection_weights_match_rate_ratios(tracker):
    # Three point sources with flux ratio 1:3:6. simulated_rate() = flux *
    # throwing_plane_size for every one of them, and throwing_plane_size is
    # the same detector property for all three, so it cancels: the expected
    # selection-probability ratio is exactly the flux ratio, 0.1:0.3:0.6.
    spec = MonoenergeticSpectrum(1 * u.MeV)
    fluxes = u.Quantity([1.0, 3.0, 6.0]) / u.cm / u.s
    sources = [PointSource(offaxis_angle=i * 10 * u.deg, spectrum=spec, flux=f)
              for i, f in enumerate(fluxes)]

    sim = Simulator(detector=tracker, sources=sources,
                    reconstructor=SimpleTraditionalReconstructor())

    expected_p = (fluxes / np.sum(fluxes)).to_value('')

    # The weight array Simulator actually built for source selection must
    # equal the flux ratios exactly -- this part is deterministic arithmetic,
    # not a statistical claim.
    np.testing.assert_allclose(sim._relative_rate, expected_p)

    # Statistical check of the draw itself. This replicates the exact
    # selection call run_events() makes per photon --
    # `np.random.choice(range(nsources), p=self._relative_rate)` -- without
    # the expensive part that follows it (random_photon + the full detector
    # walk), since which source gets picked does not depend on what happens
    # to the photon afterwards.
    n_draws = 200_000
    draws = np.random.choice(sim.nsources, size=n_draws, p=sim._relative_rate)
    empirical = np.array([np.sum(draws == i) for i in range(sim.nsources)]) / n_draws

    # Binomial standard error per source: sigma = sqrt(p(1-p)/N). We assert
    # at 5 sigma -- one full sigma of headroom beyond the 4 sigma floor --
    # so a false failure would occur with probability ~6e-7 per source even
    # if this weren't already deterministic under the seeded RNG.
    sigma = np.sqrt(expected_p * (1 - expected_p) / n_draws)
    assert np.all(np.abs(empirical - expected_p) <= 5 * sigma), (
        f"empirical={empirical}, expected={expected_p}, 5-sigma={5 * sigma}")


# --- PR 1: total_flux still works, and is None when a source has no flux ---

def test_total_flux_far_field_only_run(tracker, source):
    # Baseline: a single far-field source with a flux set.
    sim = Simulator(detector=tracker, sources=source,
                    reconstructor=SimpleTraditionalReconstructor())

    assert sim.total_flux.to_value(source.flux.unit) == pytest.approx(
        source.flux.to_value(source.flux.unit))


def test_total_flux_none_when_source_has_no_flux(tracker):
    # PointSource with neither `flux` nor `flux_pivot`/`pivot_energy` given
    # is unnormalized: flux is None, so is simulated_rate(), and so must be
    # total_flux and total_rate.
    source = PointSource(offaxis_angle=0 * u.deg, spectrum=MonoenergeticSpectrum(1 * u.MeV))

    sim = Simulator(detector=tracker, sources=source,
                    reconstructor=SimpleTraditionalReconstructor())

    assert sim.total_flux is None
    assert sim.total_rate is None
