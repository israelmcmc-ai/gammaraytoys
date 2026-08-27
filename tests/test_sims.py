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
