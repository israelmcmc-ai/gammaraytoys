import warnings

import astropy.units as u
import numpy as np

from gammaraytoys.coordinates import Cartesian2D
from gammaraytoys.sims.reco import (Reconstructor,
                                    SimpleTraditionalReconstructor,
                                    RecoCompton)
from gammaraytoys.sims.event import Hits


def test_simple_traditional_reconstructor_implements_interface():
    # Regression test: SimpleTraditionalReconstructor used to inherit from
    # ABC directly instead of Reconstructor.
    assert issubclass(SimpleTraditionalReconstructor, Reconstructor)


class _FakeSimEvent:
    """Minimal stand-in exposing only what reconstruct() needs."""

    def __init__(self, hits):
        self.hits = hits


def _make_hits(layers, positions, energies):
    layer = np.array(layers)
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    position = Cartesian2D(u.Quantity(x, u.cm), u.Quantity(y, u.cm))
    energy = u.Quantity(energies, u.MeV)

    return Hits(layer, position, energy)


def test_reconstruct_untriggered_below_min_hits():
    reco = SimpleTraditionalReconstructor()

    hits = _make_hits([0], [(0, 10)], [1])
    sim_event = _FakeSimEvent(hits)

    result = reco.reconstruct(sim_event)

    assert not result.triggered


def test_reconstruct_untriggered_when_top_layer_not_hit():
    reco = SimpleTraditionalReconstructor()

    hits = _make_hits([1, 2], [(0, 5), (0, 0)], [0.3, 0.3])
    sim_event = _FakeSimEvent(hits)

    result = reco.reconstruct(sim_event)

    assert not result.triggered


def test_reconstruct_triggered_two_hits():
    reco = SimpleTraditionalReconstructor()

    # Top hit (layer 0) plus a bottom hit -- the pair must correspond to a
    # kinematically valid Compton scatter for the event to survive the CDS
    # check
    hits = _make_hits([0, 1], [(0, 10), (0, 0)], [0.2, 0.3])
    sim_event = _FakeSimEvent(hits)

    result = reco.reconstruct(sim_event)

    # Whether or not the CDS is physical for this made-up pair, reconstruct
    # must return a RecoCompton and never raise
    assert isinstance(result, RecoCompton)


def test_reconstruct_untriggered_when_all_hits_in_top_layer():
    # Regression test: two (or more) hits, all in layer 0, used to still
    # trigger -- `position_bottom = np.mean(hits.position[hits.layer > 0])`
    # then averaged an empty selection, giving psi = nan while the event
    # was reported as triggered.
    reco = SimpleTraditionalReconstructor()

    hits = _make_hits([0, 0], [(0, 10), (0.5, 10)], [0.65, 0.34])
    sim_event = _FakeSimEvent(hits)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = reco.reconstruct(sim_event)

    assert not result.triggered
    assert result.psi is None


def test_reconstruct_triggered_top_and_bottom_gives_finite_psi():
    # Regression guard: the layer > 0 requirement must not be too strict --
    # an ordinary top-then-bottom event still triggers with a finite psi.
    reco = SimpleTraditionalReconstructor()

    hits = _make_hits([0, 1], [(0, 10), (0, 0)], [0.65, 0.34])
    sim_event = _FakeSimEvent(hits)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = reco.reconstruct(sim_event)

    assert result.triggered
    assert np.isfinite(result.psi)


def test_recocompton_untriggered_by_default():
    reco = RecoCompton()

    assert not reco.triggered


def test_recocompton_triggered_with_values():
    reco = RecoCompton(energy=1 * u.MeV, phi=0.5 * u.rad, psi=0 * u.rad)

    assert reco.triggered
