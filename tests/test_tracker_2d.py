import astropy.units as u
import numpy as np
import pytest

from gammaraytoys import ToyTracker2D
from gammaraytoys.detectors import Photon
from gammaraytoys.coordinates import Cartesian2D


def test_overlapping_layers_raise():
    with pytest.raises(ValueError):
        ToyTracker2D(material='Ge',
                    layer_length=16 * u.cm,
                    layer_positions=[0, 2] * u.mm,
                    layer_thickness=5 * u.mm,
                    energy_resolution=0.01,
                    energy_threshold=20 * u.keV)


def test_horizontal_particle_never_interacts(tracker):
    photon = Photon(position=Cartesian2D(0 * u.cm, 0 * u.mm),
                    direction=0 * u.deg,
                    energy=1 * u.MeV)

    result = tracker.simulate_event(photon)

    assert result.interaction is None


def test_simulate_event_usually_interacts_with_the_stack(tracker):
    # A photon fired straight down through the (multi-layer) Ge stack has a
    # substantial per-layer interaction probability, so across many draws
    # most should interact somewhere
    ninteracted = 0
    ntrials = 50

    for _ in range(ntrials):
        photon = Photon(position=Cartesian2D(0 * u.cm, 5 * u.cm),
                        direction=270 * u.deg,
                        energy=200 * u.keV)

        result = tracker.simulate_event(photon)

        if result.interaction is not None:
            ninteracted += 1
            # A sub-threshold deposit legitimately yields zero *measured*
            # hits even though an interaction happened
            hits = result.hits
            assert np.all(hits.energy >= 0 * u.MeV)

    assert ninteracted > ntrials / 2


def test_simulate_event_deterministic_with_seed(tracker):
    photon_kwargs = dict(position=Cartesian2D(0 * u.cm, 5 * u.cm),
                         direction=270 * u.deg,
                         energy=200 * u.keV,
                         chirality=1)

    np.random.seed(42)
    photon1 = Photon(**photon_kwargs)
    result1 = tracker.simulate_event(photon1)

    np.random.seed(42)
    photon2 = Photon(**photon_kwargs)
    result2 = tracker.simulate_event(photon2)

    assert result1.hits.nhits == result2.hits.nhits
    assert u.allclose(result1.hits.energy, result2.hits.energy)
