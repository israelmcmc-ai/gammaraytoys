import astropy.units as u
import numpy as np
import pytest

from gammaraytoys.coordinates import Cartesian2D
from gammaraytoys.detectors import (Interaction, Particle, Photon, Compton,
                                    Absorption, EventList)
from gammaraytoys.detectors.event import Hits


def test_hits_mismatched_size_raises():
    # Regression test: Hits.__init__ used to build the RuntimeError but
    # never raise it.
    with pytest.raises(RuntimeError):
        Hits(position=[1, 2])


def test_hits_default_is_empty():
    hits = Hits()

    assert hits.nhits == 0


def test_hits_from_list_round_trip():
    hit1 = Interaction('absorption', Cartesian2D(0 * u.cm, 0 * u.cm), 1 * u.MeV)
    hit1.set_measurement(layer=0, position=Cartesian2D(0 * u.cm, 0 * u.cm), energy=1 * u.MeV)

    hit2 = Interaction('compton', Cartesian2D(1 * u.cm, 1 * u.cm), 0.5 * u.MeV)
    hit2.set_measurement(layer=1, position=Cartesian2D(1 * u.cm, 1 * u.cm), energy=0.5 * u.MeV)

    hit1.add_child(hit2)

    # Hits are built from an Interaction tree via Particle.hits_iter, which
    # is how the rest of the codebase constructs them
    photon = Photon(position=Cartesian2D(0 * u.cm, 5 * u.cm),
                    direction=270 * u.deg,
                    energy=1 * u.MeV)
    photon.set_interaction(hit1)

    hits = Hits.from_list(photon.hits_iter)

    assert hits.nhits == 2
    assert list(hits.layer) == [0, 1]
    assert u.allclose(hits.energy, [1, 0.5] * u.MeV)


def test_particle_hits_iter_empty_when_no_interaction():
    photon = Photon(position=Cartesian2D(0 * u.cm, 0 * u.cm),
                    direction=0 * u.deg,
                    energy=1 * u.MeV)

    assert list(photon.hits_iter) == []
    assert photon.hits.nhits == 0


def test_interaction_children_tracked():
    absorption = Absorption(position=Cartesian2D(0 * u.cm, 0 * u.cm), energy=1 * u.MeV)
    photon = Photon(position=Cartesian2D(0 * u.cm, 1 * u.cm),
                    direction=270 * u.deg,
                    energy=1 * u.MeV)

    absorption.add_child(photon)

    assert absorption.children == [photon]


def test_photon_default_chirality_is_plus_or_minus_one():
    photon = Photon(position=Cartesian2D(0 * u.cm, 0 * u.cm),
                    direction=0 * u.deg,
                    energy=1 * u.MeV)

    assert photon.chirality in (1, -1)


def test_eventlist_append_and_write(tmp_path):
    import yaml

    events = EventList()
    events.nsim = 3
    events.sim_time = 10 * u.s

    photon1 = Photon(position=Cartesian2D(0 * u.cm, 0 * u.cm),
                     direction=0 * u.deg, energy=1 * u.MeV, chirality=1)
    photon2 = Photon(position=Cartesian2D(1 * u.cm, 1 * u.cm),
                     direction=90 * u.deg, energy=2 * u.MeV, chirality=-1)

    events.append(photon1)
    events.append(photon2)

    assert events[0] is photon1
    assert events[1] is photon2

    path = tmp_path / "events.yaml"
    events.write(str(path))

    # Regression test: sim_time (an astropy Quantity) used to be dumped as
    # a raw Python object instead of a string like Particle.to_dict() does
    # for its Quantity fields, making the file unloadable with
    # yaml.safe_load.
    with open(path) as f:
        loaded = yaml.safe_load(f)

    assert loaded['nsim'] == 3
    assert loaded['sim_time'] == str(10 * u.s)
    assert len(loaded['events']) == 2
    assert loaded['events'][0]['nevent'] == 0
    assert loaded['events'][1]['particle_type'] == 'photon'


def test_eventlist_write_with_no_sim_time(tmp_path):
    import yaml

    events = EventList()

    path = tmp_path / "events_empty.yaml"
    events.write(str(path))

    with open(path) as f:
        loaded = yaml.safe_load(f)

    assert loaded['nsim'] is None
    assert loaded['sim_time'] is None
    assert loaded['events'] == []
