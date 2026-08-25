import astropy.units as u
import numpy as np
from astropy.coordinates import CartesianRepresentation

from gammaraytoys.coordinates import Cartesian2D


def test_construction_sets_z_to_zero():
    c = Cartesian2D(1 * u.cm, 2 * u.cm)

    assert c.x == 1 * u.cm
    assert c.y == 2 * u.cm
    assert c.z == 0 * u.cm


def test_construction_broadcasts_scalar_x():
    c = Cartesian2D(0 * u.cm, [1, 2, 3] * u.cm)

    assert np.all(c.x == 0 * u.cm)
    np.testing.assert_array_equal(c.y.to_value(u.cm), [1, 2, 3])


def test_to_cartesian_round_trip():
    c = Cartesian2D(3 * u.m, 4 * u.m)

    full = c.to_cartesian()

    assert isinstance(full, CartesianRepresentation)
    assert full.x == c.x
    assert full.y == c.y
    assert full.z == 0 * u.m


def test_from_cartesian_round_trip():
    full = CartesianRepresentation(x=1 * u.m, y=2 * u.m, z=99 * u.m)

    c = Cartesian2D.from_cartesian(full)

    assert c.x == 1 * u.m
    assert c.y == 2 * u.m
    assert c.z == 0 * u.m
