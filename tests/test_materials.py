import astropy.units as u
import numpy as np
import pytest

from gammaraytoys.materials import Material


def test_from_name_caches_instance_passthrough():
    m = Material.from_name('Ge')

    assert Material.from_name(m) is m


def test_attenuation_coefficients_positive_and_consistent():
    m = Material.from_name('Ge')

    energy = 1 * u.MeV

    photo = m.photo_attenuation(energy)
    compton = m.compton_attenuation(energy)
    pair = m.pair_attenuation(energy)
    total = m.total_attenuation(energy)

    assert photo.value > 0
    assert compton.value > 0
    assert pair.value >= 0
    assert total.to_value(photo.unit) == pytest.approx(
        (photo + compton + pair).to_value(photo.unit), rel=1e-6)


def test_attenuation_accepts_array_energy():
    m = Material.from_name('Ge')

    energy = [0.5, 1, 2] * u.MeV
    total = m.total_attenuation(energy)

    assert total.shape == (3,)
    assert np.all(total.value > 0)
