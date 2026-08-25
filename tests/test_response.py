import astropy.units as u
import numpy as np
import pytest
from histpy import Histogram, Axis

from gammaraytoys.analysis import SpectralResponse
from gammaraytoys.detectors import PointSource, PowerLawSpectrum


@pytest.fixture
def response():
    # A tiny synthetic response with perfect energy resolution (identity
    # Ei -> Em mapping) and a flat 1 cm effective area (this is the 2D toy model -- "area" here means length)
    Ei = Axis(np.array([1, 2, 3, 4]) * u.MeV, label='Ei')
    Em = Axis(np.array([1, 2, 3, 4]) * u.MeV, label='Em')

    contents = np.eye(3) * u.cm

    return SpectralResponse([Ei, Em], contents)


def test_photon_and_measured_energy_axes(response):
    assert response.photon_energy_axis.label == 'Ei'
    assert response.measured_energy_axis.label == 'Em'


def test_effective_area_sums_over_measured_energy(response):
    eff_area = response.effective_area()

    assert eff_area.axes.labels == ['Ei']
    np.testing.assert_allclose(eff_area.contents.to_value(u.cm), [1, 1, 1])


def test_energy_dispersion_normalized_per_photon_bin(response):
    dispersion = response.energy_dispersion()

    # Each row (photon energy bin) should sum to 1 across measured energy,
    # since our synthetic response has uniform 1 cm effective area (this is the 2D toy model -- "area" here means length)
    row_sums = np.sum(dispersion.contents, axis=1)
    np.testing.assert_allclose(row_sums, [1, 1, 1])


def test_expected_counts_matches_discretized_spectrum_on_diagonal(response):
    spec = PowerLawSpectrum(index=0, min_energy=1 * u.MeV, max_energy=4 * u.MeV)
    source = PointSource(offaxis_angle=0 * u.deg, spectrum=spec,
                         flux=1e-3 / u.cm / u.s)
    duration = 10 * u.s

    expected = response.expected_counts(source, duration)

    binned_spec = source.discretize_spectrum(response.photon_energy_axis)
    expected_from_binned = (binned_spec.contents * duration * 1 * u.cm).to_value('')

    assert expected.axes.labels == ['Em']
    np.testing.assert_allclose(expected.contents, expected_from_binned, rtol=1e-6)


def test_open_round_trips_and_validates(tmp_path, response):
    path = tmp_path / "response.h5"
    response.write(str(path))

    reopened = SpectralResponse.open(str(path))

    assert reopened.ndim == 2
    assert set(reopened.axes.labels) == {'Ei', 'Em'}
    np.testing.assert_allclose(reopened.contents.to_value(u.cm),
                               response.contents.to_value(u.cm))


def test_open_rejects_wrong_ndim(tmp_path):
    axis = Axis(np.array([1, 2, 3]) * u.MeV, label='Ei')
    h1d = Histogram(axis, np.array([1.0, 2.0]))

    path = tmp_path / "bad.h5"
    h1d.write(str(path))

    with pytest.raises(RuntimeError):
        SpectralResponse.open(str(path))


def test_open_rejects_missing_axis_labels(tmp_path):
    a = Axis(np.array([1, 2, 3]) * u.MeV, label='foo')
    b = Axis(np.array([1, 2, 3]) * u.MeV, label='bar')
    h = Histogram([a, b], np.eye(2))

    path = tmp_path / "bad_labels.h5"
    h.write(str(path))

    with pytest.raises(RuntimeError):
        SpectralResponse.open(str(path))
