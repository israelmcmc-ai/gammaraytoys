import numpy as np
import pytest
import astropy.units as u

from gammaraytoys import ToyTracker2D, ToyCodedMaskDetector2D


@pytest.fixture(autouse=True)
def _seed_random():
    """Make every test's random draws reproducible."""
    np.random.seed(0)


@pytest.fixture
def tracker():
    return ToyTracker2D(material='Ge',
                        layer_length=16 * u.cm,
                        layer_positions=[0, 5, 10, 20, 25, 30] * u.mm,
                        layer_thickness=5 * u.mm,
                        energy_resolution=0.01,
                        energy_threshold=20 * u.keV)


@pytest.fixture
def mask_detector():
    return ToyCodedMaskDetector2D.create_random_mask(mask_size=40 * u.cm,
                                                      mask_npix=130,
                                                      mask_separation=100 * u.cm,
                                                      open_fraction=.5,
                                                      detector_size=22 * u.cm,
                                                      detector_npix=440,
                                                      detector_efficiency=0.65)
