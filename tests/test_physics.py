import astropy.units as u
import numpy as np
import pytest

from gammaraytoys.physics import ComptonPhysics2D


def test_scattering_angle_pdf_normalized():
    physics = ComptonPhysics2D(500 * u.keV)

    phi = np.linspace(-np.pi, np.pi, 200001) * u.rad
    pdf = physics.scattering_angle_pdf(phi)

    integral = np.trapezoid(np.asarray(pdf), phi.to_value(u.rad))

    assert integral == pytest.approx(1, rel=1e-3)


def test_energy_out_no_loss_at_zero_angle():
    physics = ComptonPhysics2D(1 * u.MeV)

    assert physics.energy_out(0 * u.rad).to_value(u.MeV) == pytest.approx(1, rel=1e-9)


def test_energy_out_decreases_with_scattering_angle():
    physics = ComptonPhysics2D(1 * u.MeV)

    e_forward = physics.energy_out(0 * u.rad)
    e_side = physics.energy_out(90 * u.deg)
    e_back = physics.energy_out(180 * u.deg)

    assert e_forward > e_side > e_back


def test_scattering_angle_inverts_energy_out():
    physics = ComptonPhysics2D(662 * u.keV)

    for phi in [10, 45, 90, 135, 170] * u.deg:
        energy_out = physics.energy_out(phi.to(u.rad))
        recovered = physics.scattering_angle(energy_out)

        assert recovered.to_value(u.rad) == pytest.approx(phi.to_value(u.rad), abs=1e-6)


@pytest.mark.filterwarnings("ignore:invalid value encountered in arccos:RuntimeWarning")
def test_scattering_angle_unphysical_energy_returns_nan():
    physics = ComptonPhysics2D(500 * u.keV)

    # An outgoing energy larger than the incoming energy is unphysical
    assert np.isnan(physics.scattering_angle(600 * u.keV))


def test_modulation_factor_vanishes_forward_and_backward():
    physics = ComptonPhysics2D(300 * u.keV)

    mod_forward = physics.modulation_factor(0.0001 * u.deg)
    mod_backward = physics.modulation_factor(179.9999 * u.deg)

    assert float(mod_forward) == pytest.approx(0, abs=1e-3)
    assert float(mod_backward) == pytest.approx(0, abs=1e-3)


def test_random_scattering_angle_within_domain():
    physics = ComptonPhysics2D(400 * u.keV)

    phi = physics.random_scattering_angle(size=1000)

    assert np.all(phi >= -np.pi * u.rad)
    assert np.all(phi <= np.pi * u.rad)
