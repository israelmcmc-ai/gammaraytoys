import numpy as np
import pytest

from gammaraytoys.analysis import richardson_lucy


def test_richardson_lucy_recovers_true_model_noiseless_identity_response():
    true_model = np.array([3.0, 7.0, 1.0])
    response = np.eye(3)

    data = np.dot(response, true_model)

    model = np.ones(3)
    model = richardson_lucy(data, model, response, niter=200)

    np.testing.assert_allclose(model, true_model, rtol=1e-3)


def test_richardson_lucy_recovers_true_model_noiseless_mixing_response():
    true_model = np.array([2.0, 5.0])
    # Each data bin sees a mix of both model bins
    response = np.array([[0.8, 0.3],
                         [0.2, 0.7]])

    data = np.dot(response, true_model)

    model = np.ones(2)
    model = richardson_lucy(data, model, response, niter=500)

    np.testing.assert_allclose(model, true_model, rtol=1e-2)


def test_richardson_lucy_preserves_total_counts():
    true_model = np.array([4.0, 2.0, 6.0])
    response = np.eye(3)
    data = np.dot(response, true_model)

    model = np.ones(3)
    model = richardson_lucy(data, model, response, niter=50)

    assert np.sum(model) == pytest.approx(np.sum(data), rel=1e-6)
