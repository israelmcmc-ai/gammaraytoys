import numpy as np
import pytest

from gammaraytoys.analysis import poisson_binned_log_likelihood, unbinned_log_likelihood


def test_poisson_binned_log_likelihood_matches_manual_formula():
    data = np.array([2.0, 5.0, 0.0])
    expectation = np.array([2.5, 4.0, 1.0])

    expected = np.sum(data * np.log(expectation) - expectation)

    assert poisson_binned_log_likelihood(data, expectation) == pytest.approx(expected)


@pytest.mark.filterwarnings("ignore:divide by zero encountered in log:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in multiply:RuntimeWarning")
def test_poisson_binned_log_likelihood_ignores_zero_data_zero_expectation_nan():
    # data*log(expectation) with expectation=0 and data=0 gives 0*-inf = nan,
    # which should be dropped by nansum rather than poisoning the result
    data = np.array([0.0, 3.0])
    expectation = np.array([0.0, 2.0])

    expected = 3.0 * np.log(2.0) - 2.0

    assert poisson_binned_log_likelihood(data, expectation) == pytest.approx(expected)


def test_unbinned_log_likelihood_matches_manual_formula():
    expectation_density = np.array([0.1, 0.2, 0.05])
    total_expectation = 10.0

    expected = -total_expectation + np.sum(np.log(expectation_density))

    assert unbinned_log_likelihood(expectation_density, total_expectation) == pytest.approx(expected)


def test_unbinned_log_likelihood_higher_for_better_fit():
    # A model whose density is closer to where the data actually landed
    # should score higher, for the same total expectation
    total_expectation = 5.0

    good_fit = unbinned_log_likelihood(np.array([1.0, 1.0, 1.0]), total_expectation)
    bad_fit = unbinned_log_likelihood(np.array([0.01, 0.01, 0.01]), total_expectation)

    assert good_fit > bad_fit
