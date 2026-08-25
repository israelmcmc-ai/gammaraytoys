import numpy as np
import pytest
from histpy import Axis

from gammaraytoys.analysis import LogLikeGrid


def _neg_quadratic(x, x0=2.0, y0=-1.0, y=None):
    # A single-peaked log-likelihood surface with a known maximum, used to
    # exercise LogLikeGrid without depending on the detector simulation
    if y is None:
        return -(x - x0) ** 2
    return -((x - x0) ** 2 + (y - y0) ** 2)


def test_compute_and_optimal_parameters_1d():
    axis = Axis(np.linspace(-5, 5, 101), label='x')
    grid = LogLikeGrid(axis)

    grid.compute(lambda x: _neg_quadratic(x))

    assert grid.optimal_parameters() == pytest.approx(2.0, abs=0.1)
    # The true maximum (0) falls between grid points spaced 0.1 apart
    assert grid.maximum() == pytest.approx(0.0, abs=0.01)


def test_compute_and_optimal_parameters_2d():
    x_axis = Axis(np.linspace(-5, 5, 51), label='x')
    y_axis = Axis(np.linspace(-5, 5, 51), label='y')
    grid = LogLikeGrid([x_axis, y_axis])

    grid.compute(lambda x, y: _neg_quadratic(x, y=y))

    x_opt, y_opt = grid.optimal_parameters()
    assert x_opt == pytest.approx(2.0, abs=0.2)
    assert y_opt == pytest.approx(-1.0, abs=0.2)


def test_maximum_profiles_out_an_axis():
    x_axis = Axis(np.linspace(-5, 5, 51), label='x')
    y_axis = Axis(np.linspace(-5, 5, 51), label='y')
    grid = LogLikeGrid([x_axis, y_axis])

    grid.compute(lambda x, y: _neg_quadratic(x, y=y))

    profile = grid.maximum('y')

    assert profile.ndim == 1
    assert profile.axes[0].label == 'x'
    # Profiling out y (already at its optimum everywhere x is near x0)
    # should reproduce the 1D peak location
    argmax = profile.axes[0].centers[np.argmax(profile.contents)]
    assert argmax == pytest.approx(2.0, abs=0.2)


def test_parameter_bounds_contain_optimum():
    axis = Axis(np.linspace(-5, 5, 101), label='x')
    grid = LogLikeGrid(axis)

    grid.compute(lambda x: _neg_quadratic(x))

    lo, hi = grid.parameter_bounds(cont=.9)

    assert lo < 2.0 < hi
