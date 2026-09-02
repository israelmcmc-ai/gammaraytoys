"""
Tests for source plotting (plan section 3.3, geometry conventions), added on
top of PR 1's `Source` hierarchy.

`FarFieldSource.plot_sky_circle` / `plot_sky_marker` / `plot_sky_arc` draw a
"sky" circle at `2 x` the detector's surrounding-circle radius and place a
star or arc just outside it (`1.08 x` the sky radius), along the unit
vector `(sin Nu, cos Nu)` from the sky circle's centre (`Nu = 0` is `+y`,
`Nu = 90 deg` is `+x` -- see `docs/dev/inertial_sim_plan.md` section 3.3).
`NearFieldSource.plot` instead draws a star directly at the source's own
detector-frame `position`.

Every geometric expectation below is derived independently from the
detector's own properties (`surrounding_circle_center`,
`surrounding_circle_radius`) and the stated 1.08x/2x factors, never from
what the plotting code itself prints, so these tests would actually catch a
wrong factor or a flipped sin/cos convention.

Uses the Agg backend throughout, since every test here touches plotting.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import astropy.units as u
import numpy as np
import pytest

from gammaraytoys.coordinates import Cartesian2D
from gammaraytoys.sims import (NearFieldSource, PointSource, IsotropicSource,
                               MonoenergeticSpectrum)


LENGTH_UNIT = u.cm


def _sky_radius(tracker):
    """2x the detector's surrounding-circle radius, in cm."""
    return (2 * tracker.surrounding_circle_radius).to_value(LENGTH_UNIT)


def _marker_radius(tracker):
    """1.08x the sky radius, in cm."""
    return 1.08 * _sky_radius(tracker)


def _center(tracker):
    """Sky/surrounding circle centre, as a plain (x, y) tuple in cm."""
    c = tracker.surrounding_circle_center
    return c.x.to_value(LENGTH_UNIT), c.y.to_value(LENGTH_UNIT)


def _expected_marker_xy(tracker, offaxis_angle):
    """
    Expected (x, y) of a marker at `offaxis_angle`, just outside the sky
    circle, from the geometry in plan section 3.3 directly -- not from any
    helper in `gammaraytoys.sims`.
    """
    cx, cy = _center(tracker)
    r = _marker_radius(tracker)
    nu = offaxis_angle.to_value(u.rad)
    return cx + r * np.sin(nu), cy + r * np.cos(nu)


class _StubNearFieldSource(NearFieldSource):
    """Minimal concrete `NearFieldSource` with a fixed position, standing
    in for `NearPointSource` (added in PR 4) so `NearFieldSource.plot` can
    be exercised before any concrete near-field geometry exists."""

    def __init__(self, position, rate=None):
        self._position = position
        self._rate = rate
        self._spectrum = MonoenergeticSpectrum(1 * u.MeV)

    @property
    def position(self):
        return self._position

    @property
    def rate(self):
        return self._rate

    @property
    def spectrum(self):
        return self._spectrum

    def random_photon(self, detector, pose=None, earth=None):
        raise NotImplementedError

    def simulated_rate(self, detector, pose=None):
        return self._rate


def _star_lines(ax):
    """Every Line2D on `ax` drawn as a star marker (no connecting line)."""
    return [line for line in ax.lines if line.get_marker() == '*']


# --- PointSource: marker position matches the geometry ---------------------

@pytest.mark.parametrize("offaxis_angle", [0, 90, -90, 180, 45] * u.deg)
def test_pointsource_star_position_matches_geometry(tracker, offaxis_angle):
    ax = tracker.plot()
    source = PointSource(offaxis_angle=offaxis_angle,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=1e-3 / u.cm / u.s)

    source.plot(ax, tracker)

    stars = _star_lines(ax)
    assert len(stars) == 1
    x, y = stars[0].get_xdata(), stars[0].get_ydata()

    exp_x, exp_y = _expected_marker_xy(tracker, offaxis_angle)
    assert x[0] == pytest.approx(exp_x, abs=1e-9)
    assert y[0] == pytest.approx(exp_y, abs=1e-9)

    plt.close(ax.figure)


def test_pointsource_nu_zero_is_above_the_detector(tracker):
    # Sanity check called out explicitly in the plan (section 3.3): Nu = 0
    # is straight up, +y, i.e. directly above the detector, not off to a
    # side or below it.
    ax = tracker.plot()
    source = PointSource(offaxis_angle=0 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=1e-3 / u.cm / u.s)
    source.plot(ax, tracker)

    x, y = _star_lines(ax)[0].get_xdata(), _star_lines(ax)[0].get_ydata()
    cx, cy = _center(tracker)
    assert x[0] == pytest.approx(cx, abs=1e-9)
    assert y[0] > cy

    plt.close(ax.figure)


# --- Marker/sky radius factors ---------------------------------------------

def test_marker_radius_is_1_08x_sky_radius(tracker):
    ax = tracker.plot()
    source = PointSource(offaxis_angle=30 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=1e-3 / u.cm / u.s)
    source.plot(ax, tracker)

    cx, cy = _center(tracker)
    x, y = _star_lines(ax)[0].get_xdata()[0], _star_lines(ax)[0].get_ydata()[0]
    dist_from_center = np.hypot(x - cx, y - cy)

    assert dist_from_center == pytest.approx(_marker_radius(tracker), rel=1e-6)

    plt.close(ax.figure)


def test_sky_radius_is_2x_surrounding_circle_radius(tracker):
    ax = tracker.plot()
    source = PointSource(offaxis_angle=0 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=1e-3 / u.cm / u.s)
    source.plot_sky_circle(ax, tracker)

    # The sky circle is drawn as a closed dotted line; its points should
    # all sit at exactly `2 x surrounding_circle_radius` from the centre.
    circle_line = ax.lines[-1]
    cx, cy = _center(tracker)
    x, y = np.asarray(circle_line.get_xdata()), np.asarray(circle_line.get_ydata())
    dist = np.hypot(x - cx, y - cy)

    expected = tracker.surrounding_circle_radius.to_value(LENGTH_UNIT) * 2
    np.testing.assert_allclose(dist, expected, rtol=1e-6)

    plt.close(ax.figure)


# --- Axes limits: the sky circle must actually be visible ------------------

def test_axes_limits_contain_the_sky_circle(tracker):
    # Regression guard: ToyTracker2D.plot() sets xlim/ylim to +-1.5x its own
    # surrounding radius, which is entirely inside the sky circle at 2x
    # that radius -- plotting a source must expand the limits to fit it.
    ax = tracker.plot()
    source = PointSource(offaxis_angle=0 * u.deg,
                         spectrum=MonoenergeticSpectrum(1 * u.MeV),
                         flux=1e-3 / u.cm / u.s)
    source.plot(ax, tracker)

    cx, cy = _center(tracker)
    r = _sky_radius(tracker)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    assert xlim[0] <= cx - r
    assert xlim[1] >= cx + r
    assert ylim[0] <= cy - r
    assert ylim[1] >= cy + r

    plt.close(ax.figure)


def test_two_sources_expand_limits_without_clobbering_each_other(tracker):
    ax = tracker.plot()

    source_up = PointSource(offaxis_angle=0 * u.deg,
                            spectrum=MonoenergeticSpectrum(1 * u.MeV),
                            flux=1e-3 / u.cm / u.s)
    source_right = PointSource(offaxis_angle=90 * u.deg,
                               spectrum=MonoenergeticSpectrum(1 * u.MeV),
                               flux=1e-3 / u.cm / u.s)

    source_up.plot(ax, tracker)
    xlim_after_first = ax.get_xlim()
    ylim_after_first = ax.get_ylim()

    source_right.plot(ax, tracker)
    xlim_after_second = ax.get_xlim()
    ylim_after_second = ax.get_ylim()

    # Limits only ever grow, never shrink.
    assert xlim_after_second[0] <= xlim_after_first[0]
    assert xlim_after_second[1] >= xlim_after_first[1]
    assert ylim_after_second[0] <= ylim_after_first[0]
    assert ylim_after_second[1] >= ylim_after_first[1]

    # Both stars are still inside the final limits.
    stars = _star_lines(ax)
    assert len(stars) == 2
    for line in stars:
        x, y = line.get_xdata()[0], line.get_ydata()[0]
        assert xlim_after_second[0] <= x <= xlim_after_second[1]
        assert ylim_after_second[0] <= y <= ylim_after_second[1]

    plt.close(ax.figure)


# --- Near-field source: star at its own position, not on the sky circle ----

def test_near_field_source_star_at_own_position(tracker):
    ax = tracker.plot()

    # Well inside the surrounding circle -- nowhere near the sky circle's
    # radius, which is the point of this test.
    position = Cartesian2D(1 * u.cm, 2 * u.cm)
    source = _StubNearFieldSource(position=position, rate=1 / u.s)

    source.plot(ax, tracker)

    stars = _star_lines(ax)
    assert len(stars) == 1
    x, y = stars[0].get_xdata()[0], stars[0].get_ydata()[0]

    assert x == pytest.approx(position.x.to_value(LENGTH_UNIT))
    assert y == pytest.approx(position.y.to_value(LENGTH_UNIT))

    # It is not sitting out at the marker radius used by far-field sources.
    cx, cy = _center(tracker)
    dist_from_center = np.hypot(x - cx, y - cy)
    assert dist_from_center < _sky_radius(tracker)

    plt.close(ax.figure)


def test_near_field_source_plot_does_not_draw_sky_circle(tracker):
    ax = tracker.plot()
    n_lines_before = len(ax.lines)

    source = _StubNearFieldSource(position=Cartesian2D(0 * u.cm, 3 * u.cm), rate=1 / u.s)
    source.plot(ax, tracker)

    # Exactly one new line (the star) -- no sky circle for a source that
    # isn't on the sky.
    assert len(ax.lines) == n_lines_before + 1

    plt.close(ax.figure)


# --- IsotropicSource: full 360 deg arc --------------------------------------

def test_isotropic_source_draws_full_circle_arc(tracker):
    ax = tracker.plot()
    source = IsotropicSource(spectrum=MonoenergeticSpectrum(1 * u.MeV),
                             flux=1e-3 / u.cm / u.s)

    source.plot(ax, tracker)

    # Last line drawn is the arc (the sky circle is drawn first).
    arc_line = ax.lines[-1]
    x, y = np.asarray(arc_line.get_xdata()), np.asarray(arc_line.get_ydata())

    cx, cy = _center(tracker)
    r = _marker_radius(tracker)

    # Every point of the arc sits at the marker radius from the centre...
    dist = np.hypot(x - cx, y - cy)
    np.testing.assert_allclose(dist, r, rtol=1e-6)

    # ...and it spans the full circle: the angles covered (measured the
    # same way as offaxis_angle, atan2(dx, dy)) range over a full 360 deg,
    # and the first and last points coincide (a closed loop).
    angles = np.unwrap(np.arctan2(x - cx, y - cy))
    assert (angles.max() - angles.min()) == pytest.approx(2 * np.pi, abs=1e-3)
    assert x[0] == pytest.approx(x[-1], abs=1e-6)
    assert y[0] == pytest.approx(y[-1], abs=1e-6)

    plt.close(ax.figure)


def test_isotropic_source_expands_axes_limits(tracker):
    ax = tracker.plot()
    source = IsotropicSource(spectrum=MonoenergeticSpectrum(1 * u.MeV),
                             flux=1e-3 / u.cm / u.s)
    source.plot(ax, tracker)

    cx, cy = _center(tracker)
    r = _sky_radius(tracker)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    assert xlim[0] <= cx - r
    assert xlim[1] >= cx + r
    assert ylim[0] <= cy - r
    assert ylim[1] >= cy + r

    plt.close(ax.figure)


def test_plot_returns_the_axes(tracker):
    ax = tracker.plot()
    far_source = PointSource(offaxis_angle=0 * u.deg,
                             spectrum=MonoenergeticSpectrum(1 * u.MeV),
                             flux=1e-3 / u.cm / u.s)
    iso_source = IsotropicSource(spectrum=MonoenergeticSpectrum(1 * u.MeV),
                                 flux=1e-3 / u.cm / u.s)
    near_source = _StubNearFieldSource(position=Cartesian2D(0 * u.cm, 1 * u.cm))

    assert far_source.plot(ax, tracker) is ax
    assert iso_source.plot(ax, tracker) is ax
    assert near_source.plot(ax, tracker) is ax

    plt.close(ax.figure)
