"""Tests for time-dependent source scaling (`docs/dev/inertial_sim_plan.md`,
Section 5.7, PR 6 entry in Section 7).

Sizing note, matching the house style of `tests/test_inertial_simulator.py`
and `tests/test_earth_albedo.py`: every statistical assertion below states
the Poisson sigma it is built on and asserts at **4 sigma**, one wider than
the plan's "within 3 sigma".

The `TabulatedScaling` semantics (piecewise-constant lookup, breakpoint
ownership, clamping) are checked against the 12-row table worked out by hand
in `/tmp/claude-0/-home-user-gammaraytoys/4a1d53f1-f083-5ad3-bf87-1c801ee0fb3a/scratchpad/pr6/CONTRACT.md`,
directly from the plan's rule ("the value at `t` is the scale from the last
row whose time is `<= t`") -- before any of this code existed. None of the
expected values below was obtained by running the implementation and
reading its output back.
"""

import astropy.units as u
import numpy as np
import pytest

from gammaraytoys import ToyTracker2D
from gammaraytoys.sims import (ConstantScaling, Earth, EarthAlbedoSource,
                               ExtendedSource, FunctionScaling, InertialSimulator,
                               IsotropicSource, MonoenergeticSpectrum, NearFieldSource,
                               NearPointSource, Photon, PointSource,
                               SimpleTraditionalReconstructor, SpacecraftHistory,
                               TabulatedScaling)
from gammaraytoys.coordinates import Cartesian2D


def _make_tracker():
    return ToyTracker2D(material='Ge',
                        layer_length=16 * u.cm,
                        layer_positions=[0, 5, 10, 20, 25, 30] * u.mm,
                        layer_thickness=5 * u.mm,
                        energy_resolution=0.01,
                        energy_threshold=20 * u.keV)


SPECTRUM = MonoenergeticSpectrum(1 * u.MeV)


# ===========================================================================
# Part A -- TabulatedScaling semantics (TEST_BRIEF item 5)
# ===========================================================================

# The 12-row table from CONTRACT.md, derived from the plan's rule before any
# code existed:
#     table = [(100, 1.0), (200, 2.5), (350, 0.5), (400, 3.0)]
_BREAKPOINT_TIMES = [100.0, 200.0, 350.0, 400.0]
_BREAKPOINT_SCALES = [1.0, 2.5, 0.5, 3.0]

_CONTRACT_TABLE = [
    (-1e9, 1.0),      # far before -> clamps to first
    (0.0, 1.0),       # before first
    (99.999, 1.0),    # just before first
    (100.0, 1.0),     # exactly on a breakpoint -> that row's value
    (100.001, 1.0),
    (150.0, 1.0),     # between rows
    (200.0, 2.5),     # exactly on a breakpoint
    (349.999, 2.5),
    (350.0, 0.5),     # exactly on a breakpoint
    (399.999, 0.5),
    (400.0, 3.0),     # exactly on the last row
    (1e9, 3.0),       # far after -> clamps to last
]


def _make_tabulated_scaling():
    return TabulatedScaling(time=np.array(_BREAKPOINT_TIMES) * u.s,
                            scale=_BREAKPOINT_SCALES)


@pytest.mark.parametrize('t, expected', _CONTRACT_TABLE)
def test_tabulated_scaling_matches_contract_table(t, expected):
    # The full 12-row table pinned in CONTRACT.md. In particular, t=350.0
    # (the row exactly on a breakpoint that changed value) is what a
    # `searchsorted(..., side='left')` mutation gets wrong: it would return
    # the OLD row's value (2.5), not the new one (0.5).
    scaling = _make_tabulated_scaling()
    assert scaling(t * u.s) == expected


def test_tabulated_scaling_breakpoint_belongs_to_the_new_row():
    # Semantic 1 of 3: a breakpoint belongs to the row AT it (`<=`), not the
    # previous one. At t=350.0 exactly the value must be the NEW 0.5, not
    # the OLD 2.5 that held on [200, 350). This is the exact failure mode of
    # `searchsorted(..., side='left')`.
    scaling = _make_tabulated_scaling()
    assert scaling(350.0 * u.s) == 0.5
    assert scaling(350.0 * u.s) != 2.5


def test_tabulated_scaling_is_flat_and_right_continuous():
    # Semantic 2 of 3: the value is flat on [t_i, t_{i+1}) -- the instant
    # before a breakpoint still holds the OLD value, and the breakpoint
    # itself already holds the NEW one.
    scaling = _make_tabulated_scaling()
    assert scaling(349.999 * u.s) == 2.5
    assert scaling(350.0 * u.s) == 0.5


def test_tabulated_scaling_clamps_and_never_extrapolates():
    # Semantic 3 of 3: outside the table it clamps to the first/last value.
    # A far-outside t must not raise (no extrapolation/IndexError) and must
    # not drift from the boundary value.
    scaling = _make_tabulated_scaling()
    assert scaling(-1e9 * u.s) == 1.0
    assert scaling(1e9 * u.s) == 3.0
    # Right at the edges too.
    assert scaling(0.0 * u.s) == 1.0
    assert scaling(400.0 * u.s) == 3.0


def test_tabulated_scaling_single_row_is_constant_everywhere():
    scaling = TabulatedScaling(time=[50.0] * u.s, scale=[7.0])
    for t in [-1e9, 0.0, 49.999, 50.0, 50.001, 1e9]:
        assert scaling(t * u.s) == 7.0


def test_tabulated_scaling_open_round_trips_a_csv_file(tmp_path):
    path = tmp_path / 'scaling.csv'
    path.write_text('time_s,scale\n' +
                    '\n'.join(f'{t},{s}' for t, s in
                              zip(_BREAKPOINT_TIMES, _BREAKPOINT_SCALES)) + '\n')

    scaling = TabulatedScaling.open(path)

    for t, expected in _CONTRACT_TABLE:
        assert scaling(t * u.s) == expected


# ===========================================================================
# Part B -- validation guards (TEST_BRIEF items 7-9)
# ===========================================================================

def test_tabulated_scaling_rejects_empty_table():
    with pytest.raises(ValueError):
        TabulatedScaling(time=[] * u.s, scale=[])


def test_tabulated_scaling_rejects_unsorted_times():
    # [200, 100] is not ascending.
    with pytest.raises(ValueError):
        TabulatedScaling(time=[200.0, 100.0] * u.s, scale=[1.0, 2.0])


def test_tabulated_scaling_rejects_duplicate_times():
    with pytest.raises(ValueError):
        TabulatedScaling(time=[100.0, 100.0, 200.0] * u.s, scale=[1.0, 2.0, 3.0])


def test_tabulated_scaling_rejects_negative_scale():
    with pytest.raises(ValueError):
        TabulatedScaling(time=[100.0, 200.0] * u.s, scale=[1.0, -2.0])


def test_tabulated_scaling_rejects_nan_scale():
    with pytest.raises(ValueError):
        TabulatedScaling(time=[100.0, 200.0] * u.s, scale=[1.0, np.nan])


def test_constant_scaling_rejects_negative():
    with pytest.raises(ValueError):
        ConstantScaling(-1.0)


def test_constant_scaling_rejects_nan():
    with pytest.raises(ValueError):
        ConstantScaling(np.nan)


def test_constant_scaling_scale_setter_validates_after_construction():
    # Regression for fix commit 546d840: `scale` used to be a plain
    # attribute, so `c.scale = -3.0` was accepted silently and only
    # surfaced later, far from here, as `ValueError: lam < 0 or lam is
    # NaN` raised by `numpy.random.poisson` deep inside a run -- exactly
    # the failure `_validate_scale`'s own docstring says the module exists
    # to prevent. `scale` is now a validating property, mirroring
    # `Source.scaling`.
    scaling = ConstantScaling(1.0)

    with pytest.raises(ValueError):
        scaling.scale = -3.0

    with pytest.raises(ValueError):
        scaling.scale = np.nan

    # And the object is still usable afterwards, at its last-good value --
    # a rejected assignment must not have corrupted it.
    assert scaling(0 * u.s) == 1.0


def test_function_scaling_rejects_negative_return():
    scaling = FunctionScaling(lambda t: -1.0)
    with pytest.raises(ValueError):
        scaling(0 * u.s)


def test_function_scaling_rejects_nan_return():
    scaling = FunctionScaling(lambda t: np.nan)
    with pytest.raises(ValueError):
        scaling(0 * u.s)


def test_function_scaling_rejects_non_numeric_return():
    scaling = FunctionScaling(lambda t: 'not a number')
    with pytest.raises(ValueError):
        scaling(0 * u.s)


def test_function_scaling_returns_the_callables_value_and_passes_time_through():
    # No prior test in this file ever observes a `FunctionScaling`'s return
    # value: the three above all pass callables whose bad return raises
    # before it is ever read. Pin both halves of the contract: the
    # returned value really is the callable's own value (not, say, a
    # hardcoded 1.0 after validation), and `time` really reaches the
    # callable unchanged (not, say, always called with `0.0 * u.s`).
    seen_times = []

    def constant_2p5(t):
        seen_times.append(t)
        return 2.5

    scaling = FunctionScaling(constant_2p5)
    result = scaling(123.0 * u.s)

    assert result == 2.5
    assert type(result) is float
    assert seen_times == [123.0 * u.s]

    # A callable whose return value actually depends on `time`: if `time`
    # were dropped in favour of some fixed argument, every call below
    # would return the same (wrong) number regardless of `t`.
    def linear_in_seconds(t):
        return t.to_value(u.s) / 100.0

    scaling2 = FunctionScaling(linear_in_seconds)
    assert scaling2(300.0 * u.s) == pytest.approx(3.0)
    assert scaling2(500.0 * u.s) == pytest.approx(5.0)
    assert scaling2(300.0 * u.s) != scaling2(500.0 * u.s)


# ===========================================================================
# Part C -- default scaling on every source, and its no-op guarantee
# (TEST_BRIEF item 10)
# ===========================================================================

def test_default_scaling_returns_the_exact_float_one():
    # `ConstantScaling(1.0)` is what `scaling = None` resolves to on every
    # source. Its `__call__` must return the *exact* Python float 1.0 --
    # IEEE 754 guarantees `x * 1.0 == x` bit-for-bit for every finite x (no
    # rounding), so multiplying it into the Poisson mean (Section 6) changes
    # nothing at all relative to the pre-PR6 formula `simulated_rate *
    # livetime` with no scaling term. This is what makes the default an
    # exact no-op: any seeded run through the new, multiplied formula draws
    # bit-identical Poisson means -- and hence the same random sequence --
    # as it would through the old, unscaled one.
    scaling = ConstantScaling(1.0)

    for t in [0.0, 1.0, -1.0, 0.5, 1e9, -1e9]:
        value = scaling(t * u.s)
        assert value == 1.0
        assert type(value) is float


def _sources_with_default_scaling(detector):
    earth = Earth(radius=6371.0 * u.km)
    return {
        'PointSource': PointSource(sky_angle=0 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s),
        'IsotropicSource': IsotropicSource(spectrum=SPECTRUM, flux=1 / u.cm / u.s),
        'NearPointSource': NearPointSource(position=Cartesian2D(0 * u.cm, 0 * u.cm),
                                           spectrum=SPECTRUM, rate=1 * u.Hz),
        'ExtendedSource': ExtendedSource(sky_angle=0 * u.deg, width=10 * u.deg,
                                         spectrum=SPECTRUM, flux=1 / u.cm / u.s),
        'EarthAlbedoSource': EarthAlbedoSource(emissivity=1 / u.cm / u.s,
                                               spectrum=SPECTRUM, earth=earth),
    }


@pytest.mark.parametrize('name', ['PointSource', 'IsotropicSource', 'NearPointSource',
                                  'ExtendedSource', 'EarthAlbedoSource'])
def test_scaling_none_defaults_to_constant_scaling_one(name):
    detector = _make_tracker()
    source = _sources_with_default_scaling(detector)[name]

    assert isinstance(source.scaling, ConstantScaling)
    assert source.scaling(12345.6 * u.s) == 1.0


def test_scaling_setter_rejects_a_non_scaling_value():
    source = PointSource(sky_angle=0 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s)
    with pytest.raises(TypeError):
        source.scaling = 2.0


class _BareSource(NearFieldSource):
    """Same shape as `DemoNearFieldSource` in
    `docs/examples/cosimita/00-source_normalization.ipynb`: a `Source`
    subclass that never calls a base `__init__` at all (it sets its own
    attributes directly and implements every abstract member itself), so
    no constructor ever runs `self.scaling = ...` or otherwise sets
    `self._scaling`."""

    def __init__(self, position, rate, spectrum):
        self._position = position
        self._rate = rate
        self._spectrum = spectrum

    @property
    def position(self):
        return self._position

    @property
    def rate(self):
        return self._rate

    @property
    def spectrum(self):
        return self._spectrum

    def simulated_rate(self, detector, pose=None):
        return self.rate

    def random_photon(self, detector, pose=None, earth=None):
        return Photon(position=self.position,
                      direction=np.random.uniform(0, 360) * u.deg,
                      energy=self.spectrum.random_energy())


def test_source_subclass_without_base_init_still_has_a_default_scaling():
    # Regression for fix commit 546d840: before it, `.scaling`'s getter
    # unconditionally read `self._scaling`, so a subclass shaped like
    # `_BareSource` -- never running a base `__init__` -- raised
    # `AttributeError: no attribute '_scaling'` the first time anything
    # touched `.scaling`. `Source._scaling`'s class-level default (`None`)
    # fixes this for any such third-party source, not just ones written
    # after `scaling` existed.
    source = _BareSource(position=Cartesian2D(0 * u.cm, 0 * u.cm),
                         rate=2 * u.Hz, spectrum=SPECTRUM)

    assert isinstance(source.scaling, ConstantScaling)
    assert source.scaling(0 * u.s) == 1.0


def test_source_subclass_without_base_init_runs_through_inertial_simulator():
    # The same regression, end to end: `InertialSimulator.run_events`
    # touches `source.scaling(...)` on every interval (Section 6), so this
    # is exactly where the pre-fix AttributeError actually surfaced.
    np.random.seed(20260906)

    detector = _make_tracker()
    earth = Earth(radius=6371.0 * u.km)
    history = SpacecraftHistory(time=[0.0, 500.0] * u.s,
                                orbit_radius=[7000.0, 7000.0] * u.km,
                                orbit_angle=[0.0, 0.0] * u.deg,
                                attitude=[0.0, 0.0] * u.deg,
                                uptime=[500.0, 0.0] * u.s,
                                earth=earth)

    source = _BareSource(position=Cartesian2D(0 * u.cm, 0 * u.cm),
                         rate=5 * u.Hz, spectrum=SPECTRUM)

    simulator = InertialSimulator(detector=detector, sources=[source],
                                  reconstructor=SimpleTraditionalReconstructor(),
                                  spacecraft_history=history, earth=earth)

    events = list(simulator.run_events(progress=False))
    assert len(events) > 0


# ===========================================================================
# Part D -- InertialSimulator actually uses source.scaling, at mid_time
# (TEST_BRIEF item 6; mutation targets (c) and (d))
# ===========================================================================

EARTH_RADIUS = 6371.0 * u.km
ORBIT_RADIUS = 7000.0 * u.km
# rho = arcsin(6371/7000) = 65.67 deg; with attitude = 90 deg (on-axis at
# sky_angle = 90 deg -> nadir at orbit_angle + 180 = 180 deg), the
# source-to-nadir separation is 90 deg > rho, so it is never occulted.


def _one_interval_history(livetime_s=1000.0):
    """A single-interval `SpacecraftHistory`: orbit_angle=0, attitude=90 deg
    held fixed, at `ORBIT_RADIUS`, for `livetime_s` seconds with full
    livetime. Built directly from row arrays (Section 4.3's `__init__`) so
    every geometric quantity is exactly what was asked for -- no orbit
    generator involved."""

    earth = Earth(radius=EARTH_RADIUS)
    history = SpacecraftHistory(time=[0.0, livetime_s] * u.s,
                                orbit_radius=[ORBIT_RADIUS.to_value(u.km)] * 2 * u.km,
                                orbit_angle=[0.0, 0.0] * u.deg,
                                attitude=[90.0, 90.0] * u.deg,
                                uptime=[livetime_s, 0.0] * u.s,
                                earth=earth)
    return history, earth


def _run_point_source(mu_unscaled, scaling, detector, history, earth):
    """Run a single `PointSource(sky_angle=90 deg)` (on-axis, never
    occulted -- see `_one_interval_history`) for the history's one interval,
    normalized so that with `ConstantScaling(1.0)` the expected count is
    exactly `mu_unscaled` (Sections 5.2, 6:
    `mu = flux * throwing_plane_size * livetime * scaling`)."""

    interval = next(iter(history))
    livetime = interval.livetime

    flux = (mu_unscaled / (detector.throwing_plane_size * livetime)).to(1 / u.cm / u.s)

    source = PointSource(sky_angle=90 * u.deg, spectrum=SPECTRUM, flux=flux, scaling=scaling)

    simulator = InertialSimulator(detector=detector, sources=source,
                                  reconstructor=SimpleTraditionalReconstructor(),
                                  spacecraft_history=history, earth=earth)

    return list(simulator.run_events(progress=False))


def test_scaling_of_two_doubles_counts_relative_to_scaling_of_one():
    # Section 6: mu = simulated_rate * livetime * scaling(mid_time). With
    # scaling=1 the expected count is MU (by construction, see
    # `_run_point_source`); with scaling=2 it must be 2*MU.
    #
    # sigma(MU) = sqrt(MU), asserted at 4 sigma. Mutation (c) -- dropping
    # the `source.scaling(...)` multiply entirely -- would give the same
    # expected count (MU) in both runs, off by MU = many multiples of
    # sigma(2*MU), so this is caught with enormous margin.
    np.random.seed(20260904)

    MU = 600.0
    detector = _make_tracker()
    history, earth = _one_interval_history()

    n1 = len(_run_point_source(MU, ConstantScaling(1.0), detector, history, earth))
    n2 = len(_run_point_source(MU, ConstantScaling(2.0), detector, history, earth))

    sigma1 = np.sqrt(MU)
    assert abs(n1 - MU) < 4 * sigma1

    sigma2 = np.sqrt(2 * MU)
    assert abs(n2 - 2 * MU) < 4 * sigma2


def test_function_scaling_scales_a_real_inertial_simulator_run():
    # The same guarantee as the test above, through a `FunctionScaling`
    # instead of a `ConstantScaling` -- nothing else in this file runs a
    # real `InertialSimulator` with one.
    np.random.seed(20260905)

    MU = 600.0
    detector = _make_tracker()
    history, earth = _one_interval_history()

    scaling = FunctionScaling(lambda t: 2.0)
    events = _run_point_source(MU, scaling, detector, history, earth)

    expected = 2 * MU
    sigma = np.sqrt(expected)
    assert abs(len(events) - expected) < 4 * sigma


def test_scaling_evaluated_at_interval_midpoint_not_start_time():
    # Mutation (d): applying the scaling at `start_time` instead of
    # `mid_time`. Built so the two give provably different answers: the
    # single interval spans [0, 1000) s (mid_time = 500 s), and the
    # TabulatedScaling has a breakpoint at t=250 s -- strictly between
    # start_time and mid_time -- so scaling(start_time) = 1.0 but
    # scaling(mid_time) = 3.0.
    #
    # sigma(MU) = sqrt(MU) for the correct (mid_time) expectation; the
    # start_time mutant's expectation is 3x smaller (MU vs 3*MU), which at
    # MU=600 is off by 1200 -- about 28.3 sigma of sqrt(3*600) = 42.43 --
    # so this is not a marginal test.
    np.random.seed(20260904)

    MU = 600.0
    detector = _make_tracker()
    history, earth = _one_interval_history(livetime_s=1000.0)

    interval = next(iter(history))
    assert interval.start_time.to_value(u.s) == 0.0
    assert interval.mid_time.to_value(u.s) == 500.0

    scaling = TabulatedScaling(time=[0.0, 250.0] * u.s, scale=[1.0, 3.0])
    assert scaling(interval.start_time) == 1.0
    assert scaling(interval.mid_time) == 3.0

    events = _run_point_source(MU, scaling, detector, history, earth)

    expected = 3 * MU
    sigma = np.sqrt(expected)
    assert abs(len(events) - expected) < 4 * sigma


def test_expected_counts_also_uses_scaling_at_interval_midpoint():
    # `InertialSimulator` evaluates `source.scaling(...)` at TWO call
    # sites: once inside `run_events` (the actual Poisson draw, pinned by
    # the test above) and once inside `_expected_counts` (the tqdm
    # progress-bar total -- and, per `tests/test_earth_albedo.py`'s
    # `test_expected_counts_matches_independent_total_on_an_eccentric_orbit`
    # -pattern tests, the value this suite already treats as the
    # authoritative expected photon count for a run). With a merely
    # constant scaling the two sites cannot be told apart; this needs a
    # genuinely time-varying one, with a distinct value change strictly
    # *inside* each interval, so `scaling(start_time) != scaling(mid_time)`
    # for every interval in the run:
    #
    #   interval 0 = [0, 1000) s, mid_time = 500 s
    #   interval 1 = [1000, 2000) s, mid_time = 1500 s
    #   TabulatedScaling breakpoints at t = 250 s and t = 1250 s (strictly
    #   inside interval 0 and interval 1 respectively):
    #     interval 0: scaling(start=0)    = 1.0, scaling(mid=500)  = 3.0
    #     interval 1: scaling(start=1000) = 3.0, scaling(mid=1500) = 5.0
    detector = _make_tracker()
    earth = Earth(radius=EARTH_RADIUS)

    history = SpacecraftHistory(time=[0.0, 1000.0, 2000.0] * u.s,
                                orbit_radius=[ORBIT_RADIUS.to_value(u.km)] * 3 * u.km,
                                orbit_angle=[0.0, 0.0, 0.0] * u.deg,
                                attitude=[90.0, 90.0, 90.0] * u.deg,
                                uptime=[1000.0, 1000.0, 0.0] * u.s,
                                earth=earth)

    scaling = TabulatedScaling(time=[0.0, 250.0, 1250.0] * u.s, scale=[1.0, 3.0, 5.0])

    flux = 2.0 / u.cm / u.s
    source = PointSource(sky_angle=90 * u.deg, spectrum=SPECTRUM, flux=flux, scaling=scaling)

    simulator = InertialSimulator(detector=detector, sources=[source],
                                  reconstructor=SimpleTraditionalReconstructor(),
                                  spacecraft_history=history, earth=earth)

    actual = simulator._expected_counts()

    # Section 6: mu = simulated_rate(detector, pose) * livetime *
    # scaling(mid_time), summed over intervals. simulated_rate = flux *
    # throwing_plane_size (Section 5.2) is pose-independent for a
    # PointSource, so this is a plain per-interval sum -- derived here from
    # the plan's formula, not from calling `_expected_counts` itself.
    rate_hz = (flux * detector.throwing_plane_size).to_value(u.Hz)
    expected = sum(rate_hz * interval.livetime.to_value(u.s) * scaling(interval.mid_time)
                  for interval in history)

    assert actual == pytest.approx(expected, rel=1e-12)

    # Spelled out: using start_time instead would give a different, smaller
    # total (4000 * rate_hz vs. 8000 * rate_hz), so this is not a
    # coincidental match.
    wrong_using_start_time = sum(
        rate_hz * interval.livetime.to_value(u.s) * scaling(interval.start_time)
        for interval in history)
    assert expected != pytest.approx(wrong_using_start_time, rel=1e-6)
