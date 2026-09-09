"""Tests for event CSV I/O (`docs/dev/inertial_sim_plan.md`, Section 6,
PR 6 entry in Section 7).

`true_sky_angle_deg` recovery (items 11-13 of the PR 6 test brief) is
checked against `lambda = A - Nu`, wrapped to `[-180, 180)`, computed here
with a *separate* implementation from `gammaraytoys.sims.event_csv`'s own
`_true_sky_angle_deg`: the attitude `A` is read directly off the exact
`SpacecraftInterval` each photon was drawn from (never by re-running any
timestamp search), and the wrap is a plain-Python one-liner, not a call into
the module under test. None of the expected values below was obtained by
running the implementation and reading its output back.
"""

import numpy as np
import pytest
import astropy.units as u

from gammaraytoys import ToyTracker2D
from gammaraytoys.sims import (Earth, EarthAlbedoSource, ExtendedSource,
                               InertialSimulator, IsotropicSource, MonoenergeticSpectrum,
                               NearPointSource, PointSource, SimpleTraditionalReconstructor,
                               SpacecraftHistory, read_event_csv, write_event_csv)
from gammaraytoys.sims.event import Photon
from gammaraytoys.sims.event_csv import _attitude_at, _attitude_lookup_table
from gammaraytoys.sims.reco import RecoCompton
from gammaraytoys.coordinates import Cartesian2D


def _make_tracker():
    return ToyTracker2D(material='Ge',
                        layer_length=16 * u.cm,
                        layer_positions=[0, 5, 10, 20, 25, 30] * u.mm,
                        layer_thickness=5 * u.mm,
                        energy_resolution=0.01,
                        energy_threshold=20 * u.keV)


SPECTRUM = MonoenergeticSpectrum(1 * u.MeV)

_ALL_COLUMNS = ['event_id', 'time_s', 'source',
               'true_x_cm', 'true_y_cm', 'true_direction_deg',
               'true_sky_angle_deg', 'true_offaxis_angle_deg',
               'true_energy_MeV', 'true_chirality',
               'triggered',
               'reco_energy_MeV', 'reco_phi_deg', 'reco_psi_deg']


def _photon(x_cm, y_cm, direction_deg, energy_mev, chirality):
    return Photon(position=Cartesian2D(x_cm * u.cm, y_cm * u.cm),
                 direction=direction_deg * u.deg,
                 energy=energy_mev * u.MeV,
                 chirality=chirality)


def _triggered(energy_mev, phi_deg, psi_deg):
    return RecoCompton(energy=energy_mev * u.MeV, phi=phi_deg * u.deg, psi=psi_deg * u.deg)


def _untriggered():
    return RecoCompton()


# ===========================================================================
# Part A -- round trip, triggered_only, metadata, empty file
# (TEST_BRIEF items 1-4)
# ===========================================================================

@pytest.fixture
def sky_source():
    return PointSource(sky_angle=40 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s)


@pytest.fixture
def near_source():
    return NearPointSource(position=Cartesian2D(0 * u.cm, 0 * u.cm), spectrum=SPECTRUM,
                           rate=1 * u.Hz)


@pytest.fixture
def mixed_events(sky_source, near_source):
    """Four events, hand-built (no simulator involved), covering every
    combination the round-trip must preserve: triggered vs. untriggered
    (reco_* NaN iff untriggered), and a far-field vs. a near-field source
    (true_sky_angle_deg NaN for the near one)."""

    return [
        (0.0 * u.s, sky_source, _photon(1.5, -2.25, 10.0, 0.662, 1), _triggered(0.6, 33.3, -12.5)),
        (10.25 * u.s, sky_source, _photon(-3.0, 0.0, 190.0, 0.511, -1), _untriggered()),
        (20.5 * u.s, near_source, _photon(0.1, 0.2, 300.0, 1.5, 1), _triggered(1.4, -170.0, 5.0)),
        (30.75 * u.s, near_source, _photon(-0.1, -0.2, 5.0, 0.9, -1), _untriggered()),
    ]


def test_round_trip_preserves_units_and_nans(tmp_path, mixed_events, sky_source, near_source):
    path = tmp_path / 'events.csv'

    write_event_csv(path, mixed_events,
                    source_names={sky_source: 'crab', near_source: 'calib'},
                    ori_file='test.ori', total_livetime=41.0 * u.s)

    metadata, table = read_event_csv(path)

    assert len(table) == 4
    assert list(table.columns) == _ALL_COLUMNS

    # Row 0: triggered, far-field source with a fixed sky_angle -- every
    # true_* and reco_* field must round-trip exactly.
    row = table.iloc[0]
    assert row['event_id'] == 0
    assert row['time_s'] == 0.0
    assert row['source'] == 'crab'
    assert row['true_x_cm'] == 1.5
    assert row['true_y_cm'] == -2.25
    assert row['true_direction_deg'] == 10.0
    assert row['true_sky_angle_deg'] == pytest.approx(40.0)
    # true_offaxis_angle_deg = wrap(270 - direction) = wrap(260) = -100.
    assert row['true_offaxis_angle_deg'] == pytest.approx(-100.0)
    assert row['true_energy_MeV'] == 0.662
    assert row['true_chirality'] == 1
    assert bool(row['triggered']) is True
    assert row['reco_energy_MeV'] == 0.6
    assert row['reco_phi_deg'] == 33.3
    assert row['reco_psi_deg'] == -12.5

    # Row 1: untriggered -- reco_* must be NaN, true_* still populated.
    row = table.iloc[1]
    assert row['time_s'] == 10.25
    assert row['true_x_cm'] == -3.0
    assert row['true_sky_angle_deg'] == pytest.approx(40.0)
    assert bool(row['triggered']) is False
    assert np.isnan(row['reco_energy_MeV'])
    assert np.isnan(row['reco_phi_deg'])
    assert np.isnan(row['reco_psi_deg'])

    # Row 2: near-field source -- true_sky_angle_deg must be NaN regardless
    # of triggered.
    row = table.iloc[2]
    assert row['source'] == 'calib'
    assert np.isnan(row['true_sky_angle_deg'])
    assert bool(row['triggered']) is True
    assert row['reco_energy_MeV'] == 1.4

    # Row 3: near-field AND untriggered -- both NaN reasons apply at once.
    row = table.iloc[3]
    assert np.isnan(row['true_sky_angle_deg'])
    assert np.isnan(row['reco_phi_deg'])


def test_metadata_header_survives_round_trip(tmp_path, mixed_events, sky_source, near_source):
    path = tmp_path / 'events.csv'

    written = write_event_csv(path, mixed_events,
                              source_names={sky_source: 'crab', near_source: 'calib'},
                              ori_file='iss.ori', total_livetime=5130.0 * u.s)

    # The file really does carry a leading '#' block (mutation (g): drop the
    # header entirely).
    with open(path) as f:
        first_line = f.readline()
    assert first_line.startswith('#')

    metadata, table = read_event_csv(path)

    assert metadata == written
    assert metadata['ori_file'] == 'iss.ori'
    assert metadata['total_livetime_s'] == 5130.0
    assert metadata['triggered_only'] is False
    # nsim counts every event handed to write_event_csv, by label, whether
    # or not it was written as a row.
    assert metadata['nsim'] == {'crab': 2, 'calib': 2}
    assert 'gammaraytoys_version' in metadata


def test_empty_file_still_reads_back(tmp_path):
    path = tmp_path / 'empty.csv'

    write_event_csv(path, [])

    metadata, table = read_event_csv(path)

    assert table.shape == (0, 14)
    assert list(table.columns) == _ALL_COLUMNS
    assert metadata['nsim'] == {}


def test_triggered_only_drops_exactly_the_untriggered_rows(tmp_path, mixed_events,
                                                            sky_source, near_source):
    # Pattern across the 4 fixture events: triggered, untriggered,
    # triggered, untriggered (event_ids 0, 1, 2, 3).
    path = tmp_path / 'triggered_only.csv'

    metadata = write_event_csv(path, mixed_events,
                               source_names={sky_source: 'crab', near_source: 'calib'},
                               triggered_only=True)

    assert metadata['triggered_only'] is True
    # nsim still counts every event, not just the written ones.
    assert metadata['nsim'] == {'crab': 2, 'calib': 2}

    _, table = read_event_csv(path)

    # Exactly the triggered rows, nothing more and nothing less: ids 0 and
    # 2, each carrying the true_energy_MeV that uniquely identifies it, and
    # every remaining row's `triggered` is True (an off-by-one that dropped
    # or kept the wrong row would show up in either the id set or this).
    assert sorted(table['event_id'].tolist()) == [0, 2]
    assert sorted(table['true_energy_MeV'].tolist()) == [0.662, 1.5]
    assert table['triggered'].all()

    # event_id is *not* renumbered densely -- it still identifies the same
    # underlying photon as an untriggered_only=False file would.
    assert table.loc[table['event_id'] == 2, 'source'].iloc[0] == 'calib'


def test_triggered_only_off_by_one_would_be_caught(tmp_path, mixed_events,
                                                    sky_source, near_source):
    # A second, independent angle on the same guarantee as the test above,
    # phrased as a pure count: of 4 events with 2 triggered, triggered_only
    # must write exactly 2 rows -- not 1 (a mutation dropping one true
    # trigger) and not 3 (a mutation keeping one false trigger).
    path = tmp_path / 'triggered_only_count.csv'

    write_event_csv(path, mixed_events,
                    source_names={sky_source: 'crab', near_source: 'calib'},
                    triggered_only=True)

    _, table = read_event_csv(path)
    assert len(table) == 2


def test_default_columns_are_in_contract_order(tmp_path, mixed_events, sky_source, near_source):
    path = tmp_path / 'events.csv'
    write_event_csv(path, mixed_events, source_names={sky_source: 'crab', near_source: 'calib'})

    _, table = read_event_csv(path)

    assert list(table.columns) == [
        'event_id', 'time_s', 'source', 'true_x_cm', 'true_y_cm', 'true_direction_deg',
        'true_sky_angle_deg', 'true_offaxis_angle_deg', 'true_energy_MeV', 'true_chirality',
        'triggered', 'reco_energy_MeV', 'reco_phi_deg', 'reco_psi_deg',
    ]


def test_source_falls_back_to_class_name_without_source_names(tmp_path, sky_source):
    path = tmp_path / 'events.csv'
    events = [(0.0 * u.s, sky_source, _photon(0, 0, 0, 1, 1), _untriggered())]

    write_event_csv(path, events)  # no source_names

    _, table = read_event_csv(path)
    assert table['source'].iloc[0] == 'PointSource'


# ===========================================================================
# Part B -- true_sky_angle_deg recovery (TEST_BRIEF items 11-13)
# ===========================================================================

def _wrap180(angle_deg):
    """Independent wrap, not `gammaraytoys.sims.event_csv._wrap180`."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _independent_expected_sky_angle_deg(attitude_deg, offaxis_deg):
    """`lambda = A - Nu`, wrapped -- computed here, not via
    `gammaraytoys.coordinates.transform.offaxis_to_sky_angle` nor via
    `event_csv._true_sky_angle_deg`."""
    return _wrap180(attitude_deg - offaxis_deg)


def _varying_attitude_history(earth, orbit_radius_km, n_intervals=6, interval_s=100.0):
    """A `SpacecraftHistory` with `n_intervals` intervals of equal length,
    orbit_angle held at 0 (irrelevant to this section -- occultation is
    avoided by `orbit_radius_km` being large relative to `earth`, or is
    simply not tested here), and an attitude that visibly rotates from
    interval to interval, built directly from row arrays so every attitude
    value is exactly known (not read back from an orbit generator)."""

    n_rows = n_intervals + 1
    time_s = np.arange(n_rows) * interval_s
    # A non-uniform, wide spread of attitudes -- deliberately not evenly
    # spaced multiples of a fixed step, so no accidental symmetry could mask
    # a wrong-interval attitude lookup.
    attitude_deg = np.array([0.0, 53.0, 107.0, 161.0, 214.0, 268.0, 268.0])[:n_rows]

    history = SpacecraftHistory(
        time=time_s * u.s,
        orbit_radius=[orbit_radius_km] * n_rows * u.km,
        orbit_angle=np.zeros(n_rows) * u.deg,
        attitude=attitude_deg * u.deg,
        uptime=np.full(n_rows, interval_s) * u.s,
        earth=earth)

    return history


def test_attitude_at_owns_start_time_and_clamps_before_the_first_row():
    # Direct unit tests of `_attitude_at`/`_attitude_lookup_table`
    # themselves, at the exact boundary values where the three surviving
    # mutations the coordinator found show up:
    #   M3: `_attitude_lookup_table` keyed on mid_time instead of
    #       start_time -- caught by checking exactly AT each interval's
    #       own start_time, which mid_time-keying gets wrong for every
    #       interval but the first.
    #   M4: `side='left'` -- caught the same way `TabulatedScaling`'s
    #       breakpoint-ownership test catches it: AT start_time must give
    #       the NEW (this) interval's attitude, not the previous one.
    #   M5: the clamp removed -- caught by a time before the very first
    #       row, which must clamp rather than raise or wrap around
    #       (Python's negative-index wraparound would otherwise silently
    #       return the LAST row's attitude).
    history = _varying_attitude_history(_FAR_EARTH, _FAR_ORBIT_RADIUS_KM)
    intervals = list(history)
    start_times_s, attitudes_deg = _attitude_lookup_table(history)

    eps = 1e-6

    for i, interval in enumerate(intervals):
        start_s = interval.start_time.to_value(u.s)
        stop_s = interval.stop_time.to_value(u.s)
        this_attitude = interval.attitude.to_value(u.deg)

        # AT this interval's own start_time -> THIS interval's attitude
        # (owned by the row at it, not the previous one).
        assert _attitude_at(start_s, start_times_s, attitudes_deg) == pytest.approx(this_attitude)

        # Just before this interval's own stop_time -> still THIS
        # interval's attitude (flat, right-continuous on
        # [start_time, stop_time)).
        assert _attitude_at(stop_s - eps, start_times_s, attitudes_deg) == pytest.approx(this_attitude)

        # Just before this interval's own start_time (every interval but
        # the first) -> the PREVIOUS interval's attitude, not this one.
        if i > 0:
            previous_attitude = intervals[i - 1].attitude.to_value(u.deg)
            assert _attitude_at(start_s - eps, start_times_s, attitudes_deg) == pytest.approx(
                previous_attitude)

    # Before the very first row -> clamps to the first interval's
    # attitude; never extrapolates, never raises, never wraps around to
    # the last row.
    first_attitude = intervals[0].attitude.to_value(u.deg)
    far_before_s = intervals[0].start_time.to_value(u.s) - 1.0e6
    assert _attitude_at(far_before_s, start_times_s, attitudes_deg) == pytest.approx(first_attitude)
    assert _attitude_at(far_before_s, start_times_s, attitudes_deg) != pytest.approx(
        intervals[-1].attitude.to_value(u.deg))


def _draw_events(source, detector, history, earth, n_per_interval=60):
    """Draw `n_per_interval` photons from `source` at every interval of
    `history` (skipping any the Earth occults), each timestamped UNIFORMLY
    over its own interval's `[start_time, stop_time)` span -- matching how
    `InertialSimulator.run_events` actually timestamps photons
    (`inertial_simulator.py`: "uniform over the interval's full span"), not
    pinned to `mid_time`.

    An earlier version of this helper pinned every timestamp to
    `interval.mid_time`, which is always safely inside `[start_time,
    stop_time)` and so never exercises `_attitude_at`'s interval-boundary
    search at all -- exactly the PR 5 lesson recorded in
    `.claude/cosimita-progress.md`: "isolating one effect can delete the
    effect you meant to test." Drawing uniformly means a photon can land
    arbitrarily close to either boundary of its interval, which is where a
    wrong `searchsorted` side or a missing clamp actually shows up.

    Returns `(events, expected_sky_angle_deg)`, the second being the
    *drawn* interval's own attitude for each surviving photon, known
    directly from the `SpacecraftInterval` object used to draw it -- never
    from re-running any timestamp search."""

    events = []
    attitudes_deg = []

    for interval in history:
        start_s = interval.start_time.to_value(u.s)
        stop_s = interval.stop_time.to_value(u.s)

        for _ in range(n_per_interval):
            photon = source.random_photon(detector, pose=interval, earth=earth)
            if photon is None:
                continue  # occulted; never launched, never written
            time_s = np.random.uniform(start_s, stop_s)
            events.append((time_s * u.s, source, photon, _untriggered()))
            attitudes_deg.append(interval.attitude.to_value(u.deg))

    return events, np.array(attitudes_deg)


_FAR_EARTH = Earth(radius=6371.0 * u.km)
# Orbit radius huge relative to Earth's -> rho = arcsin(6371/2e6) = 0.18 deg,
# so far-field sources (PointSource, IsotropicSource, ExtendedSource) are
# essentially never occulted.
_FAR_ORBIT_RADIUS_KM = 2.0e6
# A realistic LEO radius for EarthAlbedoSource, whose spread comes from the
# Earth's actual apparent size (rho = arcsin(6371/7000) = 65.7 deg).
_LEO_ORBIT_RADIUS_KM = 7000.0


@pytest.mark.parametrize('make_source', [
    pytest.param(lambda: PointSource(sky_angle=40 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s),
                 id='PointSource(sky_angle)'),
    pytest.param(lambda: IsotropicSource(spectrum=SPECTRUM, flux=1 / u.cm / u.s),
                 id='IsotropicSource'),
    pytest.param(lambda: ExtendedSource(sky_angle=40 * u.deg, width=25 * u.deg,
                                        spectrum=SPECTRUM, flux=1 / u.cm / u.s),
                 id='ExtendedSource'),
])
def test_true_sky_angle_deg_matches_independent_formula_with_history(tmp_path, make_source):
    detector = _make_tracker()
    history = _varying_attitude_history(_FAR_EARTH, _FAR_ORBIT_RADIUS_KM)
    source = make_source()

    events, attitudes_deg = _draw_events(source, detector, history, _FAR_EARTH)
    assert len(events) > 300  # negligible occultation at this orbit radius

    path = tmp_path / 'events.csv'
    write_event_csv(path, events, spacecraft_history=history)
    _, table = read_event_csv(path)

    expected_deg = _independent_expected_sky_angle_deg(
        attitudes_deg, table['true_offaxis_angle_deg'].to_numpy())

    # THE REGRESSION THIS GUARDS: before the fix, ExtendedSource wrote the
    # constant `source.sky_angle` (40.0) into every row instead of each
    # photon's own drawn sky angle -- measured mean error ~20.9 deg, max
    # ~107.3 deg against the true per-photon value. Checking against a
    # constant, or only "not NaN", would pass that bug; comparing every row
    # against its own independently-computed truth does not.
    diff_deg = _wrap180(table['true_sky_angle_deg'].to_numpy() - expected_deg)
    assert np.max(np.abs(diff_deg)) < 1e-6

    # And, spelled out as the brief asks -- the SPREAD, not just presence:
    # a source with real per-photon variation must show it (a constant
    # fallback collapses this to ~0), while the fixed-sky_angle PointSource
    # legitimately has none.
    spread_deg = table['true_sky_angle_deg'].to_numpy().std()
    if isinstance(source, PointSource):
        assert spread_deg < 1e-6
    else:
        assert spread_deg > 5.0


def test_true_sky_angle_deg_matches_independent_formula_for_earth_albedo(tmp_path):
    earth = Earth(radius=6371.0 * u.km)
    detector = _make_tracker()
    history = _varying_attitude_history(earth, _LEO_ORBIT_RADIUS_KM)
    source = EarthAlbedoSource(emissivity=1 / u.cm / u.s, spectrum=SPECTRUM, earth=earth)

    # EarthAlbedoSource is not occultable, so every draw survives.
    events, attitudes_deg = _draw_events(source, detector, history, earth)
    assert len(events) == 6 * 60

    path = tmp_path / 'events.csv'
    write_event_csv(path, events, spacecraft_history=history)
    _, table = read_event_csv(path)

    expected_deg = _independent_expected_sky_angle_deg(
        attitudes_deg, table['true_offaxis_angle_deg'].to_numpy())

    diff_deg = _wrap180(table['true_sky_angle_deg'].to_numpy() - expected_deg)
    assert np.max(np.abs(diff_deg)) < 1e-6

    # Real per-photon spread, driven by the Earth's own apparent size
    # (rho = 65.7 deg at this orbit radius) -- not a constant fallback.
    assert table['true_sky_angle_deg'].to_numpy().std() > 5.0


@pytest.mark.parametrize('make_source', [
    pytest.param(lambda: PointSource(sky_angle=40 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s),
                 id='PointSource(sky_angle)'),
    pytest.param(lambda: IsotropicSource(spectrum=SPECTRUM, flux=1 / u.cm / u.s),
                 id='IsotropicSource'),
    pytest.param(lambda: ExtendedSource(sky_angle=40 * u.deg, width=25 * u.deg,
                                        spectrum=SPECTRUM, flux=1 / u.cm / u.s),
                 id='ExtendedSource'),
])
def test_true_sky_angle_deg_without_history(tmp_path, make_source):
    # Item 12: without spacecraft_history, true_sky_angle_deg is populated
    # ONLY for a fixed PointSource(sky_angle=...) -- whose constant is
    # every photon's exact truth even without a per-photon attitude lookup
    # -- and NaN for every other far-field source. In particular
    # ExtendedSource.sky_angle (the von Mises centre) must NOT be used as a
    # fallback -- this is the same regression as the history case, in the
    # no-history branch of `_true_sky_angle_deg`.
    detector = _make_tracker()
    history = _varying_attitude_history(_FAR_EARTH, _FAR_ORBIT_RADIUS_KM)
    source = make_source()

    events, _ = _draw_events(source, detector, history, _FAR_EARTH, n_per_interval=20)
    assert len(events) > 0

    path = tmp_path / 'events.csv'
    write_event_csv(path, events)  # no spacecraft_history
    _, table = read_event_csv(path)

    sky = table['true_sky_angle_deg'].to_numpy()

    if isinstance(source, PointSource):
        assert np.all(np.abs(_wrap180(sky - 40.0)) < 1e-9)
    else:
        assert np.all(np.isnan(sky))


def test_true_sky_angle_deg_nan_for_near_field_even_with_history(tmp_path):
    # Item 13, near-field half: a NearFieldSource has no sky position at
    # all, with or without spacecraft_history.
    detector = _make_tracker()
    history = _varying_attitude_history(_FAR_EARTH, _FAR_ORBIT_RADIUS_KM)
    source = NearPointSource(position=Cartesian2D(0 * u.cm, 0 * u.cm), spectrum=SPECTRUM,
                             rate=1 * u.Hz)

    events = [(interval.mid_time, source, source.random_photon(detector), _untriggered())
             for interval in history for _ in range(5)]

    path = tmp_path / 'events.csv'
    write_event_csv(path, events, spacecraft_history=history)
    _, table = read_event_csv(path)

    assert len(table) == 30
    assert np.all(np.isnan(table['true_sky_angle_deg'].to_numpy()))


def test_true_sky_angle_deg_nan_for_detector_frame_point_source_even_with_history(tmp_path):
    # Item 13, detector-frame half: a PointSource(offaxis_angle=...) stays
    # welded to the detector regardless of attitude, so it has no inertial
    # sky angle to recover even when a spacecraft_history is supplied.
    detector = _make_tracker()
    history = _varying_attitude_history(_FAR_EARTH, _FAR_ORBIT_RADIUS_KM)
    source = PointSource(offaxis_angle=40 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s)

    events = [(interval.mid_time, source, source.random_photon(detector), _untriggered())
             for interval in history for _ in range(5)]

    path = tmp_path / 'events.csv'
    write_event_csv(path, events, spacecraft_history=history)
    _, table = read_event_csv(path)

    assert len(table) == 30
    assert np.all(np.isnan(table['true_sky_angle_deg'].to_numpy()))


def test_true_sky_angle_deg_uses_the_documented_180_convention(tmp_path):
    # Every comparison above checks `true_sky_angle_deg` against an
    # independently computed value modulo 360 (via `_wrap180` on the
    # DIFFERENCE), which cannot tell the documented [-180, 180) convention
    # apart from a plain `angle % 360.0` -- 200.0 and -160.0 are congruent
    # mod 360, so a diff-based check passes either way. This test compares
    # the literal returned number instead.
    #
    # attitude = 0 deg (fixed), source at sky_angle = 200 deg: the RAW
    # difference A - Nu is -160 deg (worked out below from the project's
    # Nu = A - lambda convention), which the documented range must report
    # as -160.0, not its mod-360 equivalent 200.0.
    detector = _make_tracker()
    earth = _FAR_EARTH
    history = SpacecraftHistory(time=[0.0, 1000.0] * u.s,
                                orbit_radius=[_FAR_ORBIT_RADIUS_KM] * 2 * u.km,
                                orbit_angle=[0.0, 0.0] * u.deg,
                                attitude=[0.0, 0.0] * u.deg,
                                uptime=[1000.0, 0.0] * u.s,
                                earth=earth)
    interval = next(iter(history))

    source = PointSource(sky_angle=200 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s)
    photon = source.random_photon(detector, pose=interval, earth=earth)
    assert photon is not None  # not occulted at this orbit radius

    # Sanity check on the setup, independent of write_event_csv: Nu =
    # wrap(A - lambda) = wrap(0 - 200) = wrap(-200) = 160 deg, so the
    # photon flies along 270 - 160 = 110 deg.
    assert photon.direction.to_value(u.deg) == pytest.approx(110.0)

    events = [(interval.mid_time, source, photon, _untriggered())]

    path = tmp_path / 'events.csv'
    write_event_csv(path, events, spacecraft_history=history)
    _, table = read_event_csv(path)

    sky = table['true_sky_angle_deg'].iloc[0]
    assert sky == pytest.approx(-160.0)
    assert -180.0 <= sky < 180.0


# ===========================================================================
# Part C -- a real InertialSimulator.run_events() stream, and the two
# write_event_csv defects fix commit 546d840 addresses
# ===========================================================================

def test_write_event_csv_accepts_a_real_run_events_stream(tmp_path):
    # The module docstring's own "typical use" -- piping
    # InertialSimulator.run_events() straight into write_event_csv -- was
    # never exercised: every other test in this file hand-builds its event
    # tuples. This closes that gap together with the uniform-timestamp
    # fix above, since run_events() itself draws timestamps uniformly over
    # each interval's span (inertial_simulator.py), so this is the same
    # attitude-lookup stress test as `_draw_events`, but through the real
    # simulator end to end.
    np.random.seed(20260907)

    detector = _make_tracker()
    history = _varying_attitude_history(_FAR_EARTH, _FAR_ORBIT_RADIUS_KM)

    # mu = flux * throwing_plane_size * total_livetime (Sections 5.2, 6);
    # solved here for a target of ~300 unocculted photons over the whole
    # run, so the run finishes quickly but still leaves enough rows to make
    # the per-row check below meaningful.
    target_mu = 300.0
    flux = (target_mu / (detector.throwing_plane_size * history.total_livetime)).to(1 / u.cm / u.s)
    source = ExtendedSource(sky_angle=40 * u.deg, width=25 * u.deg, spectrum=SPECTRUM, flux=flux)

    simulator = InertialSimulator(detector=detector, sources=[source],
                                  reconstructor=SimpleTraditionalReconstructor(),
                                  spacecraft_history=history, earth=_FAR_EARTH)

    path = tmp_path / 'run.csv'
    write_event_csv(path, simulator.run_events(progress=False),
                    source_names={source: 'crab'}, ori_file='test.ori',
                    total_livetime=history.total_livetime, spacecraft_history=history)

    metadata, table = read_event_csv(path)

    assert len(table) > 100  # sigma(300) = 17.3; a handful of rows would be a real failure
    assert metadata['nsim'] == {'crab': len(table)}  # nothing occulted at this orbit radius

    # Independent per-photon attitude, from a plain linear scan over the
    # history's own intervals (not `_attitude_at`'s vectorized
    # searchsorted) -- a different algorithm, so a bug shared between the
    # two would have to be a genuinely shared one.
    intervals = list(history)

    def owning_interval(time_s):
        for interval in intervals:
            if interval.start_time.to_value(u.s) <= time_s < interval.stop_time.to_value(u.s):
                return interval
        return intervals[0] if time_s < intervals[0].start_time.to_value(u.s) else intervals[-1]

    expected_attitude_deg = np.array([owning_interval(t).attitude.to_value(u.deg)
                                      for t in table['time_s'].to_numpy()])

    expected_sky_deg = _independent_expected_sky_angle_deg(
        expected_attitude_deg, table['true_offaxis_angle_deg'].to_numpy())

    diff_deg = _wrap180(table['true_sky_angle_deg'].to_numpy() - expected_sky_deg)
    assert np.max(np.abs(diff_deg)) < 1e-6


def test_source_label_with_hash_raises_at_write_time(tmp_path):
    # Regression for fix commit 546d840: `read_event_csv` parses with
    # `comment = '#'`, which truncates an unquoted field at the first '#'
    # and silently NaNs every column after it in that row -- while the
    # '#'-bearing name survives intact in the header's own `nsim`, leaving
    # the file inconsistent with itself and no error anywhere. Now refused
    # at write time, naming the source.
    source = PointSource(sky_angle=0 * u.deg, spectrum=SPECTRUM, flux=1 / u.cm / u.s)
    events = [(0.0 * u.s, source, _photon(0, 0, 0, 1, 1), _untriggered())]

    path = tmp_path / 'events.csv'
    with pytest.raises(ValueError):
        write_event_csv(path, events, source_names={source: 'crab#1'})


def test_path_like_ori_file_round_trips(tmp_path):
    # Regression for fix commit 546d840: a `pathlib.Path` ori_file used to
    # serialize with `yaml.dump`'s Python-specific
    # `!!python/object/apply:pathlib.PosixPath` tag, which
    # `yaml.safe_load` (what `read_event_csv` parses the header with)
    # refuses to construct -- the file could never be read back at all.
    # `ori_file` is now str()-coerced before it reaches the dumper.
    ori_path = tmp_path / 'iss.ori'  # a genuine pathlib.Path, not a str

    path = tmp_path / 'events.csv'
    write_event_csv(path, [], ori_file=ori_path)

    metadata, table = read_event_csv(path)  # must not raise

    assert metadata['ori_file'] == str(ori_path)
    assert isinstance(metadata['ori_file'], str)
