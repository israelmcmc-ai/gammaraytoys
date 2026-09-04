"""
Event CSV I/O (`docs/dev/inertial_sim_plan.md`, Section 6, PR 6).

An "event CSV" is a flat table, one row per photon actually launched at the
detector, with a `#`-prefixed YAML metadata block above the usual CSV header
and rows -- so a single file is self-describing: which `.ori` it came from,
how much livetime it covers, how many photons were simulated per source, and
whether it holds every launched photon or only the triggers.

This is a separate, and much flatter, format from `EventList`
(`gammaraytoys.sims.event`), which dumps the full interaction tree of a
handful of events as YAML for close inspection. The two serve different jobs
and neither replaces the other: this one is for a whole run's worth of
events, in a shape pandas (or any spreadsheet) can read directly; `EventList`
is for looking closely at exactly what happened inside one or two events.

Typical use, with `InertialSimulator.run_events()`:

```python
write_event_csv('run.csv', simulator.run_events(),
                source_names = {crab: 'crab', albedo: 'albedo'},
                ori_file = 'iss.ori',
                total_livetime = spacecraft_history.total_livetime,
                spacecraft_history = spacecraft_history)

metadata, table = read_event_csv('run.csv')
```

Passing `spacecraft_history` (the same history the run used) lets
`true_sky_angle_deg` be recovered exactly, per photon, for every far-field
source -- see `write_event_csv` for why it is not populated by guesswork
without one.
"""

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import Angle
import yaml

from .._version import __version__
from .source import PointSource, NearFieldSource


# Column order of the event CSV, exactly as specified by the PR 6 contract.
# Units are baked into the names, matching the `.ori` file's own convention.
_COLUMNS = ['event_id', 'time_s', 'source',
           'true_x_cm', 'true_y_cm', 'true_direction_deg',
           'true_sky_angle_deg', 'true_offaxis_angle_deg',
           'true_energy_MeV', 'true_chirality',
           'triggered',
           'reco_energy_MeV', 'reco_phi_deg', 'reco_psi_deg']


def _source_label(source, source_names):
    """
    The string this event's `source` column should hold.

    Parameters
    ----------
    source : `Source`
        The source that emitted this event.
    source_names : dict or None
        Maps `Source` objects to the name they should be labelled with in
        the file (e.g. `{crab: 'crab', albedo: 'albedo'}`). `None`, or a
        source missing from it, falls back to the source's class name (e.g.
        `'PointSource'`).

    Returns
    -------
    str
    """

    if source_names is not None and source in source_names:
        label = str(source_names[source])
    else:
        label = type(source).__name__

    # `read_event_csv` parses with `comment = '#'`, which truncates an
    # unquoted field at the first '#' and silently NaNs every column after
    # it -- while the '#'-bearing name survives intact in the header's
    # `nsim`, leaving the file inconsistent with itself. Refuse to write
    # such a file rather than write one that cannot be read back.
    if '#' in label:
        raise ValueError(
            f"Source label {label!r} contains '#', which cannot appear in an "
            "event file: the reader treats it as the start of a comment and "
            "would silently drop the rest of the row. Rename the source in "
            "`source_names`.")

    return label


def _wrap180(angle_deg):
    """Wrap a plain-float angle, in degrees, to `[-180, 180)`."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _attitude_lookup_table(spacecraft_history):
    """
    Build a per-timestamp spacecraft-attitude lookup from a
    `SpacecraftHistory`, for `_attitude_at`.

    Parameters
    ----------
    spacecraft_history : `SpacecraftHistory`
        Iterated once, in row order. `SpacecraftHistory.__iter__` already
        excludes the terminator row, whose pose is never meaningful.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        `(start_times_s, attitudes_deg)`: interval start times, in seconds,
        and the attitude `A` holding over each, in degrees. Both float
        arrays of the same length; `start_times_s` is strictly increasing
        because a `SpacecraftHistory`'s row timestamps are.
    """

    start_times_s = []
    attitudes_deg = []

    for interval in spacecraft_history:
        start_times_s.append(interval.start_time.to_value(u.s))
        attitudes_deg.append(interval.attitude.to_value(u.deg))

    return np.asarray(start_times_s, dtype = float), np.asarray(attitudes_deg, dtype = float)


def _attitude_at(time_s, start_times_s, attitudes_deg):
    """
    The spacecraft attitude in effect at `time_s`, from a table built by
    `_attitude_lookup_table`.

    The same interval-lookup-by-timestamp idiom
    `docs/examples/cosimita/02-inertial_simulation_and_occultation.ipynb`
    already uses: interval `i` owns `[start_times_s[i], start_times_s[i+1])`,
    so `searchsorted(..., side = 'right') - 1` finds the row whose span
    contains `time_s`; a `time_s` outside the table's own range is clamped
    to the nearest end rather than extrapolated.

    Parameters
    ----------
    time_s : float
        Event timestamp, plain seconds.
    start_times_s, attitudes_deg : numpy.ndarray of float
        From `_attitude_lookup_table`.

    Returns
    -------
    float
        Attitude `A`, in degrees.
    """

    idx = np.searchsorted(start_times_s, time_s, side = 'right') - 1
    idx = int(np.clip(idx, 0, start_times_s.size - 1))

    return float(attitudes_deg[idx])


def _true_sky_angle_deg(source, offaxis_angle_deg, attitude_deg):
    """
    This event's ground-truth inertial sky angle `lambda`, in degrees, or
    `nan` if it cannot be known.

    This is a `true_*` column -- ground truth for *this* photon, not a
    property of the source in general -- so it must never be a value that
    is merely plausible; a wrong-looking constant is worse than an honest
    `nan`. Concretely:

    - `nan` for a `NearFieldSource`: it has no sky position at all
      (Section 6: "empty for detector-frame-native near sources").
    - `nan` for a `PointSource(offaxis_angle = ...)` (detector-frame mode,
      `sky_angle` is `None`): it stays welded to the detector regardless of
      attitude, so it has no inertial sky angle either.
    - Otherwise, if `attitude_deg` is given (i.e. `write_event_csv` was
      given a `spacecraft_history`): `lambda = A - Nu`, wrapped to
      `[-180, 180)` -- the inverse of the project's standing
      `Nu = A - lambda` convention (`gammaraytoys.coordinates.transform
      .sky_angle_to_offaxis`) -- which is *exact* for this one photon,
      whatever the emitting source's own `sky_angle` attribute says (or
      whether it even has one): it needs only the attitude at the photon's
      own timestamp and its own `true_offaxis_angle_deg`, both already
      known per photon.
    - Otherwise (no `spacecraft_history`, so no per-photon attitude to look
      up), a `PointSource(sky_angle = ...)` falls back to that constant
      `sky_angle`: this is the one case where the constant genuinely *is*
      every photon's exact truth, since `Nu = A - sky_angle` for every
      single draw. Every other far-field source (`IsotropicSource`,
      `ExtendedSource`, `EarthAlbedoSource`) draws a different sky angle
      per photon, so without a `spacecraft_history` to look the attitude
      up in there is no way to recover it, and this returns `nan` for them
      -- notably, `ExtendedSource.sky_angle` is the von Mises centre, not
      any individual photon's draw, and can be wrong by many multiples of
      `width` for a given photon (measured: mean error ~21 deg, max
      ~107 deg, for `width = 25 deg`).

    Parameters
    ----------
    source : `Source`
        The source that emitted this photon.
    offaxis_angle_deg : float
        This photon's `true_offaxis_angle_deg`, already computed (plain
        float, degrees).
    attitude_deg : float or None
        The spacecraft attitude at this photon's timestamp (plain float,
        degrees), from `_attitude_at`, or `None` if `write_event_csv` was
        not given a `spacecraft_history`.

    Returns
    -------
    float
        Degrees, wrapped to `[-180, 180)`, or `nan`.
    """

    if isinstance(source, NearFieldSource):
        return np.nan

    if isinstance(source, PointSource) and source.sky_angle is None:
        return np.nan

    if attitude_deg is not None:
        return _wrap180(attitude_deg - offaxis_angle_deg)

    if isinstance(source, PointSource):
        return source.sky_angle.to_value(u.deg)

    return np.nan


def _event_row(event_id, time, source, sim_event, reco_event, source_names,
               attitude_lookup):
    """
    Build one row of the event table, as a dict keyed by `_COLUMNS`.

    Parameters
    ----------
    event_id : int
        This event's sequential id (see `write_event_csv`).
    time : `astropy.units.Quantity`
        The event's timestamp (time units), as yielded by
        `InertialSimulator.run_events`.
    source : `Source`
        The source that emitted this event.
    sim_event : `Photon`
        The simulated (detector-frame) event -- `run_events`'s `sim_event`.
    reco_event : `RecoEvent`
        The event's reconstruction -- `run_events`'s `reco_event`.
    source_names : dict or None
        See `_source_label`.
    attitude_lookup : (numpy.ndarray, numpy.ndarray) or None
        `(start_times_s, attitudes_deg)` from `_attitude_lookup_table`, or
        `None` if `write_event_csv` was not given a `spacecraft_history`
        (see `_true_sky_angle_deg`).

    Returns
    -------
    dict
        One row, keyed by every name in `_COLUMNS`.
    """

    time_s = time.to_value(u.s)

    # Same off-axis-angle convention `SimulatorBase._run_binned` uses to
    # recover Nu from a thrown photon's detector-frame direction.
    offaxis_angle_deg = Angle(270 * u.deg - sim_event.direction).wrap_at(180 * u.deg).to_value(u.deg)

    if attitude_lookup is None:
        attitude_deg = None
    else:
        attitude_deg = _attitude_at(time_s, *attitude_lookup)

    row = {
        'event_id': event_id,
        'time_s': time_s,
        'source': _source_label(source, source_names),
        'true_x_cm': sim_event.position.x.to_value(u.cm),
        'true_y_cm': sim_event.position.y.to_value(u.cm),
        'true_direction_deg': sim_event.direction.to_value(u.deg),
        'true_sky_angle_deg': _true_sky_angle_deg(source, offaxis_angle_deg, attitude_deg),
        'true_offaxis_angle_deg': offaxis_angle_deg,
        'true_energy_MeV': sim_event.energy.to_value(u.MeV),
        'true_chirality': int(sim_event.chirality),
        'triggered': bool(reco_event.triggered),
        'reco_energy_MeV': (reco_event.energy.to_value(u.MeV)
                            if reco_event.triggered else np.nan),
        'reco_phi_deg': (reco_event.phi.to_value(u.deg)
                         if reco_event.triggered else np.nan),
        'reco_psi_deg': (reco_event.psi.to_value(u.deg)
                         if reco_event.triggered else np.nan),
    }

    return row


def write_event_csv(filename, events, *, triggered_only = False,
                    source_names = None, ori_file = None, total_livetime = None,
                    spacecraft_history = None):
    """
    Write a stream of simulated events to an event CSV file (Section 6).

    The file is a `#`-prefixed YAML metadata block followed by a plain CSV
    table, one row per event, with columns

    ```
    event_id, time_s, source, true_x_cm, true_y_cm, true_direction_deg,
    true_sky_angle_deg, true_offaxis_angle_deg, true_energy_MeV,
    true_chirality, triggered, reco_energy_MeV, reco_phi_deg, reco_psi_deg
    ```

    Rows cover every photon `events` hands in -- typically every photon
    `InertialSimulator.run_events()` actually launched at the detector, since
    an occulted photon is never yielded by that generator in the first place
    (it was never launched, and the run's livetime already accounts for it).
    The `reco_*` columns are empty (`nan`) for any event with
    `triggered = False`.

    `true_sky_angle_deg` is a `true_*` column, i.e. it must be this photon's
    actual ground truth, or `nan` -- never a merely plausible-looking number.
    Recovering it exactly needs the spacecraft's attitude at the photon's own
    timestamp (`lambda = A - Nu`, the inverse of the project's
    `Nu = A - lambda` convention), which is why this needs `spacecraft_history`
    (below):

    - **with `spacecraft_history`**: populated for every far-field source
      aimed on the inertial sky (`PointSource(sky_angle = ...)`,
      `IsotropicSource`, `ExtendedSource`, `EarthAlbedoSource`), exactly, per
      photon. Still `nan` for a near-field source (no sky position at all)
      and for a detector-frame `PointSource(offaxis_angle = ...)` (no
      inertial sky angle at all -- it ignores the spacecraft's attitude
      entirely).
    - **without it** (the default): populated *only* for
      `PointSource(sky_angle = ...)`, whose constant `sky_angle` genuinely is
      every one of its photons' exact truth. `nan` for every other far-field
      source -- in particular, `ExtendedSource.sky_angle` (the von Mises
      distribution's centre) is *not* used as a fallback here, because it is
      not this photon's truth: it can differ from an individual photon's
      actual sky angle by many multiples of `width` (measured: mean error
      ~21 deg, max ~107 deg, for a `width = 25 deg` source), and writing it
      into a `true_*` column would be worse than leaving the column empty.

    See `_true_sky_angle_deg` for the exact per-source rule.

    `event_id` numbers every event `events` produces, in order, starting at
    0 -- *before* the `triggered_only` filter below is applied, so an id
    still identifies the same underlying photon whether or not the file was
    written with `triggered_only = True`; a file with `triggered_only = True`
    therefore has gaps in `event_id`, not a dense `0..n-1` range.

    Parameters
    ----------
    filename : str or path-like
        Destination path. Opened for writing; raises whatever `open()` would
        (e.g. if the containing directory does not exist).
    events : iterable of (`astropy.units.Quantity`, `Source`, `Photon`, `RecoEvent`)
        The events to write, in the shape `InertialSimulator.run_events()`
        yields: `(time, source, sim_event, reco_event)`.
    triggered_only : bool
        If `True`, write only the events with `reco_event.triggered`; every
        other event is still counted for `event_id` and `nsim` (see above)
        but not written as a row. Default `False` (write every event).
    source_names : dict or None
        Maps each `Source` object appearing in `events` to the name its rows
        should carry in the `source` column (e.g.
        `{crab_source: 'crab', albedo_source: 'albedo'}`, matching the
        `nsim` metadata example in the plan). A source missing from this
        mapping (or `source_names = None`, the default) falls back to its
        class name (e.g. `'PointSource'`).
    ori_file : str or None
        Name or path of the `.ori` file the run was simulated over, recorded
        as metadata for reference. `None` (the default) leaves it unset in
        the header.
    total_livetime : `astropy.units.Quantity` or None
        Total livetime the run covers (time units), recorded as metadata
        (`total_livetime_s`). `None` (the default) leaves it unset.
    spacecraft_history : `SpacecraftHistory` or None
        The same history the run was simulated over (or any history sharing
        its row timestamps and attitudes), used to look up the spacecraft's
        attitude at each photon's own timestamp and so recover
        `true_sky_angle_deg` exactly for every far-field source. `None` (the
        default) leaves that column populated only for
        `PointSource(sky_angle = ...)` (see above); it is never required --
        only ever an improvement to that one column -- so an existing call
        without it keeps working exactly as before.

    Returns
    -------
    dict
        The metadata actually written (see the header block above):
        `gammaraytoys_version`, `ori_file`, `total_livetime_s`, `nsim` (a
        dict of event counts per source label, over every event `events`
        produced, independent of `triggered_only`) and `triggered_only`.
    """

    attitude_lookup = (None if spacecraft_history is None
                       else _attitude_lookup_table(spacecraft_history))

    rows = []
    nsim = {}

    for event_id, (time, source, sim_event, reco_event) in enumerate(events):

        label = _source_label(source, source_names)
        nsim[label] = nsim.get(label, 0) + 1

        if triggered_only and not reco_event.triggered:
            continue

        rows.append(_event_row(event_id, time, source, sim_event, reco_event,
                               source_names, attitude_lookup))

    table = pd.DataFrame(rows, columns = _COLUMNS)

    metadata = {
        'gammaraytoys_version': __version__,
        # `str(...)` so a path-like `ori_file` -- natural to pass, since
        # `filename` itself is documented as path-like -- does not reach the
        # dumper as a `Path`. See the `safe_dump` note below.
        'ori_file': str(ori_file) if ori_file is not None else None,
        # `float(...)`, not the bare `Quantity.to_value` result: that can be
        # a numpy scalar, which serializes with a Python-specific
        # `!!python/object/apply:...` tag that `yaml.safe_load` -- used by
        # `read_event_csv` -- refuses to construct. A plain Python float
        # dumps as an ordinary YAML scalar.
        'total_livetime_s': (float(total_livetime.to_value(u.s))
                             if total_livetime is not None else None),
        'nsim': nsim,
        'triggered_only': bool(triggered_only),
    }

    # `safe_dump`, not `dump`, so the writer and the reader agree on what is
    # representable: anything `safe_load` could not reconstruct raises here,
    # at write time and naming the type, instead of producing a file that
    # only fails when somebody later tries to read it.
    header = yaml.safe_dump(metadata, sort_keys = False, default_flow_style = None)
    header = '\n'.join(f"# {line}" for line in header.splitlines())

    with open(filename, 'w') as f:
        f.write(header + '\n')
        table.to_csv(f, index = False)

    return metadata


def _read_metadata(filename):
    """
    Read the leading `#`-prefixed YAML metadata block of an event CSV.

    `pandas.read_csv(..., comment = '#')` (used by `read_event_csv` for the
    table itself) simply discards `#` lines, so the metadata has to be
    recovered in a separate pass over the file's leading lines, before the
    plain CSV header.

    Parameters
    ----------
    filename : str or path-like
        Path to the event CSV file.

    Returns
    -------
    dict
        The metadata block, parsed as YAML. `{}` if the file has no leading
        `#` lines at all.
    """

    lines = []

    with open(filename) as f:
        for line in f:
            if not line.startswith('#'):
                break
            # Strip only the comment marker itself, not the space after it
            # that `write_event_csv` always writes -- consistent leading
            # whitespace across every line is fine for a YAML block.
            lines.append(line[1:])

    if not lines:
        return {}

    metadata = yaml.safe_load(''.join(lines))

    return metadata if metadata is not None else {}


def read_event_csv(filename):
    """
    Read an event CSV file written by `write_event_csv` (Section 6).

    Parameters
    ----------
    filename : str or path-like
        Path to the event CSV file.

    Returns
    -------
    (dict, `pandas.DataFrame`)
        `(metadata, table)`: the parsed `#`-prefixed metadata block (see
        `write_event_csv`'s return value for its keys), and the event table.
        A file with no event rows still reads back as an empty (zero-row)
        `DataFrame` with the full set of columns -- the header row alone is
        enough for pandas to know them. `nan` cells (the empty
        `true_sky_angle_deg` or `reco_*` fields) round-trip as `nan` through
        plain CSV, with no special handling needed.
    """

    metadata = _read_metadata(filename)

    table = pd.read_csv(filename, comment = '#', float_precision = 'round_trip')

    return metadata, table
