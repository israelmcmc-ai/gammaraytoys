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
                total_livetime = spacecraft_history.total_livetime)

metadata, table = read_event_csv('run.csv')
```
"""

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import Angle
import yaml

from .._version import __version__


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
        return source_names[source]

    return type(source).__name__


def _true_sky_angle_deg(source):
    """
    This event's ground-truth inertial sky angle, in degrees, or `nan`.

    Taken from `source.sky_angle` when the source has one: a fixed value
    for `PointSource(sky_angle = ...)`, and the distribution's centre
    (not the individual photon's von Mises jitter, which is never
    retained after the draw) for `ExtendedSource`. `nan` for every source
    with no single meaningful sky angle: a `NearFieldSource` (it has no
    sky position at all -- Section 6's "empty for detector-frame-native
    near sources"), a detector-frame `PointSource(offaxis_angle = ...)`
    (its `sky_angle` attribute is `None`), `IsotropicSource` (uniform over
    the whole sky, no single direction) and `EarthAlbedoSource` (each
    photon comes from a different point on the Earth's surface, with no
    `sky_angle` attribute at all).

    Parameters
    ----------
    source : `Source`

    Returns
    -------
    float
        Degrees, or `nan`.
    """

    sky_angle = getattr(source, 'sky_angle', None)

    if sky_angle is None:
        return np.nan

    return sky_angle.to_value(u.deg)


def _event_row(event_id, time, source, sim_event, reco_event, source_names):
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

    Returns
    -------
    dict
        One row, keyed by every name in `_COLUMNS`.
    """

    # Same off-axis-angle convention `SimulatorBase._run_binned` uses to
    # recover Nu from a thrown photon's detector-frame direction.
    offaxis_angle = Angle(270 * u.deg - sim_event.direction).wrap_at(180 * u.deg)

    row = {
        'event_id': event_id,
        'time_s': time.to_value(u.s),
        'source': _source_label(source, source_names),
        'true_x_cm': sim_event.position.x.to_value(u.cm),
        'true_y_cm': sim_event.position.y.to_value(u.cm),
        'true_direction_deg': sim_event.direction.to_value(u.deg),
        'true_sky_angle_deg': _true_sky_angle_deg(source),
        'true_offaxis_angle_deg': offaxis_angle.to_value(u.deg),
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
                    source_names = None, ori_file = None, total_livetime = None):
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
    `true_sky_angle_deg` is empty (`nan`) whenever the emitting source has no
    single meaningful inertial sky angle -- always true for a near-field
    source, and also for `IsotropicSource` and `EarthAlbedoSource` (see
    `_true_sky_angle_deg`). The `reco_*` columns are empty (`nan`) for any
    event with `triggered = False`.

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

    Returns
    -------
    dict
        The metadata actually written (see the header block above):
        `gammaraytoys_version`, `ori_file`, `total_livetime_s`, `nsim` (a
        dict of event counts per source label, over every event `events`
        produced, independent of `triggered_only`) and `triggered_only`.
    """

    rows = []
    nsim = {}

    for event_id, (time, source, sim_event, reco_event) in enumerate(events):

        label = _source_label(source, source_names)
        nsim[label] = nsim.get(label, 0) + 1

        if triggered_only and not reco_event.triggered:
            continue

        rows.append(_event_row(event_id, time, source, sim_event, reco_event,
                               source_names))

    table = pd.DataFrame(rows, columns = _COLUMNS)

    metadata = {
        'gammaraytoys_version': __version__,
        'ori_file': ori_file,
        # `float(...)`, not the bare `Quantity.to_value` result: that can be
        # a numpy scalar, which `yaml.dump` (default Dumper) serializes with
        # a Python-specific `!!python/object/apply:...` tag that
        # `yaml.safe_load` -- used by `read_event_csv` -- refuses to
        # construct. A plain Python float dumps as an ordinary YAML scalar.
        'total_livetime_s': (float(total_livetime.to_value(u.s))
                             if total_livetime is not None else None),
        'nsim': nsim,
        'triggered_only': bool(triggered_only),
    }

    header = yaml.dump(metadata, sort_keys = False, default_flow_style = None)
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
