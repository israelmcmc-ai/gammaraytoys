"""
YAML configuration for the simulators (`docs/dev/inertial_sim_plan.md`,
Section 7): one file that names a detector, an Earth, a spacecraft history
and a list of sources, and builds the whole run from it.

The entry points are `Simulator.from_config` and
`InertialSimulator.from_config`. This module is where the schema below is
written down; the machinery that reads it lives on the classes themselves,
one kind per class:

    detector             `ToyTracker2D.from_config`   / `.to_config()`
    earth                `Earth.from_config`          / `.to_config()`
    reconstructor        `Reconstructor.from_config`  / `.to_config()`
    spacecraft_history   `SpacecraftHistory.from_config`
    observation_strategy `ObservationStrategy.from_config` / `.to_config()`
    sources[i]           `Source.from_config`   / `.to_config(name = ...)`
      .spectrum          `Spectrum.from_config` / `.to_config()`
      .scaling           `SourceScaling.from_config` / `.to_config()`
      .position          `Cartesian2D.from_config` / `.to_config()`

so that a single piece of a configuration -- one spectrum, one source, one
scaling -- can be built or written on its own, and so that the class a
reader is already looking at is the one that documents its own block.

What is left here is `load_config`, which turns a path or a mapping into a
plain configuration mapping, and the schema itself.

Python API only: there is no console script and no `python -m`.

Two rules run through the whole schema.

**Every unit-bearing value is a string astropy parses**, scalar
(`"16 cm"`, `"1e-3 1/(cm s)"`, `"45 deg"`) or array (`"[30, 0, 1] cm"`).
Note that YAML needs the array form **quoted**: an unquoted
`layer_positions: [30, 0, 1] cm` is not valid YAML at all (the plan's own
Section 7 sketch has this bug), and an unquoted `[30, 0, 1]` is a YAML list
with no unit, which this loader rejects. A value whose unit is missing,
misspelled or of the wrong physical type raises naming **the key it came
from** -- astropy's own message says only what it could not parse, never
where in the file it was written.

**Unknown keys are an error, not a warning**, at every level: top level,
detector, earth, source, spectrum, scaling, observation strategy,
spacecraft history. A silently ignored typo in a configuration file is a
debugging nightmare. Everything is validated by hand -- there is
deliberately no schema-validation dependency.

Schema
------

```yaml
detector:                       # required
  type: ToyTracker2D
  material: Ge
  layer_length: 16 cm
  layer_positions: "[30, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9] cm"
  layer_thickness: 5 mm         # scalar, or one entry per layer
  energy_resolution: 0.01       # plain number, or one per layer
  energy_threshold: 20 keV      # scalar, or one per layer

earth:                          # optional; default is astropy's R_earth
  radius: 6371 km

reconstructor:                  # optional; this is the default
  type: SimpleTraditionalReconstructor

doppler_broadening: true        # optional; default true

random_seed: 42                 # optional; seeds numpy's global RNG

spacecraft_history: iss.ori     # InertialSimulator only; required there.
                                # A path to a .ori file, or the orbit block
                                # below in its place.

sources:                        # required, at least one
  - name: crab                  # optional; labels this source's events
    type: PointSource
    sky_angle: 45 deg           # ... or offaxis_angle, never both
    flux: 1e-3 1/(cm s)
    spectrum: {type: PowerLaw, index: -2, min_energy: 0.2 MeV, max_energy: 10 MeV}
    scaling: {type: Sinusoidal, mean: 1.0, amplitude: 0.5, period: 5400 s}
```

In place of the `.ori` path, an orbit to generate:

```yaml
spacecraft_history:
  type: elliptical_orbit
  semi_major_axis: 6771 km
  eccentricity: 0.0             # optional, default 0.0
  duration: 6000 s              # optional, default one orbital period
  time_step: 100 s              # optional, default 1 s
  argument_of_periapsis: 0 deg  # optional, default 0 deg
  initial_time: 0 s             # optional, default 0 s
  livetime_fraction: 1.0        # optional, default 1.0
  observation_strategy:         # optional, default ZenithPointing
    type: ZenithPointing
```

The type names accepted for each kind are listed at the bottom of the file
that kind lives in: `_SPECTRUM_TYPES` in `spectrum.py`, `_SCALING_TYPES` in
`scaling.py`, `_SOURCE_TYPES` in `source.py`, `_STRATEGY_TYPES` in
`observation_strategy.py`. Spectra and scalings accept both the short name
the plan's sketch uses (`PowerLaw`, `Sinusoidal`) and the full class name
(`PowerLawSpectrum`, `SinusoidalScaling`); the short form is what
`to_config` writes back.

Round trips
-----------

Every kind has a `from_config` classmethod and a `to_config` method, and
`Simulator.to_config()` / `InertialSimulator.to_config()` write a whole
run back out. `to_config` emits a **canonical** configuration: it omits
keys left at their default, writes quantities in the unit they are held in,
and resolves the aliases above. So `X.from_config(config).to_config()`
equals `config` for a configuration already written in canonical form, and
is otherwise the canonical spelling of the same thing -- feeding it back
through `X.from_config` gives an equal object, and writing that out again
gives an identical configuration.

Reading is done on the base class of each kind, which picks the subclass
the block's `type` names: `Source.from_config` hands back a `PointSource`
or an `IsotropicSource` as the block says. Calling it on a concrete class
instead pins the answer down -- `PointSource.from_config` builds a
`PointSource` and refuses a block whose `type` says otherwise, rather than
quietly handing back the other class.

Sibling files
-------------

A configuration can name files beside it -- an `.ori` spacecraft history,
a `TabulatedScaling`'s CSV. **A relative path is taken relative to the
directory holding the configuration file**, so that a configuration in
`runs/` naming `iss.ori` finds `runs/iss.ori` and can be loaded from
anywhere. Absolute paths are used as written.

The one exception is a configuration handed in as an already-parsed
mapping: it came from no file, so there is no directory to resolve
against, and a relative path there falls back to the process' working
directory, as it always has.

`to_config` writes a path back **exactly as it was given**. Rewriting
`iss.ori` as `/home/someone/runs/iss.ori` on the way out would turn a
portable configuration into a machine-specific one.

Two things cannot be recovered from the objects themselves and so ride on
the configuration a simulator was built from, recorded by `from_config`:
the spacecraft history's provenance (an `.ori` path, or the orbital
elements -- a generated `SpacecraftHistory` keeps only the sampled rows,
not the Kepler elements that produced them) and the source names.
`to_config` on a simulator that was not built by `from_config` says so
rather than guessing.

Why there is no expression syntax
---------------------------------

An earlier draft of this schema let a `scaling` carry a free-form
expression in `t`, evaluated when the file was loaded. That is a
configuration file handing the loader a piece of Python to run, which is
an injection hazard however carefully it is fenced in -- and the fence was
several hundred lines of this module. The schema instead names **fixed,
parameterised scalings** (`_SCALING_TYPES` in `scaling.py`), every
argument of which is a number or a quantity: there is nothing in a
configuration file left to evaluate, and so nothing to sanitise. A shape
none of them can make belongs in a `Tabulated` scaling's CSV, or in
Python.
"""

from collections.abc import Mapping
from pathlib import Path

import yaml

from ..config_utils import _as_mapping


__all__ = ['load_config']


def load_config(config):
    """
    Turn whatever the caller passed into a plain configuration mapping.

    Parameters
    ----------
    config : str, path-like or mapping
        A path to a YAML file, or an already-parsed mapping. Which of the
        two it is also decides where the relative sibling paths inside it
        point: see `_config_base_dir` and the module docstring.

    Returns
    -------
    dict
        A shallow copy of the mapping (so the caller's own dictionary is
        never modified), or the parsed contents of the file.

    Raises
    ------
    TypeError
        If `config` is neither a path nor a mapping.
    ValueError
        If the file is not valid YAML, or does not hold a mapping at its
        top level (an empty file included).
    FileNotFoundError
        If the path does not exist.
    """

    if isinstance(config, Mapping):
        return _as_mapping(config, "config")

    if not isinstance(config, (str, Path)):
        raise TypeError(
            "config must be a path to a YAML file or an already-parsed "
            f"mapping, got {type(config).__name__}.")

    path = Path(config)

    with open(path, 'r') as file:
        try:
            parsed = yaml.safe_load(file)
        except yaml.YAMLError as err:
            raise ValueError(f"{path}: is not valid YAML ({err}).") from err

    if parsed is None:
        raise ValueError(f"{path}: is empty.")

    return _as_mapping(parsed, str(path))


def _config_base_dir(config):
    """
    The directory a configuration's relative sibling paths resolve against.

    Parameters
    ----------
    config : str, path-like or mapping
        Whatever was handed to `load_config`.

    Returns
    -------
    `pathlib.Path` or None
        The directory holding the configuration file, or `None` if
        `config` is an already-parsed mapping -- which came from no file
        and so has no directory of its own.
    """

    if isinstance(config, (str, Path)):
        return Path(config).parent

    return None
