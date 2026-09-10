"""
YAML configuration for the simulators (`docs/dev/inertial_sim_plan.md`,
Section 7): one file that names a detector, an Earth, a spacecraft history
and a list of sources, and builds the whole run from it.

The entry points are `Simulator.from_config` and
`InertialSimulator.from_config`; everything in this module is the machinery
behind them, exposed so that a single piece of a configuration -- one
spectrum, one source, one scaling -- can be built or written on its own.

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

The type names accepted for each kind are listed in `_SPECTRUM_TYPES`,
`_SCALING_TYPES`, `_SOURCE_TYPES` and `_STRATEGY_TYPES` below. Spectra and
scalings accept both the short name the plan's sketch uses (`PowerLaw`,
`Sinusoidal`) and the full class name (`PowerLawSpectrum`,
`SinusoidalScaling`); the short form is what `*_to_config` writes back.

Round trips
-----------

Every kind has a `*_from_config` / `*_to_config` pair, and
`Simulator.to_config()` / `InertialSimulator.to_config()` write a whole
run back out. `*_to_config` emits a **canonical** configuration: it omits
keys left at their default, writes quantities in the unit they are held in,
and resolves the aliases above. So `x_to_config(x_from_config(config))`
equals `config` for a configuration already written in canonical form, and
is otherwise the canonical spelling of the same thing -- feeding it back
through `*_from_config` gives an equal object, and writing that out again
gives an identical configuration.

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

`*_to_config` writes a path back **exactly as it was given**. Rewriting
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
parameterised scalings** (`_SCALING_TYPES` below), every argument of which
is a number or a quantity: there is nothing in a configuration file left
to evaluate, and so nothing to sanitise. A shape none of them can make
belongs in a `Tabulated` scaling's CSV, or in Python.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml
import astropy.units as u

from .config_utils import (_as_mapping, _boolean, _check_keys, _collapse,
                           _format_quantity, _integer, _number, _quantity,
                           _resolve_path, _searched_dir, _text, _type_name)
from .earth import Earth
from .observation_strategy import ObservationStrategy
from .reco import SimpleTraditionalReconstructor
from .source import Source
from .spacecraft_history import SpacecraftHistory


__all__ = ['load_config',
           'detector_from_config', 'detector_to_config',
           'reconstructor_from_config', 'reconstructor_to_config',
           'spacecraft_history_from_config',
           'simulator_from_config', 'simulator_to_config']


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


# ---------------------------------------------------------------------------
# Detector and Earth
# ---------------------------------------------------------------------------


#: Accepted spellings of every detector type, mapped to the canonical one.
_DETECTOR_TYPES = {'ToyTracker2D': 'ToyTracker2D'}

_DETECTOR_KEYS = ('type', 'material', 'layer_length', 'layer_positions',
                  'layer_thickness', 'energy_resolution', 'energy_threshold')


def detector_from_config(config, where = 'detector'):
    """
    Build a detector from its configuration block.

    ```yaml
    type: ToyTracker2D
    material: Ge
    layer_length: 16 cm
    layer_positions: "[30, 0, 1, 2] cm"
    layer_thickness: 5 mm       # scalar, or one entry per layer
    energy_resolution: 0.01     # plain number, or one per layer
    energy_threshold: 20 keV    # scalar, or one per layer
    ```

    Every key is required: `ToyTracker2D` has no defaults of its own, and
    guessing one here would put a number in the detector that is nowhere in
    the file.

    Parameters
    ----------
    config : mapping
        The `detector` block.
    where : str
        Label for this block in error messages.

    Returns
    -------
    `ToyTracker2D`

    Raises
    ------
    ValueError
        On an unknown or missing key, an unparseable or wrong-unit value,
        an unknown material, or anything the detector itself rejects
        (overlapping layers, a per-layer array of the wrong length).
    """

    # Imported here, not at module level: `gammaraytoys.detectors` imports
    # `gammaraytoys.sims` for `Photon`, so a module-level import would close
    # a cycle and break `import gammaraytoys` outright.
    from ..detectors import ToyTracker2D

    block = _as_mapping(config, where)
    _check_keys(block, where, _DETECTOR_KEYS, required = _DETECTOR_KEYS)
    _type_name(block, where, _DETECTOR_TYPES, 'detector')

    material = _text(block, 'material', where, required = True)

    kwargs = dict(
        material = material,
        layer_length = _quantity(block, 'layer_length', where, u.cm, required = True),
        layer_positions = _quantity(block, 'layer_positions', where, u.cm,
                                    required = True, shape = 'array'),
        layer_thickness = _quantity(block, 'layer_thickness', where, u.cm,
                                    required = True, shape = 'any'),
        energy_resolution = _number(block, 'energy_resolution', where,
                                    required = True, minimum = 0, shape = 'any'),
        energy_threshold = _quantity(block, 'energy_threshold', where, u.keV,
                                     required = True, shape = 'any'))

    try:
        return ToyTracker2D(**kwargs)
    except Exception as err:
        raise ValueError(
            f"{where}: could not build a ToyTracker2D from this block "
            f"({type(err).__name__}: {err}).") from err


def detector_to_config(detector):
    """
    Write a detector back out as a configuration block.

    Parameters
    ----------
    detector : `ToyTracker2D`
        The detector to describe.

    Returns
    -------
    dict
        A block `detector_from_config` reads back into an equal detector.
        Per-layer values that are the same on every layer are collapsed to
        the single value they were written as.

    Raises
    ------
    ValueError
        If the detector's material was built directly rather than by name,
        and so has no name to write.
    """

    material = getattr(detector.material, 'name', None)

    if material is None:
        raise ValueError(
            "this detector's material was built directly from a density and "
            "an attenuation table rather than by name, so there is no material "
            "name to write into a configuration.")

    resolution = _collapse(np.asarray(detector.energy_resolution))

    return {'type': 'ToyTracker2D',
            'material': material,
            'layer_length': _format_quantity(detector.size),
            'layer_positions': _format_quantity(detector.layer_positions),
            'layer_thickness': _format_quantity(_collapse(detector.layer_thickness)),
            'energy_resolution': (float(resolution) if np.ndim(resolution) == 0
                                  else [float(item) for item in resolution]),
            'energy_threshold': _format_quantity(_collapse(detector.energy_threshold))}


# ---------------------------------------------------------------------------
# Observation strategies, spacecraft history, reconstructor
# ---------------------------------------------------------------------------


#: Accepted spellings of every generated-orbit type.
_HISTORY_TYPES = {'elliptical_orbit': 'elliptical_orbit'}

#: Orbit keys that are quantities, with the unit each must be convertible
#: to and the smallest value it may take (`None` where it may be negative:
#: an angle around the orbit and an epoch may both run backwards, a size and
#: a span of time may not). The remaining ones (`eccentricity`,
#: `livetime_fraction`) are plain numbers.
_ORBIT_QUANTITIES = {'semi_major_axis': (u.km, 0),
                     'duration': (u.s, 0),
                     'time_step': (u.s, 0),
                     'argument_of_periapsis': (u.deg, None),
                     'initial_time': (u.s, None)}


def spacecraft_history_from_config(config, where = 'spacecraft_history',
                                   earth = None, base_dir = None):
    """
    Build a `SpacecraftHistory` from its configuration entry.

    The entry is either a path to a `.ori` file:

    ```yaml
    spacecraft_history: iss.ori
    ```

    or an orbit to generate, which is how every cosimita notebook builds
    one:

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
      observation_strategy: {type: ZenithPointing}
    ```

    The second form is an extension beyond the plan's Section 7 sketch,
    which shows only a path. A capstone that cannot express an orbit would
    be a thin capstone: `SpacecraftHistory.from_elliptical_orbit` is what
    the notebooks actually use, and it needs an observation strategy the
    sketch has nowhere to put.

    Parameters
    ----------
    config : str or mapping
        The `spacecraft_history` entry.
    where : str
        Label for this entry in error messages.
    earth : `Earth` or None
        The Earth the history is validated against (`orbit_radius >
        earth.radius`) and that a `TargetedPointing` strategy points
        around. `None` falls back to `SpacecraftHistory`'s own default.
    base_dir : `pathlib.Path` or None
        The directory a relative `.ori` path is taken relative to -- the
        directory of the configuration file, when there was one. `None`
        (the default, and what a configuration handed in as a mapping
        gets) leaves a relative path resolving against the working
        directory. An absolute path is unaffected either way.

    Returns
    -------
    `SpacecraftHistory`

    Raises
    ------
    ValueError
        On an unknown type or key, a missing required key, a bad quantity,
        or anything `SpacecraftHistory` itself rejects (a perigee inside
        the Earth, a non-positive duration, a malformed `.ori` file).
    FileNotFoundError
        If the named `.ori` file does not exist.
    """

    return _spacecraft_history_from_config(config, where, earth, base_dir)[0]


def _spacecraft_history_from_config(config, where, earth, base_dir = None):
    """
    Build a `SpacecraftHistory` and the canonical block describing it.

    A generated `SpacecraftHistory` keeps only its sampled rows, not the
    Kepler elements that produced them, and one read from a file does not
    remember the file. Neither can be recovered from the object, so the
    canonical block is built here, where both are still in hand, and the
    simulator keeps it for `simulator_to_config`.

    Parameters
    ----------
    config : str or mapping
        The `spacecraft_history` entry.
    where : str
        Label for this entry in error messages.
    earth : `Earth` or None
        The Earth to validate against and to hand to a `TargetedPointing`.
    base_dir : `pathlib.Path` or None
        The directory a relative `.ori` path resolves against. See
        `spacecraft_history_from_config`.

    Returns
    -------
    history : `SpacecraftHistory`
        The history.
    block : str or dict
        The canonical configuration entry describing it. A path is the
        string the configuration wrote, never the resolved one: writing
        the resolved path back out would turn a portable configuration
        into a machine-specific one.

    Raises
    ------
    ValueError
        See `spacecraft_history_from_config`.
    """

    if isinstance(config, str):
        path = _resolve_path(config, base_dir)
        try:
            history = SpacecraftHistory.open(path, earth = earth)
        except FileNotFoundError as err:
            # As for a Tabulated scaling's 'file': the type is kept, only
            # the message gains the key that asked for the path and the
            # directory it was looked for in.
            raise FileNotFoundError(
                f"{where} = {config!r} does not exist "
                f"(looked in {_searched_dir(path)}).") from err
        except Exception as err:
            raise ValueError(
                f"{where}: could not read the spacecraft history from "
                f"{config!r} ({type(err).__name__}: {err}).") from err

        return history, config

    block = _as_mapping(config, where)

    _type_name(block, where, _HISTORY_TYPES, 'spacecraft history')

    allowed = ('type', 'eccentricity', 'livetime_fraction',
               'observation_strategy') + tuple(_ORBIT_QUANTITIES)
    _check_keys(block, where, allowed, required = ('semi_major_axis',))

    kwargs = {}
    canonical = {'type': 'elliptical_orbit'}

    for key, (unit, minimum) in _ORBIT_QUANTITIES.items():
        value = _quantity(block, key, where, unit, minimum = minimum,
                          required = (key == 'semi_major_axis'))
        if value is not None:
            kwargs[key] = value
            canonical[key] = _format_quantity(value)

    for key, maximum in (('eccentricity', None), ('livetime_fraction', 1.0)):
        value = _number(block, key, where, minimum = 0, maximum = maximum)
        if value is not None:
            kwargs[key] = value
            canonical[key] = value

    if 'observation_strategy' in block:
        strategy = ObservationStrategy.from_config(
            block['observation_strategy'], f"{where}.observation_strategy", earth)
        kwargs['observation_strategy'] = strategy
        canonical['observation_strategy'] = strategy.to_config()

    try:
        history = SpacecraftHistory.from_elliptical_orbit(earth = earth, **kwargs)
    except Exception as err:
        raise ValueError(
            f"{where}: could not generate the orbit "
            f"({type(err).__name__}: {err}).") from err

    return history, canonical


#: Accepted spellings of every reconstructor type.
_RECONSTRUCTOR_TYPES = {'SimpleTraditionalReconstructor': 'SimpleTraditionalReconstructor'}


def reconstructor_from_config(config, where = 'reconstructor'):
    """
    Build a `Reconstructor` from its configuration block.

    ```yaml
    {type: SimpleTraditionalReconstructor}
    ```

    Parameters
    ----------
    config : mapping
        The `reconstructor` block.
    where : str
        Label for this block in error messages.

    Returns
    -------
    `Reconstructor`

    Raises
    ------
    ValueError
        On an unknown type or key.
    """

    block = _as_mapping(config, where)
    _type_name(block, where, _RECONSTRUCTOR_TYPES, 'reconstructor')
    _check_keys(block, where, ('type',))

    return SimpleTraditionalReconstructor()


def reconstructor_to_config(reconstructor):
    """
    Write a `Reconstructor` back out as a configuration block.

    Parameters
    ----------
    reconstructor : `Reconstructor`
        The reconstructor to describe.

    Returns
    -------
    dict
        `{'type': 'SimpleTraditionalReconstructor'}`.

    Raises
    ------
    ValueError
        If `reconstructor` is not a type a configuration can name.
    """

    if isinstance(reconstructor, SimpleTraditionalReconstructor):
        return {'type': 'SimpleTraditionalReconstructor'}

    raise ValueError(
        f"{type(reconstructor).__name__} is not a reconstructor a "
        f"configuration can describe; the types that are: "
        f"{sorted(set(_RECONSTRUCTOR_TYPES.values()))}.")


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------


#: Top-level keys both simulators accept.
_COMMON_TOP_KEYS = ('detector', 'earth', 'sources', 'reconstructor',
                    'doppler_broadening', 'random_seed')

#: Largest seed `numpy.random.seed` accepts.
_MAX_SEED = 2 ** 32 - 1


def _sources_from_config(config, where, earth, base_dir = None):
    """
    Build every source in the `sources` list, and collect their names.

    Parameters
    ----------
    config : sequence
        The `sources` list.
    where : str
        Label for the list in error messages.
    earth : `Earth`
        The run's single Earth, handed to every source that needs one.
    base_dir : `pathlib.Path` or None
        The directory a relative path inside a source's `scaling` block
        resolves against. See `SourceScaling.from_config`.

    Returns
    -------
    sources : list of `Source`
        The sources, in the order they were written.
    names : dict
        Maps each named `Source` to its name, in the shape
        `write_event_csv`'s `source_names` argument wants. Sources with no
        `name` key are absent from it, and fall back to their class name in
        an event file.

    Raises
    ------
    ValueError
        If the list is missing or empty, if a name is not a non-empty
        string, if a name contains `'#'` (which `write_event_csv` refuses,
        because the reader would treat it as the start of a comment and
        silently drop the rest of the row), if two sources share a name, or
        on anything `Source.from_config` rejects.
    """

    if (not isinstance(config, Sequence) or isinstance(config, (str, bytes))
            or len(config) == 0):
        raise ValueError(
            f"{where}: must be a non-empty list of source blocks, got "
            f"{config!r}.")

    sources = []
    names = {}

    for index, item in enumerate(config):
        block = _as_mapping(item, f"{where}[{index}]")

        # The source's own name goes into the label, so that a mistake in
        # the third of five sources says which one it was.
        label = f"{where}[{index}]"
        if isinstance(block.get('name'), str):
            label = f"{label} ({block['name']})"

        source = Source.from_config(block, label, earth, base_dir)
        sources.append(source)

        if 'name' in block:
            name = _text(block, 'name', label, required = True)

            if not name:
                raise ValueError(f"{label}: key 'name' must not be empty.")

            if '#' in name:
                raise ValueError(
                    f"{label}: key 'name' = {name!r} must not contain '#'. "
                    f"`write_event_csv` refuses such a label: its reader "
                    f"treats '#' as the start of a comment and would silently "
                    f"drop the rest of every row this source wrote.")

            if name in names.values():
                raise ValueError(
                    f"{label}: two sources are both named {name!r}; names "
                    f"label events in an event file and must be unique.")

            names[source] = name

    return sources, names


def simulator_from_config(cls, config, inertial):
    """
    Build a simulator from a configuration -- the body of both
    `Simulator.from_config` and `InertialSimulator.from_config`.

    Parameters
    ----------
    cls : type
        The simulator class to instantiate.
    config : str, path-like or mapping
        A path to a YAML file, or an already-parsed mapping. Relative
        paths the configuration names for files beside it -- an `.ori`
        spacecraft history, a `TabulatedScaling`'s CSV -- are taken
        relative to the directory of that YAML file, so a configuration in
        `runs/` can be loaded from anywhere. A mapping came from no file
        and has no directory, so relative paths in one fall back to the
        working directory. Absolute paths are used as written, always.
    inertial : bool
        Whether `cls` is the inertial simulator, and so takes (and
        requires) a `spacecraft_history` and an `earth`.

    Returns
    -------
    `SimulatorBase`
        An instance of `cls`, carrying `source_names` and `random_seed`,
        and the provenance `to_config` needs.

    Raises
    ------
    ValueError
        On any unknown or missing key, any bad value, or anything the
        simulator itself rejects at construction (an unnormalized source,
        a far-field source aimed in the detector frame, an Earth that
        disagrees with the orbit).
    TypeError
        If `config` is neither a path nor a mapping.
    """

    where = 'config'
    block = load_config(config)

    # Where a relative sibling path in this configuration points. `None`
    # for a mapping, which has no file and so no directory of its own.
    base_dir = _config_base_dir(config)

    allowed = _COMMON_TOP_KEYS + (('spacecraft_history',) if inertial else ())

    if not inertial and 'spacecraft_history' in block:
        raise ValueError(
            f"{where}: 'spacecraft_history' belongs to "
            f"InertialSimulator.from_config -- the detector-frame Simulator "
            f"has no spacecraft, no orbit and no occultation. Use "
            f"InertialSimulator.from_config for a configuration with a "
            f"spacecraft history.")

    required = ('detector', 'sources') + (('spacecraft_history',) if inertial else ())
    _check_keys(block, where, allowed, required = required)

    # The Earth first: it is shared by the history, by TargetedPointing and
    # by EarthAlbedoSource, and every one of them must get *this* one. A run
    # with two different planets is a bug this project has shipped before.
    earth = (Earth.from_config(block['earth'], f"{where}.earth")
             if 'earth' in block else Earth())

    detector = detector_from_config(block['detector'], f"{where}.detector")

    reconstructor = (reconstructor_from_config(block['reconstructor'],
                                               f"{where}.reconstructor")
                     if 'reconstructor' in block
                     else SimpleTraditionalReconstructor())

    doppler_broadening = _boolean(block, 'doppler_broadening', where, default = True)

    seed = _integer(block, 'random_seed', where, minimum = 0, maximum = _MAX_SEED)

    sources, names = _sources_from_config(block['sources'], f"{where}.sources",
                                          earth, base_dir)

    kwargs = {'detector': detector,
              'sources': sources,
              'reconstructor': reconstructor,
              'doppler_broadening': doppler_broadening}

    history_block = None

    if inertial:
        history, history_block = _spacecraft_history_from_config(
            block['spacecraft_history'], f"{where}.spacecraft_history", earth,
            base_dir)
        kwargs['spacecraft_history'] = history
        kwargs['earth'] = earth

    try:
        simulator = cls(**kwargs)
    except Exception as err:
        raise ValueError(
            f"{where}: could not build a {cls.__name__} from this "
            f"configuration ({type(err).__name__}: {err}).") from err

    simulator.source_names = names
    simulator.random_seed = seed
    simulator._config_earth = earth if 'earth' in block else None

    if inertial:
        simulator._config_spacecraft_history = history_block

    # Seeded last, after every object is built, so that what follows the
    # call -- the run itself -- always starts from the same RNG state, no
    # matter what construction did or did not draw. Two `from_config` calls
    # with the same seed, each followed by a run, give byte-identical runs.
    if seed is not None:
        np.random.seed(seed)

    return simulator


def simulator_to_config(simulator):
    """
    Write a simulator back out as a configuration.

    Parameters
    ----------
    simulator : `SimulatorBase`
        The simulator to describe. Its detector, sources, reconstructor and
        Earth are read off the objects themselves; the source names and the
        spacecraft history's provenance -- neither of which the objects
        keep -- come from what `from_config` recorded.

    Returns
    -------
    dict
        A configuration `from_config` reads back into an equal simulator,
        holding only plain strings, numbers, lists and dictionaries, so it
        can be written straight out with `yaml.safe_dump`.

    Raises
    ------
    ValueError
        If any part of the simulator is not something a configuration can
        describe, or if the simulator has a spacecraft history but was not
        built by `from_config` and so cannot say where that history came
        from.
    """

    config = {'detector': detector_to_config(simulator.detector)}

    earth = getattr(simulator, 'earth', None)

    if earth is None:
        earth = simulator._config_earth

    if earth is not None:
        config['earth'] = earth.to_config()

    config['reconstructor'] = reconstructor_to_config(simulator.reconstructor)

    if not simulator.doppler_broadening:
        config['doppler_broadening'] = False

    if hasattr(simulator, 'spacecraft_history'):
        history_block = simulator._config_spacecraft_history

        if history_block is None:
            raise ValueError(
                "this simulator's spacecraft history was not built by "
                "`from_config`, so there is nothing to write for it: a "
                "generated `SpacecraftHistory` keeps its sampled rows, not "
                "the orbital elements that produced them, and one read from "
                "a file does not remember the file.")

        config['spacecraft_history'] = history_block

    if simulator.random_seed is not None:
        config['random_seed'] = simulator.random_seed

    names = simulator.source_names or {}

    config['sources'] = [source.to_config(names.get(source))
                         for source in simulator.sources]

    return config
