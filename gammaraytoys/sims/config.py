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
    scaling: {type: Function, expression: "1 + 0.5*sin(2*pi*t/5400)"}
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
`Function`) and the full class name (`PowerLawSpectrum`,
`FunctionScaling`); the short form is what `*_to_config` writes back.

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

The expression evaluator
------------------------

`FunctionScaling` built from a configuration string is the one place with
an obvious injection hazard, and it is handled by `TimeExpression`. The
defence is an **AST whitelist**, not string matching: see that class for
what is allowed, what is rejected, and why the string-matching defence the
plan describes is necessary but not sufficient.
"""

import ast
import difflib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml
import astropy.units as u

from ..coordinates import Cartesian2D
from .earth import Earth
from .observation_strategy import (ZenithPointing, NadirPointing, InertialPointing,
                                   SpinPointing, TargetedPointing)
from .reco import SimpleTraditionalReconstructor
from .scaling import ConstantScaling, TabulatedScaling, FunctionScaling
from .source import (PointSource, IsotropicSource, NearPointSource, ExtendedSource,
                     EarthAlbedoSource)
from .spacecraft_history import SpacecraftHistory
from .spectrum import MonoenergeticSpectrum, PowerLawSpectrum, MultiComponentSpectrum


__all__ = ['load_config', 'TimeExpression',
           'detector_from_config', 'detector_to_config',
           'earth_from_config', 'earth_to_config',
           'spectrum_from_config', 'spectrum_to_config',
           'scaling_from_config', 'scaling_to_config',
           'source_from_config', 'source_to_config',
           'observation_strategy_from_config', 'observation_strategy_to_config',
           'reconstructor_from_config', 'reconstructor_to_config',
           'spacecraft_history_from_config',
           'simulator_from_config', 'simulator_to_config']


# ---------------------------------------------------------------------------
# Low-level validation
#
# Every helper here takes the `where` of the block it is validating -- a
# human-readable path into the file, like "sources[1] (albedo).spectrum" --
# and puts it in front of every message. Astropy's own errors say what
# could not be parsed but never where it came from, which in a
# configuration file is the only thing the user needs to know.
# ---------------------------------------------------------------------------


def _suggest(key, allowed):
    """
    A `did you mean ...?` fragment for a mistyped key, or an empty string.

    Parameters
    ----------
    key : str
        The key that was not recognised.
    allowed : iterable of str
        The keys that would have been recognised.

    Returns
    -------
    str
        Either `""` or a fragment like `" Did you mean 'radius'?"`.
    """

    close = difflib.get_close_matches(key, sorted(allowed), n = 1, cutoff = 0.6)

    if not close:
        return ""

    return f" Did you mean {close[0]!r}?"


def _as_mapping(value, where):
    """
    Check that a configuration block is a mapping with string keys.

    Parameters
    ----------
    value : object
        The block, as parsed from YAML.
    where : str
        Human-readable path to this block, used in error messages.

    Returns
    -------
    dict
        A shallow copy of `value`, so nothing downstream can mutate the
        caller's configuration.

    Raises
    ------
    ValueError
        If `value` is not a mapping, or if any of its keys is not a string.
    """

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{where}: expected a block of key/value pairs, got "
            f"{type(value).__name__} ({value!r}).")

    for key in value:
        if not isinstance(key, str):
            raise ValueError(
                f"{where}: keys must be strings, got {key!r} "
                f"({type(key).__name__}).")

    return dict(value)


def _check_keys(block, where, allowed, required = ()):
    """
    Reject unknown keys and demand required ones.

    Unknown keys are an error, not a warning (Section 7 of the plan): a
    typo that is silently ignored turns into a run that quietly does
    something other than what the file says.

    Parameters
    ----------
    block : dict
        The block to check.
    where : str
        Human-readable path to this block, used in error messages.
    allowed : iterable of str
        Every key that may appear here.
    required : iterable of str
        Keys that must appear here.

    Raises
    ------
    ValueError
        If `block` has a key not in `allowed`, or lacks one in `required`.
    """

    allowed = set(allowed)

    for key in block:
        if key not in allowed:
            raise ValueError(
                f"{where}: unknown key {key!r}.{_suggest(key, allowed)} "
                f"Keys allowed here: {sorted(allowed)}.")

    missing = [key for key in required if key not in block]

    if missing:
        raise ValueError(
            f"{where}: missing required key(s) {missing}. "
            f"Keys allowed here: {sorted(allowed)}.")


def _parse_quantity(text):
    """
    Turn one unit-bearing string into a `Quantity`, arrays included.

    The scalar form (`"16 cm"`) goes straight to astropy, which has always
    understood it. The array form (`"[30, 0, 1] cm"`) is taken apart here
    instead: parsing a bracketed list out of a string is a *recent*
    astropy feature, this package requires no particular astropy version
    and supports Python 3.10, where an older one is what gets installed --
    and there the whole array form of the schema silently stops working.
    Splitting the brackets off ourselves and handing astropy only a unit
    and a list of numbers works on every version. It also suits this
    module: everything here is validated by hand.

    Parameters
    ----------
    text : str
        The value as it was written, e.g. `"16 cm"` or `"[30, 0, 1] cm"`.
        Whitespace inside the brackets is free.

    Returns
    -------
    `astropy.units.Quantity`
        Scalar for the scalar form, one-dimensional for the array form --
        `"[5] mm"` is a one-element array, not a scalar.

    Raises
    ------
    ValueError
        If the brackets do not close, hold no numbers, or hold something
        that is not a number, or if the trailing text is not a unit
        astropy knows. `_quantity` wraps whatever comes out of here in the
        message that names the key it came from.
    """

    stripped = text.strip()

    if not stripped.startswith('['):
        return u.Quantity(stripped)

    closing = stripped.find(']')

    if closing < 0:
        raise ValueError(
            f'Cannot parse "{text}" as a Quantity: the array is missing its '
            f'closing "]".')

    inside = stripped[1:closing]
    unit_text = stripped[closing + 1:].strip()

    try:
        # `ast.literal_eval` is the safe half of `eval`: it reads number
        # literals and lists of them, and nothing else -- no names, no
        # calls. The same reason `TimeExpression` below never uses bare
        # `eval`, for a much smaller job.
        values = ast.literal_eval(f"[{inside}]")
    except (SyntaxError, ValueError) as err:
        raise ValueError(
            f'Cannot parse "{text}" as a Quantity: {inside!r} is not a list '
            f'of numbers.') from err

    if len(values) == 0:
        raise ValueError(
            f'Cannot parse "{text}" as a Quantity: the array is empty.')

    try:
        numbers = np.array(values, dtype = float)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f'Cannot parse "{text}" as a Quantity: {inside!r} is not a list '
            f'of numbers.') from err

    return numbers * u.Unit(unit_text)


def _quantity(block, key, where, unit, default = None, required = False,
              minimum = None, maximum = None, shape = 'scalar'):
    """
    Read a unit-bearing value: a string astropy parses.

    Parameters
    ----------
    block : dict
        The block to read from.
    key : str
        The key to read.
    where : str
        Human-readable path to `block`, used in error messages.
    unit : `astropy.units.UnitBase`
        The unit the value must be convertible to. Only the physical type
        matters -- `"5 mm"` is accepted where `u.cm` is expected -- and the
        value is returned in whatever unit it was written in, not converted.
    default : `astropy.units.Quantity` or None
        Returned when `key` is absent and `required` is `False`.
    required : bool
        Whether the key must be present.
    minimum, maximum : `astropy.units.Quantity`, 0 or None
        Inclusive bounds, checked on every element when given. Almost
        always `0`: a negative flux or a negative duration is not a
        physical thing a user meant, and left alone it does not fail here
        but deep inside numpy, at the first interval, with no idea which
        key or which file it came from. A bare `0` is allowed because zero
        is zero in any unit; any other bound must carry one.
    shape : {'scalar', 'array', 'any'}
        `'scalar'` demands a single value, `'array'` demands a
        one-dimensional array of at least one element, `'any'` accepts
        either.

    Returns
    -------
    `astropy.units.Quantity` or None
        The parsed value, or `default` when the key is absent.

    Raises
    ------
    ValueError
        If the key is required and absent; if the value is not a string; if
        astropy cannot parse it; if its unit is not convertible to `unit`;
        if it is not finite; if it falls outside the bounds; or if its
        shape is not the one asked for.
    """

    if key not in block:
        if required:
            raise ValueError(f"{where}: missing required key {key!r}.")
        return default

    text = block[key]

    if not isinstance(text, str):
        extra = ""
        if isinstance(text, Sequence) and not isinstance(text, (str, bytes)):
            extra = (" A YAML list carries no unit: write the whole array as one "
                     "quoted string instead, e.g. \"[30, 0, 1] cm\".")
        elif isinstance(text, (int, float)):
            extra = " A bare number carries no unit: write it as a string, with one."
        raise ValueError(
            f"{where}: key {key!r} must be a string astropy can parse as a "
            f"quantity (e.g. \"16 cm\"), got {type(text).__name__} "
            f"({text!r}).{extra}")

    try:
        quantity = _parse_quantity(text)
    except Exception as err:
        raise ValueError(
            f"{where}: key {key!r} = {text!r} is not a valid quantity "
            f"({type(err).__name__}: {err}). Expected a number and a unit "
            f"astropy understands, e.g. \"16 cm\".") from err

    if not quantity.unit.is_equivalent(unit):
        raise ValueError(
            f"{where}: key {key!r} = {text!r} has unit "
            f"{quantity.unit.to_string()!r}, which is not convertible to "
            f"{u.Unit(unit).to_string()!r} (physical type "
            f"{str(u.Unit(unit).physical_type)!r}).")

    if not np.all(np.isfinite(quantity.value)):
        raise ValueError(
            f"{where}: key {key!r} = {text!r} must be finite.")

    if minimum is not None and not np.all(quantity >= minimum):
        raise ValueError(
            f"{where}: key {key!r} = {text!r} must be >= {minimum}.")

    if maximum is not None and not np.all(quantity <= maximum):
        raise ValueError(
            f"{where}: key {key!r} = {text!r} must be <= {maximum}.")

    ndim = np.ndim(quantity.value)

    if shape == 'scalar' and ndim != 0:
        raise ValueError(
            f"{where}: key {key!r} = {text!r} must be a single value, not an "
            f"array of {np.size(quantity.value)}.")

    if shape == 'array':
        if ndim != 1:
            raise ValueError(
                f"{where}: key {key!r} = {text!r} must be a one-dimensional "
                f"array, e.g. \"[30, 0, 1] cm\".")
        if quantity.size == 0:
            raise ValueError(f"{where}: key {key!r} must have at least one element.")

    return quantity


def _number(block, key, where, default = None, required = False,
            minimum = None, maximum = None, shape = 'scalar'):
    """
    Read a plain (unitless) number, or a list of them.

    A string is accepted and converted with `float`, because PyYAML does
    **not** resolve `1e-3` as a float -- its implicit float pattern needs
    either a decimal point or a signed exponent -- so an ordinary-looking
    `energy_resolution: 1e-2` arrives here as the string `'1e-2'`.

    Parameters
    ----------
    block : dict
        The block to read from.
    key : str
        The key to read.
    where : str
        Human-readable path to `block`, used in error messages.
    default : float, list of float or None
        Returned when `key` is absent and `required` is `False`.
    required : bool
        Whether the key must be present.
    minimum, maximum : float or None
        Inclusive bounds, checked on every element when given.
    shape : {'scalar', 'any'}
        `'scalar'` demands a single number; `'any'` also accepts a
        non-empty list of numbers.

    Returns
    -------
    float, list of float, or None
        The parsed value, or `default` when the key is absent.

    Raises
    ------
    ValueError
        If the key is required and absent, if the value (or any element) is
        not a real number, is not finite, or falls outside the bounds.
    """

    if key not in block:
        if required:
            raise ValueError(f"{where}: missing required key {key!r}.")
        return default

    value = block[key]

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if shape != 'any':
            raise ValueError(
                f"{where}: key {key!r} must be a single number, got a list "
                f"({value!r}).")
        if len(value) == 0:
            raise ValueError(f"{where}: key {key!r} must have at least one element.")
        return [_scalar_number(item, key, where, minimum, maximum) for item in value]

    return _scalar_number(value, key, where, minimum, maximum)


def _scalar_number(value, key, where, minimum = None, maximum = None):
    """
    Convert one configuration value to a finite, in-range `float`.

    Parameters
    ----------
    value : object
        The value as parsed from YAML: a number, or a string spelling one.
    key : str
        The key it came from, for error messages.
    where : str
        Human-readable path to the block it came from.
    minimum, maximum : float or None
        Inclusive bounds, when given.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If `value` is not a real number (booleans are deliberately not
        numbers here), is not finite, or is out of range.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(
            f"{where}: key {key!r} must be a number, got "
            f"{type(value).__name__} ({value!r}).")

    try:
        number = float(value)
    except ValueError as err:
        raise ValueError(
            f"{where}: key {key!r} = {value!r} is not a number.") from err

    if not np.isfinite(number):
        raise ValueError(f"{where}: key {key!r} = {value!r} must be finite.")

    if minimum is not None and number < minimum:
        raise ValueError(
            f"{where}: key {key!r} = {value!r} must be >= {minimum}.")

    if maximum is not None and number > maximum:
        raise ValueError(
            f"{where}: key {key!r} = {value!r} must be <= {maximum}.")

    return number


def _integer(block, key, where, default = None, required = False,
             minimum = None, maximum = None):
    """
    Read an integer.

    Parameters
    ----------
    block : dict
        The block to read from.
    key : str
        The key to read.
    where : str
        Human-readable path to `block`, used in error messages.
    default : int or None
        Returned when `key` is absent and `required` is `False`.
    required : bool
        Whether the key must be present.
    minimum, maximum : int or None
        Inclusive bounds, when given.

    Returns
    -------
    int or None

    Raises
    ------
    ValueError
        If the key is required and absent, if the value is not an integer
        (a `bool` is not), or if it is out of range.
    """

    if key not in block:
        if required:
            raise ValueError(f"{where}: missing required key {key!r}.")
        return default

    value = block[key]

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{where}: key {key!r} must be an integer, got "
            f"{type(value).__name__} ({value!r}).")

    if minimum is not None and value < minimum:
        raise ValueError(f"{where}: key {key!r} = {value!r} must be >= {minimum}.")

    if maximum is not None and value > maximum:
        raise ValueError(f"{where}: key {key!r} = {value!r} must be <= {maximum}.")

    return value


def _boolean(block, key, where, default = None):
    """
    Read a boolean.

    Parameters
    ----------
    block : dict
        The block to read from.
    key : str
        The key to read.
    where : str
        Human-readable path to `block`, used in error messages.
    default : bool or None
        Returned when `key` is absent.

    Returns
    -------
    bool or None

    Raises
    ------
    ValueError
        If the value is not a YAML boolean. Strings are refused on purpose:
        `"false"` is a true string, and quietly reading it as `True` is the
        kind of silent misconfiguration this schema exists to prevent.
    """

    if key not in block:
        return default

    value = block[key]

    if not isinstance(value, bool):
        raise ValueError(
            f"{where}: key {key!r} must be a boolean (`true` or `false`), got "
            f"{type(value).__name__} ({value!r}).")

    return value


def _text(block, key, where, default = None, required = False, choices = None):
    """
    Read a string.

    Parameters
    ----------
    block : dict
        The block to read from.
    key : str
        The key to read.
    where : str
        Human-readable path to `block`, used in error messages.
    default : str or None
        Returned when `key` is absent and `required` is `False`.
    required : bool
        Whether the key must be present.
    choices : iterable of str or None
        When given, the only accepted values.

    Returns
    -------
    str or None

    Raises
    ------
    ValueError
        If the key is required and absent, if the value is not a string, or
        if it is not one of `choices`.
    """

    if key not in block:
        if required:
            raise ValueError(f"{where}: missing required key {key!r}.")
        return default

    value = block[key]

    if not isinstance(value, str):
        raise ValueError(
            f"{where}: key {key!r} must be a string, got "
            f"{type(value).__name__} ({value!r}).")

    if choices is not None and value not in choices:
        raise ValueError(
            f"{where}: key {key!r} = {value!r} is not one of "
            f"{sorted(choices)}.{_suggest(value, choices)}")

    return value


def _type_name(block, where, table, kind):
    """
    Read and resolve a block's `type` key.

    Parameters
    ----------
    block : dict
        The block to read from.
    where : str
        Human-readable path to `block`, used in error messages.
    table : dict
        Maps every accepted spelling (including aliases) to the canonical
        type name.
    kind : str
        What is being built (`'spectrum'`, `'source'`, ...), for the error
        message.

    Returns
    -------
    str
        The canonical type name.

    Raises
    ------
    ValueError
        If `type` is missing, is not a string, or is not a known type.
    """

    name = _text(block, 'type', where, required = True)

    if name not in table:
        known = sorted(set(table.values()))
        raise ValueError(
            f"{where}: unknown {kind} type {name!r}.{_suggest(name, table)} "
            f"Known {kind} types: {known}.")

    return table[name]


def _format_quantity(quantity):
    """
    Write a `Quantity` as a string this module reads back exactly.

    Values go out through `repr`, so the round trip is bit-exact rather
    than merely close, and arrays get the bracketed, comma-separated form
    astropy's own string parser accepts.

    Parameters
    ----------
    quantity : `astropy.units.Quantity`
        Scalar or one-dimensional.

    Returns
    -------
    str
        E.g. `'16.0 cm'`, `'[30.0, 0.0, 1.0] cm'`, `'0.001 1 / (cm s)'`.
    """

    value = quantity.value

    if np.ndim(value) == 0:
        text = repr(float(value))
    else:
        text = "[" + ", ".join(repr(float(item)) for item in np.atleast_1d(value)) + "]"

    unit = quantity.unit.to_string()

    return f"{text} {unit}" if unit else text


def _collapse(values):
    """
    Reduce a per-layer array to a single value when every entry is equal.

    The detector broadcasts `layer_thickness`, `energy_resolution` and
    `energy_threshold` to one entry per layer, so a configuration that gave
    one scalar comes back as an array. Collapsing it keeps the written
    configuration identical to the one that was read.

    Parameters
    ----------
    values : `astropy.units.Quantity` or array-like
        The per-layer values.

    Returns
    -------
    `astropy.units.Quantity` or numpy.ndarray
        A scalar if every entry is equal, otherwise `values` unchanged.
    """

    if np.ndim(values) == 0:
        return values

    if np.size(values) > 0 and np.all(values == np.ravel(values)[0]):
        return np.ravel(values)[0]

    return values


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


def _resolve_path(filename, base_dir):
    """
    Resolve a path a configuration named for a file beside it.

    Parameters
    ----------
    filename : str
        The path exactly as it was written in the configuration.
    base_dir : `pathlib.Path` or None
        The directory holding the configuration file, from
        `_config_base_dir`. `None` when there is no such directory.

    Returns
    -------
    `pathlib.Path`
        `filename` taken relative to `base_dir`, so a configuration in
        `runs/` finds `runs/iss.ori` however the process was started. An
        absolute `filename` is used as written, and so is a relative one
        when `base_dir` is `None`, which leaves it resolving against the
        working directory as it always has.
    """

    path = Path(filename)

    if path.is_absolute() or base_dir is None:
        return path

    return base_dir / path


def _searched_dir(path):
    """
    The directory a path was looked for in, for a "does not exist" message.

    Parameters
    ----------
    path : `pathlib.Path`
        The resolved path, from `_resolve_path`.

    Returns
    -------
    `pathlib.Path`
        The containing directory, made absolute. "Not found" without a
        location is what makes a relative path in a configuration
        confusing in the first place: the whole question is *where* it was
        looked for.
    """

    return path.absolute().parent


# ---------------------------------------------------------------------------
# The expression evaluator
# ---------------------------------------------------------------------------


#: Constants an expression may name, on top of the time variable `t`.
_EXPRESSION_CONSTANTS = {'pi': np.pi,
                         'e': np.e}

#: Functions an expression may call. Numpy's, so they behave the same way
#: they do everywhere else in this package.
_EXPRESSION_FUNCTIONS = {'sin': np.sin,
                         'cos': np.cos,
                         'tan': np.tan,
                         'arcsin': np.arcsin,
                         'arccos': np.arccos,
                         'arctan': np.arctan,
                         'arctan2': np.arctan2,
                         'sinh': np.sinh,
                         'cosh': np.cosh,
                         'tanh': np.tanh,
                         'exp': np.exp,
                         'expm1': np.expm1,
                         'log': np.log,
                         'log1p': np.log1p,
                         'log2': np.log2,
                         'log10': np.log10,
                         'sqrt': np.sqrt,
                         'abs': np.abs,
                         'sign': np.sign,
                         'floor': np.floor,
                         'ceil': np.ceil,
                         'mod': np.mod,
                         'power': np.power,
                         'minimum': np.minimum,
                         'maximum': np.maximum,
                         'clip': np.clip,
                         'heaviside': np.heaviside}

#: Everything an expression may name. `t` is added per call.
_EXPRESSION_NAMESPACE = dict(_EXPRESSION_CONSTANTS, **_EXPRESSION_FUNCTIONS)

#: Binary operators an expression may use.
_EXPRESSION_BINARY_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                          ast.Mod, ast.Pow)

#: Unary operators an expression may use.
_EXPRESSION_UNARY_OPS = (ast.UAdd, ast.USub)

#: The name `_pow` is called under inside a rewritten expression. It is
#: deliberately *not* in `_EXPRESSION_NAMESPACE`, so an expression that
#: writes it itself is rejected by the whitelist, long before the rewrite.
_POW_NAME = '_pow'

#: The longest expression a configuration file may hold, checked before it
#: is parsed.
#:
#: Two costs survive everything else on this page, and one rule bounds
#: both, because both are paid for one character at a time.
#:
#: Integers are still arbitrary precision under `*` -- only `**` was taken
#: away, by `_pow` -- so `9999...*9999...*...` builds an integer as large
#: as the file is long, at quadratic cost: 200 four-thousand-digit factors
#: are 781 kB of YAML and 1.5 seconds of CPU for a 2.7-million-bit number.
#: And `_check_expression_node` and `_FloatPower.visit` both recurse once
#: per node, so a long chain of anything -- `-------...-1`, `2*2*2*...`
#: -- runs the interpreter out of stack and raises `RecursionError`, which
#: is not the `ValueError` this class documents.
#:
#: 250 characters, and not the rounder 1000, because the recursion is the
#: tighter of the two constraints: 496 characters of `-` are already
#: enough to exhaust CPython's default 1000-frame stack, so a cap has to
#: sit well below that to be worth having. It is still generous for what
#: it bounds -- a scaling is a one-liner, and the plan's own example,
#: `1 + 0.5*sin(2*pi*t/5400)`, is 24 characters.
_MAX_EXPRESSION_LENGTH = 250


def _reject(expression, reason):
    """
    Raise the one error every expression rejection raises.

    Parameters
    ----------
    expression : str
        The offending expression, quoted back to the user.
    reason : str
        Why it was rejected.

    Raises
    ------
    ValueError
        Always.
    """

    raise ValueError(
        f"expression {expression!r} is not allowed: {reason} An expression may "
        f"use the time variable 't' (in seconds), numbers, the operators "
        f"+ - * / // % **, the constants {sorted(_EXPRESSION_CONSTANTS)} and "
        f"calls to {sorted(_EXPRESSION_FUNCTIONS)} -- nothing else.")


def _suggest_function(name):
    """
    A `did you mean ...?` fragment for a mistyped function name.

    `_suggest` on its own is wrong here often enough to be worse than
    silence. It judges by edit distance, and a short word is close to
    everything: it answers both `min(1, t)` and `cosine(t)` with `'sin'`,
    when `minimum` and `cos` are the names plainly meant, and a wrong
    suggestion sends a student looking in the wrong place.

    A mistyped function name is nearly always a real name cut short or run
    long, so look for that first: a whitelisted name that extends what was
    typed, or that what was typed extends. Of those, take the one closest
    in length to what was typed -- the least-changed reading of it -- and
    break a remaining tie alphabetically. Only when nothing is related
    that way fall back to `_suggest`, which is shared with the key, type
    and choice messages and is left alone.

    Parameters
    ----------
    name : str
        The name that was called and is not a whitelisted function.

    Returns
    -------
    str
        Either `""` or a fragment like `" Did you mean 'minimum'?"`.
    """

    related = [candidate for candidate in sorted(_EXPRESSION_FUNCTIONS)
               if candidate.startswith(name) or name.startswith(candidate)]

    if not related:
        return _suggest(name, _EXPRESSION_FUNCTIONS)

    # `min` over a sorted list keeps the first of equally close names, so
    # the alphabetical tie-break costs nothing extra.
    closest = min(related, key = lambda candidate: abs(len(candidate) - len(name)))

    return f" Did you mean {closest!r}?"


def _check_expression_node(node, expression):
    """
    Recursively check one node of a parsed expression against the whitelist.

    Whitelist, not blacklist: any node type not named here is rejected,
    so a syntax this function has never heard of cannot slip through.

    Parameters
    ----------
    node : `ast.AST`
        The node to check.
    expression : str
        The whole expression, for error messages.

    Raises
    ------
    ValueError
        If the node, or anything below it, is not allowed.
    """

    if isinstance(node, ast.Expression):
        _check_expression_node(node.body, expression)

    elif isinstance(node, ast.Constant):
        # `True`/`False` are `int`s to `isinstance`, hence the explicit
        # `bool` check; strings, bytes, `None` and complex are all out.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            _reject(expression,
                    f"the constant {node.value!r} is not a real number.")

    elif isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            _reject(expression, f"the name {node.id!r} is being assigned to.")
        if node.id != 't' and node.id not in _EXPRESSION_NAMESPACE:
            _reject(expression, f"the name {node.id!r} is not defined.")
        if node.id in _EXPRESSION_FUNCTIONS:
            # Only reached for a function name used as a *value* -- a call's
            # own `func` is checked in the `ast.Call` branch and never
            # recursed into. `sin` on its own is a function object, not a
            # number: harmless, since `FunctionScaling` refuses to turn it
            # into a float, but the complaint belongs here, where the
            # expression is written, not at the first interval of a run.
            _reject(expression,
                    f"{node.id!r} is a function and must be called, as "
                    f"{node.id}(...).")

    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _EXPRESSION_BINARY_OPS):
            _reject(expression,
                    f"the operator {type(node.op).__name__} is not allowed.")

        _check_expression_node(node.left, expression)
        _check_expression_node(node.right, expression)

    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _EXPRESSION_UNARY_OPS):
            _reject(expression,
                    f"the operator {type(node.op).__name__} is not allowed.")
        _check_expression_node(node.operand, expression)

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            _reject(expression, "only a plain function name may be called.")
        if node.func.id not in _EXPRESSION_FUNCTIONS:
            _reject(expression,
                    f"{node.func.id!r} is not a callable function."
                    f"{_suggest_function(node.func.id)}")
        if node.keywords:
            _reject(expression, "keyword arguments are not allowed in a call.")
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                _reject(expression, "argument unpacking is not allowed in a call.")
            _check_expression_node(argument, expression)

    elif isinstance(node, ast.Attribute):
        # Named explicitly because it is one of the three holes the plan's
        # string-matching rule leaves open: `t.real.conjugate()` contains no
        # dunder at all. Today's namespace happens to expose nothing useful
        # through an attribute chain, but that is luck, not a property --
        # adding one convenience (numpy itself, a Quantity) would turn it
        # into an escape hatch.
        _reject(expression, "attribute access is not allowed.")

    elif isinstance(node, ast.Lambda):
        # The second hole: `(lambda: 1)()` contains no dunder either.
        _reject(expression, "lambdas are not allowed.")

    else:
        _reject(expression,
                f"{type(node).__name__} expressions are not allowed.")


def _pow(base, exponent):
    """
    Raise `base` to `exponent` in floating point, never in integers.

    Every `**` in a configuration expression is rewritten into a call to
    this function (see `_FloatPower`), so this is the *only* exponentiation
    a configuration file can reach.

    Why floats: Python's integers are arbitrary-precision, so `2**64` is an
    exact 65-bit integer, `(2**64)**64` an exact 4097-bit one, and every
    further nesting *squares* the number of digits. Four or five levels of
    that -- `(((2**64)**64)**64)**64`, which is one short line in a YAML
    file -- spend minutes of CPU and gigabytes of memory building a number
    the run then multiplies by zero and throws away. A float cannot do
    that: it is 64 bits wide however hard you push it, so the same
    expression overflows to `inf`, or raises `OverflowError`, in
    microseconds. Nothing legitimate is lost, because a scaling is a
    unitless multiplier and therefore a float in the end anyway.

    Parameters
    ----------
    base, exponent : float
        The operands. Converted with `float` before the power is taken, so
        that an integer written in the file cannot bring arbitrary
        precision back in through the side door.

    Returns
    -------
    float

    Raises
    ------
    OverflowError
        If the result is too large for a float. `TimeExpression.__call__`
        catches it along with every other arithmetic failure and re-raises
        a `ValueError` naming the expression and the time.
    """

    return float(base) ** float(exponent)


class _FloatPower(ast.NodeTransformer):
    """
    Rewrite every `**` in a checked expression into a `_pow(...)` call.

    Run this after `_check_expression_node` and, crucially, **before**
    `compile`: CPython folds constant arithmetic at compile time, so a
    literal `2**64` left in the tree would be evaluated inside `compile`
    itself and the defence would be worth nothing.

    A rewrite rather than one more rule about the shape of the tree,
    because the shape rules kept losing. "An exponent may not itself be a
    `**`" catches `9**9**9**9` but not `(2**64)**64`, where all of the
    growth is on the *left*. Adding "nor may the base be one" does not
    catch `((2**64*1)**64*1)**64` either, which breaks the pattern with a
    harmless `*1`. There is no end to that game, so stop playing it and
    take away the thing being attacked: the arbitrary-precision integer.
    """

    def visit_BinOp(self, node):
        """
        Parameters
        ----------
        node : `ast.BinOp`
            A binary operation, already checked against the whitelist.

        Returns
        -------
        `ast.AST`
            `node` itself, unless it is a `**`, in which case the
            equivalent call to `_pow`.
        """

        # Operands first, so that a `**` nested inside this one is
        # rewritten too, however deep it is.
        self.generic_visit(node)

        if not isinstance(node.op, ast.Pow):
            return node

        return ast.Call(func = ast.Name(id = _POW_NAME, ctx = ast.Load()),
                        args = [node.left, node.right],
                        keywords = [])


class TimeExpression:
    """
    A safe callable built from a configuration expression string, for
    `FunctionScaling`.

    The variable `t` is the evaluation time **in seconds**, so the plan's
    own example, `"1 + 0.5*sin(2*pi*t/5400)"`, is a 5400-second period.

    Security
    --------

    An expression from a configuration file is the one obvious injection
    hazard in this package. The plan's rule -- no bare `eval`, a namespace
    of `t` plus whitelisted numpy functions, and reject anything containing
    `__` -- is necessary but **not sufficient**. Three expressions get
    through it with no `__` anywhere:

    - `9**9**9**9` hangs the process. Pure arithmetic; there is no string
      to match on, and no node type to forbid either -- every node in it
      is one an ordinary expression needs.
    - `(lambda: 1)()` is allowed. A lambda body is evaluated in the same
      restricted namespace, but the shape of the attack -- building and
      calling new code -- has no business here.
    - `t.real.conjugate()` is allowed, because attribute access is
      permitted. Nothing reachable through it is useful *today*; that is an
      accident of which objects the namespace happens to hold, not a
      property of the defence.

    So the defence here is two things. The first is an **AST whitelist**.
    The expression is parsed with `ast.parse(..., mode='eval')` and every
    node is checked against an explicit list (`_check_expression_node`):
    numbers, names that resolve in the namespace, the arithmetic
    operators, and calls to whitelisted functions by name. `ast.Attribute`
    and `ast.Lambda` are rejected outright, and any node type not on the
    list -- including syntax that does not exist yet -- is rejected by
    default. The `__` string check is kept as well, as belt and braces; it
    is not the defence.

    That disposes of the lambda and of the attribute chain, but not of
    `9**9**9**9`, which is made entirely of things the whitelist has to
    allow. So the second half of the defence is a **rewrite**: once the
    tree has been checked, `_FloatPower` turns every `**` in it into a
    call to `_pow`, which takes the power in floating point. A float is 64
    bits wide whatever is asked of it, so a runaway power overflows in
    microseconds instead of filling memory with an exact integer. See
    `_pow` for why this is done by rewriting the tree rather than by one
    more rule about its shape.

    Only then is the rewritten tree compiled and evaluated, with
    `__builtins__` emptied. The rewrite has to come *before* the compile,
    because the compiler folds constant arithmetic itself and would
    happily build the huge integer on our behalf.

    One thing is left over once `**` is a float, and it is dealt with
    first of all, before the expression is even parsed: the expression may
    be at most `_MAX_EXPRESSION_LENGTH` characters long. Multiplying long
    integer literals still grows an exact integer, and every walk over the
    tree recurses once per node, so both the arithmetic and the walker are
    bounded by bounding the length. See `_MAX_EXPRESSION_LENGTH`.
    """

    def __init__(self, expression):
        """
        Parameters
        ----------
        expression : str
            The expression, e.g. `"1 + 0.5*sin(2*pi*t/5400)"`.

        Raises
        ------
        ValueError
            If `expression` is not a string, is longer than
            `_MAX_EXPRESSION_LENGTH` characters, is not valid Python
            syntax, or uses anything outside the whitelist (see the class
            docstring).
        """

        if not isinstance(expression, str):
            raise ValueError(
                "an expression must be a string, got "
                f"{type(expression).__name__} ({expression!r}).")

        if len(expression) > _MAX_EXPRESSION_LENGTH:
            # Before `ast.parse`, because parsing is itself one of the two
            # things this bounds; see `_MAX_EXPRESSION_LENGTH`. The
            # expression is not repeated back, unlike every other
            # rejection here: it is by definition too long to read.
            raise ValueError(
                f"an expression may be at most {_MAX_EXPRESSION_LENGTH} "
                f"characters long, and this one is {len(expression)}. It is "
                f"not shown here, being far too long to read in an error "
                f"message.")

        if '__' in expression:
            # Belt and braces: everything this catches is already caught by
            # the AST whitelist below (a dunder can only be reached through
            # an attribute or a name, and both are checked), but it costs
            # nothing and it is the rule the plan asks for by name.
            _reject(expression, "it contains '__'.")

        try:
            tree = ast.parse(expression, mode = 'eval')
        except SyntaxError as err:
            raise ValueError(
                f"expression {expression!r} is not valid syntax "
                f"({err.msg}).") from err

        _check_expression_node(tree, expression)

        # Float-only `**` (see `_pow`), and before the compile, not after:
        # the compiler folds constant arithmetic itself, so a `**` still in
        # the tree here would be evaluated by `compile` below.
        tree = ast.fix_missing_locations(_FloatPower().visit(tree))

        self._expression = expression
        self._code = compile(tree, '<gammaraytoys config expression>', 'eval')

    @property
    def expression(self):
        """
        str: the expression this callable was built from.

        Kept so that a `FunctionScaling` built from a configuration can be
        written back out as one (`scaling_to_config`).
        """
        return self._expression

    def __call__(self, time):
        """
        Evaluate the expression at a given time.

        Parameters
        ----------
        time : `astropy.units.Quantity`
            The time to evaluate at (time units). Converted to seconds and
            handed to the expression as `t`.

        Returns
        -------
        float
            The expression's value. `FunctionScaling` is what checks it is
            finite and non-negative -- that check belongs to the scaling,
            not to the arithmetic.

        Raises
        ------
        TypeError
            If `time` is not a `Quantity` in time units.
        ValueError
            If evaluating raises (a division by zero, a domain error, a
            power that overflows a float, ...); the message names the
            expression and the time.
        """

        if not isinstance(time, u.Quantity):
            raise TypeError(
                f"a TimeExpression is evaluated at a Quantity in time units, "
                f"got {type(time).__name__} ({time!r}).")

        namespace = dict(_EXPRESSION_NAMESPACE)
        namespace['t'] = time.to_value(u.s)
        # The name `_FloatPower` rewrote every `**` into. It lives here and
        # not in `_EXPRESSION_NAMESPACE` so that the whitelist still refuses
        # an expression that tries to call it by hand.
        namespace[_POW_NAME] = _pow

        try:
            # Not a bare `eval`: `self._code` is the compiled form of a tree
            # every node of which was checked against the whitelist above,
            # no builtins are reachable, and the only names that resolve are
            # the ones in `namespace`.
            return eval(self._code, {'__builtins__': {}}, namespace)
        except Exception as err:
            raise ValueError(
                f"expression {self._expression!r} failed to evaluate at "
                f"time={time} ({type(err).__name__}: {err}).") from err

    def __repr__(self):
        """
        Returns
        -------
        str
            `TimeExpression('...')`.
        """
        return f"{type(self).__name__}({self._expression!r})"


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


def earth_from_config(config, where = 'earth'):
    """
    Build an `Earth` from its configuration block.

    ```yaml
    radius: 6371 km
    ```

    Parameters
    ----------
    config : mapping
        The `earth` block. An empty block is legal and gives the default
        `Earth()` -- astropy's nominal `R_earth`, which is 6378.1 km, not
        the 6371 km the plan's own sketch uses. Write the radius out.
    where : str
        Label for this block in error messages.

    Returns
    -------
    `Earth`

    Raises
    ------
    ValueError
        On an unknown key, or a radius that is not a positive length.
    """

    block = _as_mapping(config, where)
    _check_keys(block, where, ('radius',))

    radius = _quantity(block, 'radius', where, u.km)

    if radius is not None and radius <= 0 * radius.unit:
        raise ValueError(f"{where}: key 'radius' must be positive, got {radius}.")

    return Earth(radius = radius)


def earth_to_config(earth):
    """
    Write an `Earth` back out as a configuration block.

    Parameters
    ----------
    earth : `Earth`
        The Earth to describe.

    Returns
    -------
    dict
        `{'radius': ...}`. The radius is always written, including when the
        configuration that built this Earth left it out: which planet a run
        used is exactly the thing this project has already shipped a silent
        disagreement about.
    """

    return {'radius': _format_quantity(earth.radius)}


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------


#: Accepted spellings of every spectrum type, mapped to the canonical one.
#: Both the short name the plan's Section 7 sketch uses and the full class
#: name are accepted; the short one is what is written back out.
_SPECTRUM_TYPES = {'Monoenergetic': 'Monoenergetic',
                   'MonoenergeticSpectrum': 'Monoenergetic',
                   'PowerLaw': 'PowerLaw',
                   'PowerLawSpectrum': 'PowerLaw',
                   'MultiComponent': 'MultiComponent',
                   'MultiComponentSpectrum': 'MultiComponent'}


def spectrum_from_config(config, where = 'spectrum'):
    """
    Build a `Spectrum` from its configuration block.

    ```yaml
    {type: Monoenergetic, energy: 511 keV}
    {type: PowerLaw, index: -2, min_energy: 0.2 MeV, max_energy: 10 MeV}
    {type: MultiComponent, components: [...], weights: [1, 3]}
    ```

    Parameters
    ----------
    config : mapping
        The `spectrum` block.
    where : str
        Label for this block in error messages.

    Returns
    -------
    `Spectrum`

    Raises
    ------
    ValueError
        On an unknown type or key, a missing required key, a bad quantity,
        a non-positive `min_energy`, a `max_energy` that does not exceed
        it, or weights that do not match the components.
    """

    block = _as_mapping(config, where)
    name = _type_name(block, where, _SPECTRUM_TYPES, 'spectrum')

    if name == 'Monoenergetic':
        _check_keys(block, where, ('type', 'energy'), required = ('energy',))

        return MonoenergeticSpectrum(
            energy = _quantity(block, 'energy', where, u.keV, required = True))

    if name == 'PowerLaw':
        keys = ('type', 'index', 'min_energy', 'max_energy')
        _check_keys(block, where, keys, required = keys[1:])

        index = _number(block, 'index', where, required = True)
        min_energy = _quantity(block, 'min_energy', where, u.keV, required = True)
        max_energy = _quantity(block, 'max_energy', where, u.keV, required = True)

        # Both checked here rather than left to the sampler: a non-positive
        # or inverted range reaches `NumericalInverseHermite` as a log of
        # zero or a backwards domain, and what comes back is either a
        # failure from deep inside scipy or -- worse -- a table that
        # silently samples nonsense.
        if min_energy <= 0 * min_energy.unit:
            raise ValueError(
                f"{where}: key 'min_energy' must be positive, got {min_energy}.")

        if max_energy <= min_energy:
            raise ValueError(
                f"{where}: key 'max_energy' ({max_energy}) must be greater than "
                f"'min_energy' ({min_energy}).")

        return PowerLawSpectrum(index = index,
                                min_energy = min_energy,
                                max_energy = max_energy)

    keys = ('type', 'components', 'weights')
    _check_keys(block, where, keys, required = ('components',))

    blocks = block['components']

    if (not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes))
            or len(blocks) == 0):
        raise ValueError(
            f"{where}: key 'components' must be a non-empty list of spectrum "
            f"blocks, got {blocks!r}.")

    components = [spectrum_from_config(item, f"{where}.components[{i}]")
                  for i, item in enumerate(blocks)]

    weights = _number(block, 'weights', where, minimum = 0, shape = 'any')

    if weights is not None:
        if np.ndim(weights) == 0:
            weights = [weights]
        if len(weights) != len(components):
            raise ValueError(
                f"{where}: key 'weights' has {len(weights)} entries but there "
                f"are {len(components)} components.")
        if sum(weights) <= 0:
            raise ValueError(f"{where}: key 'weights' must not sum to zero.")

    return MultiComponentSpectrum(*components, weights = weights)


def spectrum_to_config(spectrum):
    """
    Write a `Spectrum` back out as a configuration block.

    Parameters
    ----------
    spectrum : `Spectrum`
        The spectrum to describe.

    Returns
    -------
    dict
        A block `spectrum_from_config` reads back into an equal spectrum.
        Note two canonicalizations: `PowerLawSpectrum` holds `max_energy`
        converted to `min_energy`'s unit, and `MultiComponentSpectrum`
        holds its weights normalized to sum to one (and equal weights are
        omitted, since that is the default).

    Raises
    ------
    ValueError
        If `spectrum` is not one of the three types a configuration can
        name.
    """

    if isinstance(spectrum, MonoenergeticSpectrum):
        return {'type': 'Monoenergetic',
                'energy': _format_quantity(spectrum.energy)}

    if isinstance(spectrum, PowerLawSpectrum):
        return {'type': 'PowerLaw',
                'index': float(spectrum.index),
                'min_energy': _format_quantity(spectrum.min_energy),
                'max_energy': _format_quantity(spectrum.max_energy)}

    if isinstance(spectrum, MultiComponentSpectrum):
        block = {'type': 'MultiComponent',
                 'components': [spectrum_to_config(component)
                                for component in spectrum.components]}

        weights = np.asarray(spectrum.weights, dtype = float)

        if not np.all(weights == weights[0]):
            block['weights'] = [float(weight) for weight in weights]

        return block

    raise ValueError(
        f"{type(spectrum).__name__} is not a spectrum a configuration can "
        f"describe; the types that are: "
        f"{sorted(set(_SPECTRUM_TYPES.values()))}.")


# ---------------------------------------------------------------------------
# Scalings
# ---------------------------------------------------------------------------


#: Accepted spellings of every scaling type, mapped to the canonical one.
_SCALING_TYPES = {'Constant': 'Constant',
                  'ConstantScaling': 'Constant',
                  'Tabulated': 'Tabulated',
                  'TabulatedScaling': 'Tabulated',
                  'Function': 'Function',
                  'FunctionScaling': 'Function'}


def scaling_from_config(config, where = 'scaling', base_dir = None):
    """
    Build a `SourceScaling` from its configuration block.

    ```yaml
    {type: Constant, scale: 1.0}
    {type: Tabulated, time: "[0, 100, 200] s", scale: [1.0, 2.0, 0.5]}
    {type: Tabulated, file: lightcurve.csv}
    {type: Function, expression: "1 + 0.5*sin(2*pi*t/5400)"}
    ```

    A tabulated scaling is given either inline (`time` and `scale`
    together) or as a two-column `time_s,scale` CSV `file`, never both.

    Parameters
    ----------
    config : mapping
        The `scaling` block.
    where : str
        Label for this block in error messages.
    base_dir : `pathlib.Path` or None
        The directory a relative `file` is taken relative to -- the
        directory of the configuration file, when there was one. `None`
        (the default, and what a configuration handed in as a mapping
        gets) leaves a relative path resolving against the working
        directory. An absolute `file` is unaffected either way.

    Returns
    -------
    `SourceScaling`

    Raises
    ------
    ValueError
        On an unknown type or key, a missing required key, a table given
        both ways or neither, a table the scaling itself rejects (unsorted
        times, a negative scale), or an expression outside the whitelist
        (see `TimeExpression`).
    """

    block = _as_mapping(config, where)
    name = _type_name(block, where, _SCALING_TYPES, 'scaling')

    if name == 'Constant':
        _check_keys(block, where, ('type', 'scale'))

        return ConstantScaling(
            scale = _number(block, 'scale', where, default = 1.0, minimum = 0))

    if name == 'Tabulated':
        _check_keys(block, where, ('type', 'time', 'scale', 'file'))

        has_file = 'file' in block
        has_inline = 'time' in block or 'scale' in block

        if has_file and has_inline:
            raise ValueError(
                f"{where}: a Tabulated scaling is given either as a 'file' or "
                f"as inline 'time' and 'scale', not both.")

        if has_file:
            filename = _text(block, 'file', where, required = True)
            path = _resolve_path(filename, base_dir)
            try:
                return TabulatedScaling.open(path)
            except FileNotFoundError as err:
                # Still a FileNotFoundError -- "the file is missing" is a
                # different problem from "the file is wrong", and a caller
                # may reasonably want to tell them apart. Only the message
                # changes: on its own it says nothing but the path, which in
                # a file with a dozen scalings in it does not say which one,
                # nor where it was looked for.
                raise FileNotFoundError(
                    f"{where}: key 'file' = {filename!r} does not exist "
                    f"(looked in {_searched_dir(path)}).") from err
            except Exception as err:
                raise ValueError(
                    f"{where}: could not read the table from {filename!r} "
                    f"({type(err).__name__}: {err}).") from err

        if not has_inline:
            raise ValueError(
                f"{where}: a Tabulated scaling needs either a 'file' or inline "
                f"'time' and 'scale'.")

        time = _quantity(block, 'time', where, u.s, required = True, shape = 'array')
        scale = _number(block, 'scale', where, required = True, minimum = 0,
                        shape = 'any')

        if np.ndim(scale) == 0:
            scale = [scale]

        try:
            return TabulatedScaling(time = time, scale = scale)
        except Exception as err:
            raise ValueError(f"{where}: {err}") from err

    _check_keys(block, where, ('type', 'expression'), required = ('expression',))

    expression = _text(block, 'expression', where, required = True)

    try:
        return FunctionScaling(TimeExpression(expression))
    except ValueError as err:
        raise ValueError(f"{where}: {err}") from err


def scaling_to_config(scaling):
    """
    Write a `SourceScaling` back out as a configuration block.

    Parameters
    ----------
    scaling : `SourceScaling`
        The scaling to describe.

    Returns
    -------
    dict
        A block `scaling_from_config` reads back into an equal scaling. A
        `TabulatedScaling` is always written inline, including when it was
        read from a file: the table is data, and inlining it keeps the
        written configuration self-contained.

    Raises
    ------
    ValueError
        If `scaling` is not one of the three types a configuration can
        name, or if it is a `FunctionScaling` wrapping a callable that did
        not come from a configuration expression -- an arbitrary Python
        function has no expression to write.
    """

    if isinstance(scaling, ConstantScaling):
        return {'type': 'Constant', 'scale': float(scaling.scale)}

    if isinstance(scaling, TabulatedScaling):
        return {'type': 'Tabulated',
                'time': _format_quantity(scaling.time),
                'scale': [float(item) for item in scaling.scale]}

    if isinstance(scaling, FunctionScaling):
        if not isinstance(scaling.function, TimeExpression):
            raise ValueError(
                "this FunctionScaling wraps "
                f"{type(scaling.function).__name__}, not an expression built "
                "from a configuration, so there is no expression to write "
                "back out.")

        return {'type': 'Function', 'expression': scaling.function.expression}

    raise ValueError(
        f"{type(scaling).__name__} is not a scaling a configuration can "
        f"describe; the types that are: "
        f"{sorted(set(_SCALING_TYPES.values()))}.")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


#: Accepted spellings of every source type. Unlike spectra and scalings,
#: sources are named by their class name in the plan's own sketch, and that
#: is the only spelling accepted.
_SOURCE_TYPES = {'PointSource': 'PointSource',
                 'IsotropicSource': 'IsotropicSource',
                 'NearPointSource': 'NearPointSource',
                 'ExtendedSource': 'ExtendedSource',
                 'EarthAlbedoSource': 'EarthAlbedoSource'}

#: Keys every source block may carry, whatever its type.
_COMMON_SOURCE_KEYS = ('name', 'type', 'spectrum', 'scaling',
                       'chirality', 'chirality_degree')

#: Extra keys, by source type.
_SOURCE_KEYS = {'PointSource': ('offaxis_angle', 'sky_angle', 'flux',
                                'flux_pivot', 'pivot_energy'),
                'IsotropicSource': ('flux',),
                'NearPointSource': ('position', 'rate'),
                'ExtendedSource': ('sky_angle', 'width', 'flux'),
                'EarthAlbedoSource': ('emissivity', 'law', 'emission_law')}


def _position_from_config(block, where):
    """
    Read a near-field source's `position` block into a `Cartesian2D`.

    ```yaml
    position: {x: 0 cm, y: 1 cm}
    ```

    Parameters
    ----------
    block : dict
        The source block, which must carry a `position` key.
    where : str
        Label for the source block in error messages.

    Returns
    -------
    `Cartesian2D`

    Raises
    ------
    ValueError
        If `position` is missing, is not a block of `x` and `y`, or if
        either coordinate is not a length.
    """

    if 'position' not in block:
        raise ValueError(f"{where}: missing required key 'position'.")

    inner_where = f"{where}.position"
    inner = _as_mapping(block['position'], inner_where)
    _check_keys(inner, inner_where, ('x', 'y'), required = ('x', 'y'))

    return Cartesian2D(_quantity(inner, 'x', inner_where, u.cm, required = True),
                       _quantity(inner, 'y', inner_where, u.cm, required = True))


def _common_source_kwargs(block, where, base_dir = None):
    """
    Read the constructor arguments every source type shares.

    Parameters
    ----------
    block : dict
        The source block.
    where : str
        Label for this block in error messages.
    base_dir : `pathlib.Path` or None
        The directory a relative path inside the `scaling` block resolves
        against. See `scaling_from_config`.

    Returns
    -------
    dict
        `spectrum`, `scaling`, `chirality` and `chirality_degree`, ready to
        be passed to any source constructor.

    Raises
    ------
    ValueError
        If `spectrum` is missing or bad, if the scaling is bad, or if
        `chirality` is not `+1`/`-1` or `chirality_degree` is outside
        `[0, 1]`.
    """

    if 'spectrum' not in block:
        raise ValueError(f"{where}: missing required key 'spectrum'.")

    spectrum = spectrum_from_config(block['spectrum'], f"{where}.spectrum")

    scaling = None
    if 'scaling' in block:
        scaling = scaling_from_config(block['scaling'], f"{where}.scaling",
                                      base_dir = base_dir)

    chirality = _integer(block, 'chirality', where)

    if chirality is not None and chirality not in (-1, 1):
        raise ValueError(
            f"{where}: key 'chirality' must be -1 or +1 (or left out, for no "
            f"chirality preference), got {chirality}.")

    return {'spectrum': spectrum,
            'scaling': scaling,
            'chirality': chirality,
            'chirality_degree': _number(block, 'chirality_degree', where,
                                        default = 0, minimum = 0, maximum = 1)}


def source_from_config(config, where = 'source', earth = None, base_dir = None):
    """
    Build a `Source` from its configuration block.

    ```yaml
    name: crab                  # optional; labels this source's events
    type: PointSource
    sky_angle: 45 deg           # ... or offaxis_angle, never both
    flux: 1e-3 1/(cm s)
    spectrum: {type: PowerLaw, index: -2, min_energy: 0.2 MeV, max_energy: 10 MeV}
    scaling: {type: Function, expression: "1 + 0.5*sin(2*pi*t/5400)"}
    ```

    The five types and the keys each adds to the common ones (`name`,
    `type`, `spectrum`, `scaling`, `chirality`, `chirality_degree`):

    - `PointSource`: `offaxis_angle` **or** `sky_angle` -- exactly one, the
      first putting the source at a fixed detector-frame angle and the
      second on the inertial sky -- plus `flux`, or `flux_pivot` and
      `pivot_energy` together, never both normalisations and never half
      of the pivot pair. All three may be left out, which is a source
      with no flux: it can be drawn from but not counted (see
      `PointSource`).
    - `IsotropicSource`: `flux`.
    - `NearPointSource`: `position` (a block of `x` and `y`) and `rate`.
    - `ExtendedSource`: `sky_angle`, `width` and `flux`.
    - `EarthAlbedoSource`: `emissivity` and `law`
      (`lambertian` or `isotropic`).

    Parameters
    ----------
    config : mapping
        The source block.
    where : str
        Label for this block in error messages.
    earth : `Earth` or None
        The Earth an `EarthAlbedoSource` emits from. `None` leaves the
        source to build its own default `Earth()` -- which is astropy's
        6378.1 km, *not* the 6371 km a configuration typically names, so
        the top-level loader always passes the run's single Earth here
        rather than letting a run end up with two different planets.
    base_dir : `pathlib.Path` or None
        The directory a relative path inside this source's `scaling` block
        resolves against. See `scaling_from_config`.

    Returns
    -------
    `Source`

    Raises
    ------
    ValueError
        On an unknown type or key, a missing required key, a bad value, or
        anything the source class itself rejects -- including a
        `PointSource` given both `offaxis_angle` and `sky_angle` or
        neither, whose own error is surfaced rather than replaced. Also
        on a `PointSource` given `flux` beside the pivot pair, or only
        one half of that pair, or a pivot pair whose `pivot_energy` lands
        where the spectrum has zero probability density, which would
        otherwise resolve to an infinite `flux`.
    """

    block = _as_mapping(config, where)
    name = _type_name(block, where, _SOURCE_TYPES, 'source')

    allowed = _COMMON_SOURCE_KEYS + _SOURCE_KEYS[name]
    _check_keys(block, where, allowed)

    kwargs = _common_source_kwargs(block, where, base_dir)

    if name == 'PointSource':
        if 'flux' in block and ('flux_pivot' in block or 'pivot_energy' in block):
            raise ValueError(
                f"{where}: a PointSource's normalisation is given either as "
                f"'flux' or as 'flux_pivot' with 'pivot_energy', not both. "
                f"`PointSource` prefers 'flux' and drops the pivot pair "
                f"without saying so, and 'to_config' then writes the file "
                f"back out with only the 'flux' in it.")

        # The same silent wrong answer one step along: `flux_pivot` and
        # `pivot_energy` are two halves of one number, and `PointSource`
        # uses neither of them without the other. Given only one it leaves
        # the flux unset rather than complaining, so the run draws from an
        # unnormalised source and 'to_config' writes the file back out
        # with the lone key gone.
        if ('flux_pivot' in block) != ('pivot_energy' in block):
            given = 'flux_pivot' if 'flux_pivot' in block else 'pivot_energy'
            missing = 'pivot_energy' if given == 'flux_pivot' else 'flux_pivot'
            raise ValueError(
                f"{where}: a PointSource given {given!r} needs {missing!r} "
                f"as well -- the two are halves of one normalisation, the "
                f"differential flux and the energy it is quoted at, and "
                f"`PointSource` uses neither without the other. Give both, "
                f"or give 'flux' instead.")

        # Both are read and both are passed on, even when one (or neither)
        # is there: `PointSource` itself enforces "exactly one of the two",
        # and its message is better than anything invented here.
        kwargs['offaxis_angle'] = _quantity(block, 'offaxis_angle', where, u.deg)
        kwargs['sky_angle'] = _quantity(block, 'sky_angle', where, u.deg)
        kwargs['flux'] = _quantity(block, 'flux', where, u.Unit('1 / (cm s)'),
                                   minimum = 0)
        kwargs['flux_pivot'] = _quantity(block, 'flux_pivot', where,
                                         u.Unit('1 / (cm s keV)'), minimum = 0)
        kwargs['pivot_energy'] = _quantity(block, 'pivot_energy', where, u.keV,
                                           minimum = 0)

        # A third silent wrong answer, one step past the pair being
        # complete: `PointSource` divides `flux_pivot` by the spectrum's
        # probability density at `pivot_energy`, and that density is
        # exactly zero outside the spectrum's own energy range. The
        # division still "succeeds" -- it just returns infinity, with
        # nothing louder than a numpy warning that is easy to have
        # suppressed -- and 'to_config' then writes `flux: inf ...` back
        # into the file as if it were a deliberate value.
        if kwargs['flux'] is None and kwargs['flux_pivot'] is not None:
            spectrum = kwargs['spectrum']
            pivot_energy = kwargs['pivot_energy']
            pivot_density = spectrum.pdf(pivot_energy)
            zero_density_error = ValueError(
                f"{where}: key 'pivot_energy' = {pivot_energy} has zero "
                f"probability density on this spectrum, so there is no "
                f"total flux that 'flux_pivot' could correspond to. The "
                f"spectrum's energy range is "
                f"[{spectrum.min_energy}, {spectrum.max_energy}]; "
                f"'pivot_energy' ordinarily needs to fall inside it "
                f"(a `MultiComponentSpectrum` can still have zero "
                f"density inside its overall range, in a gap none of "
                f"its components cover).")

            # Check the density itself before dividing by it, rather than
            # dividing and then checking whether the result came out
            # infinite: a zero density is exactly the condition being
            # guarded against, and dividing by it first only earns a numpy
            # RuntimeWarning on the way to the same error raised here.
            if pivot_density == 0:
                raise zero_density_error

            resolved_flux = (kwargs['flux_pivot'] / pivot_density).to(u.Unit('1 / (cm s)'))

            # Belt and braces: the zero-density check above is the only way
            # this division was going wrong, but keep this in case some
            # other route (a denormal density, say) still produces a
            # non-finite flux.
            if not np.isfinite(resolved_flux):
                raise zero_density_error

        source_class = PointSource

    elif name == 'IsotropicSource':
        kwargs['flux'] = _quantity(block, 'flux', where, u.Unit('1 / (cm s)'),
                                   minimum = 0)
        source_class = IsotropicSource

    elif name == 'NearPointSource':
        kwargs['position'] = _position_from_config(block, where)
        kwargs['rate'] = _quantity(block, 'rate', where, u.Unit('1 / s'),
                                   minimum = 0)
        source_class = NearPointSource

    elif name == 'ExtendedSource':
        kwargs['sky_angle'] = _quantity(block, 'sky_angle', where, u.deg,
                                        required = True)
        # No `minimum` on `width`: `ExtendedSource` demands a *strictly*
        # positive one and says so much better than a bound here could
        # ("for a source at a single exact direction use PointSource"), and
        # its message already arrives with this block's `where` in front.
        # The same goes for `EarthAlbedoSource`'s `emissivity` below.
        kwargs['width'] = _quantity(block, 'width', where, u.deg, required = True)
        kwargs['flux'] = _quantity(block, 'flux', where, u.Unit('1 / (cm s)'),
                                   minimum = 0)
        source_class = ExtendedSource

    else:
        if 'law' in block and 'emission_law' in block:
            raise ValueError(
                f"{where}: give either 'law' or 'emission_law', not both -- "
                f"they are the same key. ('emission_law' is the spelling in "
                f"the plan's Section 7 sketch; 'law' is the constructor "
                f"argument and the canonical name here.)")

        law_key = 'emission_law' if 'emission_law' in block else 'law'

        kwargs['emissivity'] = _quantity(block, 'emissivity', where,
                                         u.Unit('1 / (cm s)'), required = True)
        kwargs['law'] = _text(block, law_key, where, default = 'lambertian',
                              choices = ('lambertian', 'isotropic'))
        kwargs['earth'] = earth
        source_class = EarthAlbedoSource

    try:
        return source_class(**kwargs)
    except Exception as err:
        raise ValueError(f"{where}: {err}") from err


def source_to_config(source, name = None):
    """
    Write a `Source` back out as a configuration block.

    Parameters
    ----------
    source : `Source`
        The source to describe.
    name : str or None
        The source's name, written as the block's `name` key when given.

    Returns
    -------
    dict
        A block `source_from_config` reads back into an equal source. Keys
        left at their default -- no chirality, `chirality_degree` of 0, no
        scaling, an unset flux or rate -- are left out, and a `PointSource`
        given `flux_pivot`/`pivot_energy` comes back as the `flux` those
        two resolved to.

    Raises
    ------
    ValueError
        If `source` is not one of the five types a configuration can name.
    """

    block = {}

    if name is not None:
        block['name'] = name

    block['type'] = type(source).__name__

    if block['type'] not in _SOURCE_TYPES:
        raise ValueError(
            f"{type(source).__name__} is not a source a configuration can "
            f"describe; the types that are: "
            f"{sorted(set(_SOURCE_TYPES.values()))}.")

    if isinstance(source, PointSource):
        if source.sky_angle is not None:
            block['sky_angle'] = _format_quantity(source.sky_angle)
        else:
            block['offaxis_angle'] = _format_quantity(source.offaxis_angle)

    elif isinstance(source, ExtendedSource):
        block['sky_angle'] = _format_quantity(source.sky_angle)
        block['width'] = _format_quantity(source.width)

    elif isinstance(source, NearPointSource):
        block['position'] = {'x': _format_quantity(source.position.x),
                             'y': _format_quantity(source.position.y)}

    elif isinstance(source, EarthAlbedoSource):
        block['emissivity'] = _format_quantity(source.emissivity)
        if source.law != 'lambertian':
            block['law'] = source.law

    if isinstance(source, NearPointSource):
        if source.rate is not None:
            block['rate'] = _format_quantity(source.rate)
    elif not isinstance(source, EarthAlbedoSource):
        # Every other far-field source carries an optional sky-integrated
        # flux; the albedo carries an emissivity instead, and its flux is a
        # function of the orbit rather than a free parameter.
        flux = source.flux()
        if flux is not None:
            block['flux'] = _format_quantity(flux)

    block['spectrum'] = spectrum_to_config(source.spectrum)

    scaling = source.scaling

    if not (isinstance(scaling, ConstantScaling) and scaling.scale == 1.0):
        block['scaling'] = scaling_to_config(scaling)

    if source.chirality is not None:
        block['chirality'] = int(source.chirality)

    if source.chirality_degree != 0:
        block['chirality_degree'] = float(source.chirality_degree)

    return block


# ---------------------------------------------------------------------------
# Observation strategies, spacecraft history, reconstructor
# ---------------------------------------------------------------------------


#: Accepted spellings of every observation strategy type.
_STRATEGY_TYPES = {'ZenithPointing': 'ZenithPointing',
                   'NadirPointing': 'NadirPointing',
                   'InertialPointing': 'InertialPointing',
                   'SpinPointing': 'SpinPointing',
                   'TargetedPointing': 'TargetedPointing'}


def observation_strategy_from_config(config, where = 'observation_strategy',
                                     earth = None):
    """
    Build an `ObservationStrategy` from its configuration block.

    ```yaml
    {type: ZenithPointing}
    {type: NadirPointing}
    {type: InertialPointing, attitude: 30 deg}
    {type: SpinPointing, rate: 0.1 deg/s, initial_attitude: 0 deg}
    {type: TargetedPointing, sky_angle: 45 deg}
    ```

    Parameters
    ----------
    config : mapping
        The `observation_strategy` block.
    where : str
        Label for this block in error messages.
    earth : `Earth` or None
        The Earth a `TargetedPointing` decides occultation against.
        Required for that strategy and refused as `None`: it is the one
        strategy whose whole job is deciding when the target is behind the
        Earth, and letting it build its own would be exactly the
        two-different-planets bug this codebase has already shipped once.

    Returns
    -------
    `ObservationStrategy`

    Raises
    ------
    ValueError
        On an unknown type or key, a missing required key, a bad quantity,
        or a `TargetedPointing` with no Earth to point around.
    """

    block = _as_mapping(config, where)
    name = _type_name(block, where, _STRATEGY_TYPES, 'observation strategy')

    if name == 'ZenithPointing':
        _check_keys(block, where, ('type',))
        return ZenithPointing()

    if name == 'NadirPointing':
        _check_keys(block, where, ('type',))
        return NadirPointing()

    if name == 'InertialPointing':
        _check_keys(block, where, ('type', 'attitude'), required = ('attitude',))
        return InertialPointing(
            attitude = _quantity(block, 'attitude', where, u.deg, required = True))

    if name == 'SpinPointing':
        _check_keys(block, where, ('type', 'rate', 'initial_attitude'),
                    required = ('rate',))
        return SpinPointing(
            rate = _quantity(block, 'rate', where, u.deg / u.s, required = True),
            initial_attitude = _quantity(block, 'initial_attitude', where, u.deg,
                                         default = 0 * u.deg))

    _check_keys(block, where, ('type', 'sky_angle'), required = ('sky_angle',))

    if earth is None:
        raise ValueError(
            f"{where}: a TargetedPointing needs the Earth it decides "
            f"occultation against, and must not build its own -- pass the "
            f"run's Earth (the top-level loader always does).")

    return TargetedPointing(
        sky_angle = _quantity(block, 'sky_angle', where, u.deg, required = True),
        earth = earth)


def observation_strategy_to_config(strategy):
    """
    Write an `ObservationStrategy` back out as a configuration block.

    Parameters
    ----------
    strategy : `ObservationStrategy`
        The strategy to describe.

    Returns
    -------
    dict
        A block `observation_strategy_from_config` reads back into an equal
        strategy. A `TargetedPointing`'s Earth is not written here: a
        configuration has exactly one `earth` block, and the loader hands
        it to this strategy.

    Raises
    ------
    ValueError
        If `strategy` is not one of the five types a configuration can
        name.
    """

    if isinstance(strategy, ZenithPointing):
        return {'type': 'ZenithPointing'}

    if isinstance(strategy, NadirPointing):
        return {'type': 'NadirPointing'}

    if isinstance(strategy, InertialPointing):
        return {'type': 'InertialPointing',
                'attitude': _format_quantity(strategy.attitude)}

    if isinstance(strategy, SpinPointing):
        block = {'type': 'SpinPointing',
                 'rate': _format_quantity(strategy.rate)}

        if strategy.initial_attitude != 0 * u.deg:
            block['initial_attitude'] = _format_quantity(strategy.initial_attitude)

        return block

    if isinstance(strategy, TargetedPointing):
        return {'type': 'TargetedPointing',
                'sky_angle': _format_quantity(strategy.sky_angle)}

    raise ValueError(
        f"{type(strategy).__name__} is not an observation strategy a "
        f"configuration can describe; the types that are: "
        f"{sorted(set(_STRATEGY_TYPES.values()))}.")


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
        strategy = observation_strategy_from_config(
            block['observation_strategy'], f"{where}.observation_strategy", earth)
        kwargs['observation_strategy'] = strategy
        canonical['observation_strategy'] = observation_strategy_to_config(strategy)

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
        resolves against. See `scaling_from_config`.

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
        on anything `source_from_config` rejects.
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

        source = source_from_config(block, label, earth, base_dir)
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
    earth = (earth_from_config(block['earth'], f"{where}.earth")
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
        config['earth'] = earth_to_config(earth)

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

    config['sources'] = [source_to_config(source, names.get(source))
                         for source in simulator.sources]

    return config
