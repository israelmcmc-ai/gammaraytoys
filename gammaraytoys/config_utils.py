"""
The validation and parsing primitives the YAML configuration layer is built
from (`docs/dev/inertial_sim_plan.md`, Section 7).

Every kind a configuration can name -- a spectrum, a source, a scaling, a
spacecraft history -- reads and writes itself, through a `from_config`
classmethod and a `to_config` method on its own class. All of them need the
same handful of primitives: read a unit-bearing string, reject an unknown
key, resolve a `type` name, format a `Quantity` so it reads back exactly.

Those primitives live here, at the root of the package, in a module that
imports nothing from `gammaraytoys` itself. The classes that read and write
themselves are spread across the package -- `sims/source.py`,
`sims/spectrum.py`, `sims/scaling.py`, `detectors/tracker/tracker_2d.py`,
`coordinates/twodim.py` -- so the helpers they share cannot live inside any
one subpackage without the others reaching sideways into it. They cannot
live in `config.py` either, which imports *them*, to wire a whole run
together. That is the whole reason this file exists and the reason it sits
this high up: it is the bottom of the layering, and keeping it there is what
stops the imports going in a circle.

Every helper here takes the `where` of the block it is validating -- a
human-readable path into the file, like "sources[1] (albedo).spectrum" --
and puts it in front of every message. Astropy's own errors say what could
not be parsed but never where it came from, which in a configuration file is
the only thing the user needs to know.
"""

import ast
import difflib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import astropy.units as u


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
        # calls. Nothing in this module ever reaches for bare `eval`; see
        # "Why there is no expression syntax" above.
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


def _dispatch_type_name(cls, base, block, where, table, classes, kind):
    """
    Resolve a block's `type` key for a `from_config` inherited by subclasses.

    A hierarchy's `from_config` is written once, on its base class, and
    reads `type` to decide which subclass to build: `Source.from_config`
    hands back a `PointSource` or an `IsotropicSource` as the block says.
    Every subclass inherits that method, though, so `PointSource.from_config`
    would read the same key and hand back an `IsotropicSource` whenever the
    block happened to say so -- a trap, because the caller already named the
    class they wanted.

    So the class the method was called on has the last word. Called on the
    base (or on an intermediate one, like `FarFieldSource`), `type` chooses
    freely among the classes below it. Called on a concrete class, `type`
    may name that class or be left out altogether, and naming anything else
    is an error rather than a silent substitution.

    Parameters
    ----------
    cls : type
        The class `from_config` was called on.
    base : type
        The class `from_config` is defined on, named in the error message as
        the place to call it when the block's `type` should choose.
    block : dict
        The block to read from.
    where : str
        Human-readable path to `block`, used in error messages.
    table : dict
        Maps every accepted spelling (including aliases) to the canonical
        type name. See `_type_name`.
    classes : dict
        Maps each canonical type name to the class that builds it.
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
        If `type` is missing (and `cls` is not itself one of `classes`), is
        not a string, is not a known type, or names a class that is not
        `cls` or a subclass of it.
    """

    # `classes` runs canonical name -> class; this runs it back the other
    # way, to ask "is the class we were called on one a `type` could name?"
    named = {klass: name for name, klass in classes.items()}

    if 'type' not in block and cls in named:
        # A concrete class asked to build itself: the key is redundant, and
        # `{type: PointSource}` inside `PointSource.from_config` is noise.
        return named[cls]

    name = _type_name(block, where, table, kind)

    if not issubclass(classes[name], cls):
        raise ValueError(
            f"{where}: key 'type' = {name!r} names a "
            f"{classes[name].__name__}, which is not a {cls.__name__}. "
            f"`{cls.__name__}.from_config` builds a {cls.__name__} and "
            f"nothing else; call `{base.__name__}.from_config` to let the "
            f"block's 'type' choose the class.")

    return name
