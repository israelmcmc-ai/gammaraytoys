"""
Time-dependent source scaling (`docs/dev/inertial_sim_plan.md`, Section 5.7).

A `SourceScaling` is a unitless multiplier on a source's normalization
(flux or rate), evaluated by `InertialSimulator` once per interval, at that
interval's midpoint (`SpacecraftInterval.mid_time`) -- see Section 6's
per-interval loop. It is deliberately not itself a physical quantity: it
multiplies whatever normalization the source already has, so a scaling of
`2.0` means "twice as bright right now", independent of what "bright" means
for that source's family.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import astropy.units as u


__all__ = ['SourceScaling', 'ConstantScaling', 'TabulatedScaling', 'FunctionScaling']


def _validate_scale(value, what):
    """
    Check that a scaling value is a finite, non-negative real number.

    A negative or non-finite scale would make a negative or non-finite
    Poisson mean inside `InertialSimulator.run_events`, which either raises
    deep inside `numpy.random.poisson` or (for a NaN) fails silently with a
    NaN count -- far from where the bad value was actually produced. Every
    `SourceScaling` validates at the point the value is set or produced,
    not where it is later consumed.

    Parameters
    ----------
    value : float
        The candidate scale.
    what : str
        Short description of where this value came from, for the error
        message (e.g. "ConstantScaling's scale", "row 3 of the table").

    Returns
    -------
    float
        `value`, unchanged.

    Raises
    ------
    ValueError
        If `value` is not finite or is negative.
    """

    if not np.isfinite(value) or value < 0:
        raise ValueError(
            f"{what} must be a finite, non-negative number; got {value!r}.")

    return value


class SourceScaling(ABC):
    """
    Abstract base class for a unitless, time-dependent multiplier on a
    source's normalization.

    `InertialSimulator` calls `scaling(interval.mid_time)` once per
    (source, interval) and multiplies it into the Poisson mean, alongside
    `simulated_rate(detector, pose)` and the interval's livetime -- exactly
    the product in Section 6 of the plan. Every concrete source accepts a
    `scaling` constructor argument, defaulting to `ConstantScaling(1.0)`
    (no scaling at all), so existing code that never passes one behaves
    exactly as before.
    """

    @abstractmethod
    def __call__(self, time):
        """
        The scale factor at a given time.

        Parameters
        ----------
        time : `astropy.units.Quantity`
            The time to evaluate the scaling at (time units). In practice
            this is always an interval's midpoint,
            `SpacecraftInterval.mid_time`.

        Returns
        -------
        float
            A finite, non-negative, unitless multiplier.
        """
        pass


class ConstantScaling(SourceScaling):
    """
    A scaling that is the same at every time. `ConstantScaling(1.0)` is the
    default every source uses when no `scaling` is given, i.e. "not scaled
    at all".
    """

    def __init__(self, scale = 1.0):
        """
        Parameters
        ----------
        scale : float
            The constant scale factor. Must be a finite, non-negative
            number. Defaults to 1.0.

        Raises
        ------
        ValueError
            If `scale` is not finite or is negative.
        """

        self.scale = scale

    @property
    def scale(self):
        """
        float: the constant scale factor.

        Validated on assignment, not only at construction: an unchecked
        `constant.scale = -3.0` would otherwise surface far away, as
        `ValueError: lam < 0 or lam is NaN` from `numpy.random.poisson`
        deep inside a run.

        Returns
        -------
        float
        """
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = _validate_scale(float(scale), "ConstantScaling's scale")

    def __call__(self, time):
        """
        The scale factor: always `self.scale`, regardless of `time`.

        Parameters
        ----------
        time : `astropy.units.Quantity`
            Ignored.

        Returns
        -------
        float
            `self.scale`.
        """
        return self.scale


class TabulatedScaling(SourceScaling):
    """
    A piecewise-constant scaling read from a table of `(time, scale)`
    breakpoints, matching the `.ori` file's own interval semantics
    (`docs/dev/inertial_sim_plan.md`, Section 4.2): the value at time `t` is
    the scale of **the last row whose time is `<= t`**.

    Concretely, for breakpoints `t_0 < t_1 < ... < t_{n-1}`:

    - a breakpoint belongs to the row **at** it, not the previous one --
      `scaling(t_i)` is `scale[i]`, not `scale[i-1]`;
    - the value is flat on each `[t_i, t_{i+1})` -- right-continuous at
      `t_i`, left-discontinuous;
    - outside the table it **clamps**, it never extrapolates:
      `scaling(t)` is `scale[0]` for any `t < t_0`, and `scale[-1]` for any
      `t >= t_{n-1}`.

    A single-row table is therefore a constant scaling, everywhere.
    """

    def __init__(self, time, scale):
        """
        Parameters
        ----------
        time : `astropy.units.Quantity`
            Breakpoint times `t_0 < t_1 < ... < t_{n-1}` (time units),
            strictly increasing. At least one row is required.
        scale : array-like of float
            The scale at (and after) each breakpoint, same length as
            `time`. Every value must be finite and non-negative.

        Raises
        ------
        ValueError
            If the table is empty, if `time` and `scale` have different
            lengths, if `time` is not strictly increasing (this also
            catches duplicate times), or if any `scale` is not finite and
            non-negative. Accepting an unsorted or duplicated table
            silently would make the `<=` search below return nonsense.
        """

        time_s = np.asarray(time.to_value(u.s), dtype = float)
        scale = np.asarray(scale, dtype = float)

        if time_s.size == 0:
            raise ValueError("TabulatedScaling needs a non-empty table; got 0 rows.")

        if time_s.shape != scale.shape:
            raise ValueError(
                f"TabulatedScaling's time and scale must have the same length; "
                f"got {time_s.size} times and {scale.size} scales.")

        if not np.all(np.isfinite(time_s)):
            raise ValueError("TabulatedScaling's times must all be finite.")

        if time_s.size > 1 and not np.all(np.diff(time_s) > 0):
            raise ValueError(
                "TabulatedScaling's times must be strictly increasing (this also "
                "forbids duplicate times); a table that is unsorted or has "
                "repeated times would make the piecewise-constant lookup return "
                "nonsense.")

        for i, s in enumerate(scale):
            _validate_scale(float(s), f"TabulatedScaling row {i}'s scale")

        self._time_s = time_s
        self._scale = scale

    @classmethod
    def open(cls, filename):
        """
        Read a `TabulatedScaling` from a two-column CSV file.

        Parameters
        ----------
        filename : str or path-like
            Path to a CSV file with columns `time_s,scale`, one header
            line.

        Returns
        -------
        TabulatedScaling

        Raises
        ------
        ValueError
            If the file is missing either column, or any of the checks in
            `__init__` fails.
        """

        df = pd.read_csv(filename, float_precision = 'round_trip')

        expected_cols = {'time_s', 'scale'}
        missing_cols = expected_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"{filename}: missing column(s) {sorted(missing_cols)}; a "
                f"TabulatedScaling file needs both of {sorted(expected_cols)}.")

        return cls(time = df['time_s'].to_numpy(dtype = float) * u.s,
                   scale = df['scale'].to_numpy(dtype = float))

    def __call__(self, time):
        """
        The piecewise-constant scale factor at `time` (see the class
        docstring for the exact semantics).

        Parameters
        ----------
        time : `astropy.units.Quantity`
            The time to evaluate (time units). Scalar.

        Returns
        -------
        float
            The scale of the last breakpoint at or before `time`, clamped
            to the table's first value before it starts and its last value
            after it ends.
        """

        t = time.to_value(u.s)

        # `side='right'` puts the insertion point just past every entry
        # equal to `t`, so subtracting 1 lands exactly on the row *at* `t`
        # when `t` is itself a breakpoint (not the row before it), and on
        # the last row `<= t` otherwise. Clip handles both clamped ends:
        # `t` before the first breakpoint gives index -1, clipped to 0;
        # `t` at or after the last gives the last index already.
        idx = np.searchsorted(self._time_s, t, side = 'right') - 1
        idx = int(np.clip(idx, 0, self._time_s.size - 1))

        return float(self._scale[idx])


class FunctionScaling(SourceScaling):
    """
    A scaling that wraps an arbitrary callable of time.
    """

    def __init__(self, function):
        """
        Parameters
        ----------
        function : callable
            `function(time) -> float`, where `time` is an
            `astropy.units.Quantity` (time units) and the return value is a
            finite, non-negative real number. Validated on every call, not
            at construction (the function itself is not evaluated here).
        """

        self.function = function

    def __call__(self, time):
        """
        The scale factor at `time`: `self.function(time)`, validated.

        Parameters
        ----------
        time : `astropy.units.Quantity`
            The time to evaluate (time units).

        Returns
        -------
        float
            `self.function(time)`, as a finite, non-negative `float`.

        Raises
        ------
        ValueError
            If `self.function(time)` is not convertible to `float`, or is
            not finite and non-negative. A negative scaling would give a
            negative Poisson mean and blow up deep inside the run, far from
            this callable; catching it here names the actual offender.
        """

        value = self.function(time)

        try:
            value = float(value)
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"FunctionScaling's callable must return a real number; at "
                f"time={time} it returned {value!r} ({type(value).__name__}), "
                f"which is not convertible to float.") from err

        return _validate_scale(value, f"FunctionScaling's callable, at time={time},")
