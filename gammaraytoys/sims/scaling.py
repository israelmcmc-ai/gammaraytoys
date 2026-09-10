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


__all__ = ['SourceScaling', 'ConstantScaling', 'TabulatedScaling',
           'BurstScaling', 'SinusoidalScaling']


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

    @property
    def time(self):
        """
        `astropy.units.Quantity`: the breakpoint times, in seconds.

        A copy, not the stored array: the lookup in `__call__` assumes the
        times are still strictly increasing, and handing out the array
        itself would let a caller break that assumption in place, long
        after `__init__` validated it.

        Returns
        -------
        `astropy.units.Quantity`
        """
        return self._time_s.copy() * u.s

    @property
    def scale(self):
        """
        numpy.ndarray: the scale at (and after) each breakpoint.

        A copy, for the same reason as `time`: every value was validated
        finite and non-negative at construction.

        Returns
        -------
        numpy.ndarray
        """
        return self._scale.copy()

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


class BurstScaling(SourceScaling):
    """
    A scaling that is zero everywhere except inside a single window: a
    burst that switches on at `start`, holds `amplitude` for `duration`,
    and is `0.0` before and after. This is the shape a transient has in a
    teaching run -- a gamma-ray burst against a quiet background.

    The window is **half-open**, `start <= t < start + duration`: the
    instant the burst starts belongs to the burst, and the instant it ends
    belongs to what comes after. That is not an arbitrary choice.
    `TabulatedScaling` above is already right-continuous -- a breakpoint
    belongs to the row *at* it -- and two scalings in the same package
    disagreeing about which side of an edge a time falls on would be a
    trap: a burst and a table describing the same lightcurve would differ
    by one interval, at exactly the times a user is most likely to check
    by hand.
    """

    def __init__(self, start, duration, amplitude = 1.0):
        """
        Parameters
        ----------
        start : `astropy.units.Quantity`
            The time the burst switches on (time units). Any finite time,
            including a negative one.
        duration : `astropy.units.Quantity`
            How long the burst lasts (time units). Must be positive.
        amplitude : float
            The scale factor while the burst lasts. Must be a finite,
            non-negative number. Defaults to 1.0.

        Raises
        ------
        ValueError
            If `start` is not finite, if `duration` is not positive (a
            zero or negative window is never what a user meant, and would
            make a burst that never happens), or if `amplitude` is not
            finite and non-negative.
        """

        start_s = float(start.to_value(u.s))
        duration_s = float(duration.to_value(u.s))

        if not np.isfinite(start_s):
            raise ValueError(
                f"BurstScaling's start must be finite; got {start}.")

        if not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError(
                f"BurstScaling's duration must be a finite, positive time; "
                f"got {duration}.")

        self._start_s = start_s
        self._duration_s = duration_s
        self._amplitude = _validate_scale(float(amplitude),
                                          "BurstScaling's amplitude")

    @property
    def start(self):
        """
        `astropy.units.Quantity`: the time the burst switches on, in
        seconds.

        Returns
        -------
        `astropy.units.Quantity`
        """
        return self._start_s * u.s

    @property
    def duration(self):
        """
        `astropy.units.Quantity`: how long the burst lasts, in seconds.

        Returns
        -------
        `astropy.units.Quantity`
        """
        return self._duration_s * u.s

    @property
    def amplitude(self):
        """
        float: the scale factor while the burst lasts.

        Returns
        -------
        float
        """
        return self._amplitude

    def __call__(self, time):
        """
        The scale factor at `time`: `self.amplitude` inside the half-open
        window `start <= time < start + duration`, `0.0` outside it.

        Parameters
        ----------
        time : `astropy.units.Quantity`
            The time to evaluate (time units). Scalar.

        Returns
        -------
        float
            `self.amplitude` or `0.0`. Both were checked finite and
            non-negative at construction, so there is nothing left to
            validate here.
        """

        t = time.to_value(u.s)

        if self._start_s <= t < self._start_s + self._duration_s:
            return self._amplitude

        return 0.0


class SinusoidalScaling(SourceScaling):
    """
    A scaling that oscillates smoothly about a mean:

        `mean + amplitude * sin(2 * pi * (t - reference_time) / period)`

    so it runs between `mean - amplitude` and `mean + amplitude`, once
    every `period`, and equals `mean` (and is rising) at
    `reference_time`. This is the shape a periodic modulation has in a
    teaching run -- a source brightening and dimming once per orbit.

    `amplitude` may not exceed `mean`: see `__init__`.
    """

    def __init__(self, mean, amplitude, period, reference_time = 0 * u.s):
        """
        Parameters
        ----------
        mean : float
            The scale factor the oscillation is centred on. Must be a
            finite, non-negative number.
        amplitude : float
            How far the scale factor swings either side of `mean`. Must be
            a finite, non-negative number, and no larger than `mean`.
        period : `astropy.units.Quantity`
            The **full** period of the oscillation (time units), not an
            angular frequency: one whole cycle takes exactly this long, so
            a caller who wants a scaling that repeats once per 5400-second
            orbit writes `5400 * u.s` and never has to get a factor of
            `2 * pi` right. Must be positive.
        reference_time : `astropy.units.Quantity`
            The time the sine is zero and rising, i.e. the phase origin
            (time units). Defaults to `0 * u.s`.

        Raises
        ------
        ValueError
            If `mean` or `amplitude` is not finite and non-negative, if
            `period` is not positive, if `reference_time` is not finite,
            or if `amplitude` is greater than `mean`.

            The last one is the interesting check. With
            `amplitude > mean` the sine dips below zero for part of every
            cycle, which is a negative scaling, which is a negative
            Poisson mean deep inside `InertialSimulator.run_events` --
            precisely the failure `_validate_scale` exists to prevent, and
            precisely as far from its cause. The oscillation is negative
            *by construction* there, not by accident at one time, so the
            honest place to say so is here, where the two numbers that
            disagree are both in hand.
        """

        period_s = float(period.to_value(u.s))
        reference_time_s = float(reference_time.to_value(u.s))

        mean = _validate_scale(float(mean), "SinusoidalScaling's mean")
        amplitude = _validate_scale(float(amplitude),
                                    "SinusoidalScaling's amplitude")

        if not np.isfinite(period_s) or period_s <= 0:
            raise ValueError(
                f"SinusoidalScaling's period must be a finite, positive time; "
                f"got {period}.")

        if not np.isfinite(reference_time_s):
            raise ValueError(
                f"SinusoidalScaling's reference_time must be finite; got "
                f"{reference_time}.")

        if amplitude > mean:
            raise ValueError(
                f"SinusoidalScaling's amplitude ({amplitude}) must not exceed "
                f"its mean ({mean}); otherwise the scaling is negative for "
                f"part of every cycle, which is a negative Poisson mean in "
                f"the simulator.")

        self._mean = mean
        self._amplitude = amplitude
        self._period_s = period_s
        self._reference_time_s = reference_time_s

    @property
    def mean(self):
        """
        float: the scale factor the oscillation is centred on.

        Returns
        -------
        float
        """
        return self._mean

    @property
    def amplitude(self):
        """
        float: how far the scale factor swings either side of `mean`.

        Returns
        -------
        float
        """
        return self._amplitude

    @property
    def period(self):
        """
        `astropy.units.Quantity`: the full period of the oscillation, in
        seconds.

        Returns
        -------
        `astropy.units.Quantity`
        """
        return self._period_s * u.s

    @property
    def reference_time(self):
        """
        `astropy.units.Quantity`: the phase origin, in seconds.

        Returns
        -------
        `astropy.units.Quantity`
        """
        return self._reference_time_s * u.s

    def __call__(self, time):
        """
        The scale factor at `time` (see the class docstring for the
        formula).

        Parameters
        ----------
        time : `astropy.units.Quantity`
            The time to evaluate (time units). Scalar.

        Returns
        -------
        float
            A value between `mean - amplitude` and `mean + amplitude`.
            `__init__` already guarantees the lower end of that range is
            non-negative, so, as with the other scalings, the value is not
            re-checked on every call.
        """

        t = time.to_value(u.s)

        phase = 2 * np.pi * (t - self._reference_time_s) / self._period_s

        return float(self._mean + self._amplitude * np.sin(phase))
