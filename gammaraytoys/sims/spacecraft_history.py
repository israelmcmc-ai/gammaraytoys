from dataclasses import dataclass

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.constants import G, M_earth
import matplotlib.pyplot as plt

from .earth import Earth
from .observation_strategy import ZenithPointing

# Column names of the .ori file format (Section 4.1). Units are baked into
# the names themselves.
_TIME_COL = 'time_s'
_RADIUS_COL = 'orbit_radius_km'
_ANGLE_COL = 'orbit_angle_deg'
_ATTITUDE_COL = 'attitude_deg'
_UPTIME_COL = 'uptime_s'


@dataclass(frozen = True)
class SpacecraftInterval:
    """
    One interval of a `SpacecraftHistory`: a span of time over which the
    spacecraft's orbital pose and attitude are held fixed (Section 4.2).

    Attributes
    ----------
    start_time : Quantity
        Time at the start of the interval (inclusive), seconds.
    stop_time : Quantity
        Time at the end of the interval (exclusive), seconds.
    livetime : Quantity
        Live seconds within `[start_time, stop_time)`, i.e. the time during
        which the detector could have recorded events. Always satisfies
        `0 <= livetime <= stop_time - start_time`.
    orbit_radius : Quantity
        Spacecraft orbital radius `r` during this interval, length units.
    orbit_angle : Quantity
        Spacecraft orbital position angle `theta` during this interval, CCW
        from inertial +X, angle units.
    attitude : Quantity
        Spacecraft attitude `A` during this interval: the inertial angle of
        the detector's +y axis, CCW from inertial +X, angle units.
    """

    start_time: u.Quantity
    stop_time: u.Quantity
    livetime: u.Quantity
    orbit_radius: u.Quantity
    orbit_angle: u.Quantity
    attitude: u.Quantity

    @property
    def mid_time(self):
        """
        Quantity: the interval's midpoint time, `(start_time + stop_time) /
        2`. This is the single representative time used to evaluate
        time-dependent source scaling once per interval.
        """

        return 0.5 * (self.start_time + self.stop_time)


def _solve_kepler_equation(mean_anomaly, eccentricity, tol = 1e-10, max_iter = 100):
    """
    Solve Kepler's equation `M = E - e sin(E)` for the eccentric anomaly `E`
    by Newton iteration.

    Parameters
    ----------
    mean_anomaly : numpy.ndarray of float
        Mean anomaly `M`, radians. Any shape; may be outside `[0, 2 pi)`
        for multi-orbit durations, in which case `E` comes back equally
        unwrapped.
    eccentricity : float
        Orbital eccentricity `e`, in `[0, 1)`.
    tol : float
        Convergence tolerance on the Newton step size, radians.
    max_iter : int
        Maximum number of Newton iterations to attempt.

    Returns
    -------
    numpy.ndarray of float
        Eccentric anomaly `E`, radians, same shape as `mean_anomaly`.

    Raises
    ------
    RuntimeError
        If the iteration has not converged to `tol` after `max_iter` steps.
    """

    E = np.array(mean_anomaly, dtype = float, copy = True)

    for _ in range(max_iter):
        delta = (E - eccentricity * np.sin(E) - mean_anomaly) / (1 - eccentricity * np.cos(E))
        E -= delta
        if np.all(np.abs(delta) < tol):
            return E

    raise RuntimeError(
        f"Kepler's equation failed to converge to tol={tol} rad after {max_iter} "
        f"Newton iterations (worst residual {np.max(np.abs(delta)):.3e} rad); "
        f"eccentricity={eccentricity}")


class SpacecraftHistory:
    """
    A spacecraft's orbital and attitude history over time: a sequence of
    timestamped rows read from, or written to, a `.ori` file (Section 4).

    Rows define timestamps `t_0 < t_1 < ... < t_N`, giving rise to `N`
    intervals (not `N+1`). Interval `i` spans `[t_i, t_{i+1})` and takes
    both its pose (`orbit_radius`, `orbit_angle`, `attitude`) and its
    livetime (`uptime_s`) from row `i`. Row `N` is a pure terminator: it
    contributes only `t_N`, closing the last interval, and its pose and
    uptime are never read.
    """

    def __init__(self, time, orbit_radius, orbit_angle, attitude, uptime, earth = None):
        """
        Construct directly from row arrays. Most users will use `open()` or
        `from_elliptical_orbit()` instead.

        Parameters
        ----------
        time : Quantity (array)
            Row timestamps `t_0 < t_1 < ... < t_N`, seconds. There are
            `len(time) - 1` intervals.
        orbit_radius : Quantity (array)
            Row orbital radius `r`, length units, same length as `time`.
            `orbit_radius[N]` (the terminator row) is stored but never read
            by `__iter__`.
        orbit_angle : Quantity (array)
            Row orbital position angle `theta`, angle units, same length as
            `time`. `orbit_angle[N]` is stored but never read.
        attitude : Quantity (array)
            Row attitude `A`, angle units, same length as `time`.
            `attitude[N]` is stored but never read.
        uptime : Quantity (array)
            Row forward-looking livetime, seconds, same length as `time`.
            `uptime[i]` is the livetime of interval `i = [time[i],
            time[i+1])`; `uptime[N]` is stored but never read.
        earth : Earth or None
            Earth model used only to validate `orbit_radius > earth.radius`
            for every interval. Defaults to `Earth()`.

        Raises
        ------
        ValueError
            If there are fewer than 2 rows, if timestamps are not strictly
            increasing, if any interval's livetime falls outside
            `[0, stop_time - start_time]`, or if any interval's
            `orbit_radius` does not exceed `earth.radius`. The terminator
            row (index `N`) is exempt from the livetime and radius checks,
            since neither of its values is ever read.
        """

        if earth is None:
            earth = Earth()

        self._time_s = np.asarray(time.to_value(u.s), dtype = float)
        self._orbit_radius_km = np.asarray(orbit_radius.to_value(u.km), dtype = float)
        self._orbit_angle_deg = np.asarray(orbit_angle.to_value(u.deg), dtype = float)
        self._attitude_deg = np.asarray(attitude.to_value(u.deg), dtype = float)
        self._uptime_s = np.asarray(uptime.to_value(u.s), dtype = float)

        self._validate(earth)

    def _validate(self, earth):
        """Run every check listed in Section 4.2; raise ValueError, with a
        clear message, on the first one that fails."""

        n_rows = len(self._time_s)

        if n_rows < 2:
            raise ValueError(
                f"A SpacecraftHistory needs at least 2 rows (1 interval); got {n_rows}.")

        if not (len(self._orbit_radius_km) == len(self._orbit_angle_deg)
                == len(self._attitude_deg) == len(self._uptime_s) == n_rows):
            raise ValueError("time, orbit_radius, orbit_angle, attitude and uptime "
                             "must all have the same length.")

        dt = np.diff(self._time_s)
        if not np.all(dt > 0):
            bad = np.nonzero(dt <= 0)[0]
            raise ValueError(
                f"SpacecraftHistory timestamps must be strictly increasing; "
                f"row(s) {bad.tolist()} do not increase over the next row.")

        # Interval quantities: row i = 0..N-1. The terminator (row N) is
        # exempt -- its uptime and orbit_radius are never read.
        livetime = self._uptime_s[:-1]
        bad_livetime = np.nonzero((livetime < 0) | (livetime > dt))[0]
        if len(bad_livetime) > 0:
            raise ValueError(
                f"Interval livetime (uptime_s) must satisfy 0 <= uptime_s <= "
                f"stop_time - start_time; violated at interval(s) {bad_livetime.tolist()}.")

        earth_radius_km = earth.radius.to_value(u.km)
        interval_radius = self._orbit_radius_km[:-1]
        bad_radius = np.nonzero(interval_radius <= earth_radius_km)[0]
        if len(bad_radius) > 0:
            raise ValueError(
                f"orbit_radius_km must exceed the Earth's radius ({earth.radius}); "
                f"violated at interval(s) {bad_radius.tolist()}.")

    @classmethod
    def open(cls, filename, earth = None):
        """
        Read a `.ori` file: a CSV with columns `time_s`, `orbit_radius_km`,
        `orbit_angle_deg`, `attitude_deg`, `uptime_s`, one header line,
        `#`-prefixed comment lines and blank lines ignored (Section 4.1).

        Parameters
        ----------
        filename : str or path-like
            Path to the `.ori` file.
        earth : Earth or None
            Earth model used to validate `orbit_radius > earth.radius`.
            Defaults to `Earth()`.

        Returns
        -------
        SpacecraftHistory

        Raises
        ------
        ValueError
            See `__init__` -- any of the Section 4.2 validations failing.
        """

        # pandas' default C float parser is not bit-exact for full-precision
        # values (it can be off by a ULP), which is enough to break the
        # strict 0 <= uptime_s <= dt check right at its boundary after a
        # write/read round trip. `float_precision='round_trip'` uses the
        # slower but exact parser instead.
        df = pd.read_csv(filename, comment = '#', float_precision = 'round_trip')

        return cls(time = df[_TIME_COL].to_numpy() * u.s,
                  orbit_radius = df[_RADIUS_COL].to_numpy() * u.km,
                  orbit_angle = df[_ANGLE_COL].to_numpy() * u.deg,
                  attitude = df[_ATTITUDE_COL].to_numpy() * u.deg,
                  uptime = df[_UPTIME_COL].to_numpy() * u.s,
                  earth = earth)

    def write(self, filename):
        """
        Write this history to a `.ori` file: a CSV with columns `time_s`,
        `orbit_radius_km`, `orbit_angle_deg`, `attitude_deg`, `uptime_s`
        (Section 4.1).

        Parameters
        ----------
        filename : str or path-like
            Destination path.
        """

        df = pd.DataFrame({
            _TIME_COL: self._time_s,
            _RADIUS_COL: self._orbit_radius_km,
            _ANGLE_COL: self._orbit_angle_deg,
            _ATTITUDE_COL: self._attitude_deg,
            _UPTIME_COL: self._uptime_s,
        })

        df.to_csv(filename, index = False)

    @classmethod
    def from_elliptical_orbit(cls, semi_major_axis,
                              eccentricity = 0.0,
                              earth = None,
                              observation_strategy = None,
                              time_step = 1 * u.s,
                              duration = None,
                              argument_of_periapsis = 0 * u.deg,
                              initial_time = 0 * u.s,
                              livetime_fraction = 1.0):
        """
        Generate a `SpacecraftHistory` for a Keplerian elliptical orbit,
        sampled at (approximately) `time_step` over `duration` (Section 4.4).

        Kepler's equation is solved properly, by Newton iteration on the
        eccentric anomaly `E`, rather than approximated by a uniform-in-angle
        ellipse:

        ```
        n     = sqrt(mu / a^3)                     # mean motion
        M     = n * t                              # mean anomaly
        M     = E - e sin(E)                       # solved for E
        nu    = true anomaly, from E and e
        r     = a (1 - e cos E)
        theta = nu + argument_of_periapsis
        ```

        The orbital clock has periapsis passage at the absolute time `t = 0
        s` (not at `initial_time`, which only shifts where the *generated
        rows* start); this is the same zero point `SpinPointing` measures
        `initial_attitude` from.

        Parameters
        ----------
        semi_major_axis : Quantity
            Orbital semi-major axis `a`, length units.
        eccentricity : float
            Orbital eccentricity `e`, in `[0, 1)`. Default 0.0 (a circular
            orbit, the simplest case to reason about).
        earth : Earth or None
            Earth model, used only to check that the perigee clears the
            surface. Defaults to `Earth()`. The gravitational parameter
            used for the orbit itself always comes from `astropy.constants`
            (`G * M_earth`), never from this `Earth` instance.
        observation_strategy : callable or None
            `(time, orbit_radius, orbit_angle) -> attitude`, evaluated at
            every generated row. Defaults to `ZenithPointing()`.
        time_step : Quantity
            Requested spacing between rows, time units. Default 1 s. The
            actual spacing is adjusted slightly so that a whole number of
            equal intervals exactly tiles `duration`.
        duration : Quantity or None
            Total time span covered by the generated history, time units.
            Defaults to one full orbital period, `2 pi sqrt(a^3 / mu)`.
        argument_of_periapsis : Quantity
            Argument of periapsis `omega`, angle units, added to the true
            anomaly to give `orbit_angle`. Default 0 deg.
        initial_time : Quantity
            Time value assigned to the first row, time units. Default 0 s.
        livetime_fraction : float
            Fraction of each interval's span written to `uptime_s`, in
            `[0, 1]`. Default 1.0 (fully live). A convenience for generated
            files only -- a real, hand-edited or externally-produced file
            may vary the livetime fraction row by row.

        Returns
        -------
        SpacecraftHistory

        Raises
        ------
        ValueError
            If `eccentricity` is outside `[0, 1)`, if `livetime_fraction`
            is outside `[0, 1]`, if `duration` or `time_step` is not
            positive, or if the perigee `a (1 - e)` does not clear the
            Earth's surface.
        RuntimeError
            If the Kepler solve fails to converge.
        """

        if earth is None:
            earth = Earth()

        if observation_strategy is None:
            observation_strategy = ZenithPointing()

        if not (0.0 <= eccentricity < 1.0):
            raise ValueError(f"eccentricity must be in [0, 1); got {eccentricity}.")

        if not (0.0 <= livetime_fraction <= 1.0):
            raise ValueError(f"livetime_fraction must be in [0, 1]; got {livetime_fraction}.")

        a = semi_major_axis.to_value(u.km)
        e = float(eccentricity)

        perigee = a * (1 - e) * u.km
        if perigee <= earth.radius:
            raise ValueError(
                f"Perigee ({perigee}) does not clear the Earth's radius "
                f"({earth.radius}); this orbit passes through the planet.")

        mu = (G * M_earth).to_value(u.km**3 / u.s**2)  # gravitational parameter, from constants
        n = np.sqrt(mu / a**3)  # mean motion, rad/s
        period = (2 * np.pi / n) * u.s

        if duration is None:
            duration = period

        duration_s = duration.to_value(u.s)
        step_s = time_step.to_value(u.s)
        initial_time_s = initial_time.to_value(u.s)

        if duration_s <= 0:
            raise ValueError(f"duration must be positive; got {duration}.")
        if step_s <= 0:
            raise ValueError(f"time_step must be positive; got {time_step}.")

        n_intervals = max(1, int(np.round(duration_s / step_s)))
        row_times_s = initial_time_s + np.linspace(0, duration_s, n_intervals + 1)

        # Mean anomaly on the absolute clock (periapsis passage at t = 0).
        mean_anomaly = n * row_times_s

        E = _solve_kepler_equation(mean_anomaly, e)

        # True anomaly from E. atan2's principal range matches E/2 in
        # [0, pi] only for E in [0, 2 pi), so unwrap by the number of full
        # revolutions to keep nu (and hence orbit_angle) continuous even
        # when `duration` spans more than one orbit.
        revs = np.floor(E / (2 * np.pi))
        E_mod = E - revs * 2 * np.pi
        nu = (2 * np.arctan2(np.sqrt(1 + e) * np.sin(E_mod / 2), np.sqrt(1 - e) * np.cos(E_mod / 2))
              + revs * 2 * np.pi)

        r = a * (1 - e * np.cos(E))
        theta_deg = np.degrees(nu) + argument_of_periapsis.to_value(u.deg)

        time_q = row_times_s * u.s
        orbit_radius_q = r * u.km
        orbit_angle_q = theta_deg * u.deg

        attitude_q = observation_strategy(time_q, orbit_radius_q, orbit_angle_q)
        attitude_deg = np.broadcast_to(
            np.atleast_1d(attitude_q.to_value(u.deg)), row_times_s.shape)

        dt_s = np.diff(row_times_s)
        uptime_s = np.zeros_like(row_times_s)
        # Terminator row's uptime is never read; 0 is the "sensible" value a
        # writer emits for it (Section 4.2), matching the example .ori file.
        uptime_s[:-1] = livetime_fraction * dt_s

        return cls(time = time_q,
                  orbit_radius = orbit_radius_q,
                  orbit_angle = orbit_angle_q,
                  attitude = attitude_deg * u.deg,
                  uptime = uptime_s * u.s,
                  earth = earth)

    @property
    def nintervals(self):
        """int: the number of intervals, `len(rows) - 1`."""

        return len(self._time_s) - 1

    @property
    def total_livetime(self):
        """Quantity: the sum of the per-interval livetimes `L_i`, seconds.
        The terminator row's `uptime_s` is excluded, since it belongs to no
        interval."""

        return np.sum(self._uptime_s[:-1]) * u.s

    def __iter__(self):
        """
        Iterate over the `nintervals` intervals of this history, in order.

        Yields
        ------
        SpacecraftInterval
            Interval `i`, spanning `[time[i], time[i+1])`, with pose and
            livetime taken from row `i`. The terminator row (index `N`)
            never yields an interval of its own; it only supplies the
            `stop_time` of the last one.
        """

        for i in range(self.nintervals):
            yield SpacecraftInterval(
                start_time = self._time_s[i] * u.s,
                stop_time = self._time_s[i + 1] * u.s,
                livetime = self._uptime_s[i] * u.s,
                orbit_radius = self._orbit_radius_km[i] * u.km,
                orbit_angle = self._orbit_angle_deg[i] * u.deg,
                attitude = self._attitude_deg[i] * u.deg,
            )

    def plot(self, ax = None, earth = None, nposes = 12):
        """
        Plot this history's orbit path in the inertial frame, together with
        the Earth and a handful of representative spacecraft poses (position
        markers with an attitude arrow).

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to draw into. A new figure and axes are created if None.
        earth : Earth or None
            Earth model to draw for scale and context. Defaults to
            `Earth()`; purely cosmetic, no validation is performed against
            it.
        nposes : int
            Number of representative spacecraft poses (markers + attitude
            arrows) to draw along the history, roughly evenly spaced by
            interval index. Default 12.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the history was drawn into.
        """

        if ax is None:
            fig, ax = plt.subplots()

        if earth is None:
            earth = Earth()

        length_unit = u.km

        earth.plot(ax = ax)

        r = self._orbit_radius_km
        theta = np.radians(self._orbit_angle_deg)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        ax.plot(x, y, '-', color = 'tab:orange', lw = 1, label = 'orbit')

        n_marks = min(nposes, self.nintervals)
        idx = np.unique(np.linspace(0, self.nintervals - 1, n_marks).astype(int))

        arrow_len = 0.08 * np.max(r)
        attitude = np.radians(self._attitude_deg)
        for i in idx:
            ax.plot(x[i], y[i], 'o', color = 'tab:blue', ms = 4)
            ax.annotate('', xytext = (x[i], y[i]),
                       xy = (x[i] + arrow_len * np.cos(attitude[i]),
                             y[i] + arrow_len * np.sin(attitude[i])),
                       arrowprops = dict(arrowstyle = '->', color = 'tab:blue'))

        ax.set_xlabel(f'X [{length_unit}]')
        ax.set_ylabel(f'Y [{length_unit}]')
        ax.set_aspect('equal')
        ax.legend(loc = 'upper right')

        return ax
