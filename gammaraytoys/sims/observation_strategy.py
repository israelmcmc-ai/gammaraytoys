from abc import ABC, abstractmethod
import numpy as np
import astropy.units as u


class ObservationStrategy(ABC):
    """
    Abstract base for spacecraft pointing strategies, used by
    `SpacecraftHistory.from_elliptical_orbit` to decide the attitude at
    every generated row.

    A strategy is a small callable, `(time, orbit_radius, orbit_angle) ->
    attitude`: this is what a mission planner chooses, hence the name
    `observation_strategy` rather than `attitude_model`.
    """

    @abstractmethod
    def __call__(self, time, orbit_radius, orbit_angle):
        """
        Parameters
        ----------
        time : Quantity
            Time, seconds. Scalar or array.
        orbit_radius : Quantity
            Spacecraft orbital radius `r`, length units. Scalar or array,
            same shape as `time`.
        orbit_angle : Quantity
            Spacecraft orbital position angle `theta`, CCW from inertial
            +X, angle units. Scalar or array, same shape as `time`.

        Returns
        -------
        Quantity
            Spacecraft attitude `A`: the inertial angle of the detector's
            +y axis, CCW from inertial +X, angle units.
        """

        pass


class ZenithPointing(ObservationStrategy):
    """
    Zenith-pointing strategy: the detector's +y axis always points radially
    outward, away from the Earth. Attitude equals the orbital position
    angle: `A = theta`.
    """

    def __call__(self, time, orbit_radius, orbit_angle):
        """
        Parameters
        ----------
        time : Quantity
            Time, seconds. Scalar or array. Unused by this strategy.
        orbit_radius : Quantity
            Spacecraft orbital radius `r`, length units. Scalar or array.
            Unused by this strategy.
        orbit_angle : Quantity
            Spacecraft orbital position angle `theta`, CCW from inertial
            +X, angle units. Scalar or array, same shape as `time`.

        Returns
        -------
        Quantity
            Attitude `A = theta`, angle units, same shape as `orbit_angle`.
        """

        return orbit_angle


class NadirPointing(ObservationStrategy):
    """
    Nadir-pointing strategy: the detector's +y axis always points radially
    inward, toward the Earth. `A = theta + 180 deg`.
    """

    def __call__(self, time, orbit_radius, orbit_angle):
        """
        Parameters
        ----------
        time : Quantity
            Time, seconds. Scalar or array. Unused by this strategy.
        orbit_radius : Quantity
            Spacecraft orbital radius `r`, length units. Scalar or array.
            Unused by this strategy.
        orbit_angle : Quantity
            Spacecraft orbital position angle `theta`, CCW from inertial
            +X, angle units. Scalar or array, same shape as `time`.

        Returns
        -------
        Quantity
            Attitude `A = theta + 180 deg`, angle units, same shape as
            `orbit_angle`.
        """

        return orbit_angle + 180 * u.deg


class InertialPointing(ObservationStrategy):
    """
    Inertially-fixed pointing strategy: the attitude never changes,
    regardless of orbital position. `A = A_0`.
    """

    def __init__(self, attitude):
        """
        Parameters
        ----------
        attitude : Quantity
            The fixed attitude `A_0`, angle units.
        """

        self.attitude = attitude

    def __call__(self, time, orbit_radius, orbit_angle):
        """
        Parameters
        ----------
        time : Quantity
            Time, seconds. Scalar or array. Used only for its shape.
        orbit_radius : Quantity
            Spacecraft orbital radius `r`, length units. Scalar or array,
            same shape as `time`. Used only for its shape.
        orbit_angle : Quantity
            Spacecraft orbital position angle `theta`, CCW from inertial
            +X, angle units. Scalar or array, same shape as `time`. Used
            only for its shape.

        Returns
        -------
        Quantity
            The fixed attitude `A_0` passed to `__init__`, angle units,
            broadcast to the common shape of `time`, `orbit_radius` and
            `orbit_angle` -- an array for array input, a scalar-shaped
            `Quantity` for scalar input, exactly like `ZenithPointing`,
            `NadirPointing` and `SpinPointing`. (Earlier versions of this
            method returned `self.attitude` unmodified, a bare scalar
            regardless of input shape; that mismatch has been fixed.)
        """

        shape = np.broadcast_shapes(
            np.shape(time), np.shape(orbit_radius), np.shape(orbit_angle))

        return np.broadcast_to(self.attitude.value, shape) * self.attitude.unit


class SpinPointing(ObservationStrategy):
    """
    Constant-rate spin strategy: the attitude increases linearly with time,
    `A = A_0 + rate * t`, where `t` is the absolute simulation time (the
    same clock in which orbital periapsis passage occurs at `t = 0`, see
    `SpacecraftHistory.from_elliptical_orbit`).
    """

    def __init__(self, rate, initial_attitude = 0 * u.deg):
        """
        Parameters
        ----------
        rate : Quantity
            Spin rate `dA/dt`, angle-per-time units (e.g. `deg / s`).
        initial_attitude : Quantity
            Attitude `A_0` at `t = 0`, angle units. Default 0 deg. Note
            this is the attitude at the absolute time origin, not
            necessarily at the first row of a generated history (which may
            start at a nonzero `initial_time`).
        """

        self.rate = rate
        self.initial_attitude = initial_attitude

    def __call__(self, time, orbit_radius, orbit_angle):
        """
        Parameters
        ----------
        time : Quantity
            Time, seconds. Scalar or array.
        orbit_radius : Quantity
            Spacecraft orbital radius `r`, length units. Scalar or array.
            Unused by this strategy.
        orbit_angle : Quantity
            Spacecraft orbital position angle `theta`, CCW from inertial
            +X, angle units. Scalar or array. Unused by this strategy.

        Returns
        -------
        Quantity
            Attitude `A = initial_attitude + rate * time`, angle units,
            same shape as `time`.
        """

        return self.initial_attitude + self.rate * time


class TargetedPointing(ObservationStrategy):
    """
    Target-tracking strategy: the detector's +y axis points directly at a
    fixed-sky-angle source whenever that source is not occulted by the
    Earth, and falls back to zenith pointing whenever it is.

    Pointing *at* a source at inertial sky angle `lambda` means `A =
    lambda`: this puts the source on-axis (`Nu = A - lambda = 0`, Section
    3.4 of the plan). While the target is occulted, this strategy matches
    `ZenithPointing` exactly: `A = theta`.

    **This is the only observation strategy whose attitude is
    discontinuous.** The other four all grow smoothly -- and unwrapped, past
    360 deg -- with time or orbital position. This one mixes a constant
    `sky_angle` with a growing `orbit_angle`, so `A` jumps by `lambda -
    theta` (mod 360 deg) at every rise and every set, the way a real
    spacecraft slews onto a target and slews back off it. That is inherent
    and correct, not a bug: a `.ori` file (or a plot of `attitude` vs. time)
    generated with this strategy shows a sawtooth, not a smooth ramp, and
    any code that unwraps or differentiates `attitude` across an
    occultation boundary must expect that jump.
    """

    def __init__(self, sky_angle, earth):
        """
        Parameters
        ----------
        sky_angle : Quantity
            The target's fixed inertial sky direction `lambda`, angle
            units.
        earth : Earth
            The Earth model used to decide occultation. Required -- this
            does *not* default to `Earth()`. This strategy's entire job is
            deciding *when* the target is occulted, and if it silently
            built its own `Earth` instead of using the one the resulting
            `SpacecraftHistory` is validated against (and the one used
            downstream for the simulation), the two could disagree with
            nothing to catch it. That exact class of bug was already
            found and fixed elsewhere in this PR: `SpacecraftHistory` now
            stores its own `Earth` rather than letting several methods each
            default to their own. Pass the same `Earth` instance used for
            the history and the simulation.
        """

        self.sky_angle = sky_angle
        self.earth = earth

    def __call__(self, time, orbit_radius, orbit_angle):
        """
        Parameters
        ----------
        time : Quantity
            Time, seconds. Scalar or array. Unused by this strategy other
            than (via `orbit_radius`/`orbit_angle`) to determine the output
            shape.
        orbit_radius : Quantity
            Spacecraft orbital radius `r`, length units. Scalar or array,
            same shape as `time`.
        orbit_angle : Quantity
            Spacecraft orbital position angle `theta`, CCW from inertial
            +X, angle units. Scalar or array, same shape as `time`.

        Returns
        -------
        Quantity
            Attitude `A`: `sky_angle` wherever the target is visible,
            `orbit_angle` (the `ZenithPointing` fallback) wherever it is
            occulted. Angle units, with the same shape as `orbit_angle`
            (equivalently `time`/`orbit_radius`) -- a proper array for
            array input, matching every other strategy in this module.
        """

        occulted = self.earth.is_occulted(self.sky_angle, orbit_angle, orbit_radius)

        # np.where() on Quantity objects is not reliable (it can silently
        # drop or mismatch units), so convert both branches to plain floats
        # in one common unit first, select elementwise, then re-attach the
        # unit exactly once.
        sky_angle_deg = self.sky_angle.to_value(u.deg)
        orbit_angle_deg = orbit_angle.to_value(u.deg)

        attitude_deg = np.where(occulted, orbit_angle_deg, sky_angle_deg)

        return attitude_deg * u.deg
