from abc import ABC, abstractmethod
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
        return orbit_angle


class NadirPointing(ObservationStrategy):
    """
    Nadir-pointing strategy: the detector's +y axis always points radially
    inward, toward the Earth. `A = theta + 180 deg`.
    """

    def __call__(self, time, orbit_radius, orbit_angle):
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
        return self.attitude


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
        return self.initial_attitude + self.rate * time
