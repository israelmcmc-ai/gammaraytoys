from abc import ABC, abstractmethod
import numpy as np
import astropy.units as u

from ..config_utils import (_as_mapping, _check_keys, _dispatch_type_name,
                            _format_quantity, _quantity)


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

    @classmethod
    def from_config(cls, config, where = 'observation_strategy', earth = None):
        """
        Build an `ObservationStrategy` from its configuration block.

        ```yaml
        {type: ZenithPointing}
        {type: NadirPointing}
        {type: InertialPointing, attitude: 30 deg}
        {type: SpinPointing, rate: 0.1 deg/s, initial_attitude: 0 deg}
        {type: TargetedPointing, sky_angle: 45 deg}
        ```

        Called on `ObservationStrategy`, the block's `type` chooses the class.
        Called on one of the five concrete classes, that class is what gets
        built: `type` may name it or be left out, and naming a different one
        raises rather than quietly handing back the other class.

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
        name = _dispatch_type_name(cls, ObservationStrategy, block, where,
                                   _STRATEGY_TYPES, _STRATEGY_CLASSES,
                                   'observation strategy')

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
    def to_config(self):
        """
        Write this observation strategy back out as a configuration block.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A block `ObservationStrategy.from_config` reads back into an equal
            strategy. A `TargetedPointing`'s Earth is not written here: a
            configuration has exactly one `earth` block, and the loader hands
            it to this strategy.

        Raises
        ------
        ValueError
            If this is not one of the five types a configuration can name.
            Anything else that subclasses `ObservationStrategy` lands here,
            which is the only honest answer -- a configuration has no `type`
            name for it.
        """

        if isinstance(self, ZenithPointing):
            return {'type': 'ZenithPointing'}

        if isinstance(self, NadirPointing):
            return {'type': 'NadirPointing'}

        if isinstance(self, InertialPointing):
            return {'type': 'InertialPointing',
                    'attitude': _format_quantity(self.attitude)}

        if isinstance(self, SpinPointing):
            block = {'type': 'SpinPointing',
                     'rate': _format_quantity(self.rate)}

            if self.initial_attitude != 0 * u.deg:
                block['initial_attitude'] = _format_quantity(self.initial_attitude)

            return block

        if isinstance(self, TargetedPointing):
            return {'type': 'TargetedPointing',
                    'sky_angle': _format_quantity(self.sky_angle)}

        raise ValueError(
            f"{type(self).__name__} is not an observation strategy a "
            f"configuration can describe; the types that are: "
            f"{sorted(set(_STRATEGY_TYPES.values()))}.")


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


# ---------------------------------------------------------------------------
# What a configuration may call each of these classes.
#
# Both tables sit at the bottom of the file because the second one names the
# classes above: `ObservationStrategy.from_config` looks them up when it
# runs, long after this module has finished importing.
# ---------------------------------------------------------------------------


#: Accepted spellings of every observation strategy type.
_STRATEGY_TYPES = {'ZenithPointing': 'ZenithPointing',
                   'NadirPointing': 'NadirPointing',
                   'InertialPointing': 'InertialPointing',
                   'SpinPointing': 'SpinPointing',
                   'TargetedPointing': 'TargetedPointing'}

#: The class each canonical observation strategy type builds.
_STRATEGY_CLASSES = {'ZenithPointing': ZenithPointing,
                     'NadirPointing': NadirPointing,
                     'InertialPointing': InertialPointing,
                     'SpinPointing': SpinPointing,
                     'TargetedPointing': TargetedPointing}
