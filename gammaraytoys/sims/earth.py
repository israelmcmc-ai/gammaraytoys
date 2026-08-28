import numpy as np
import astropy.units as u
from astropy.constants import R_earth
import matplotlib.pyplot as plt


class Earth:
    """
    A 2D Earth: a single circle of a given radius, centred on the origin of
    the inertial frame (see the plan's Section 3.3). The same radius is used
    both for occulting far-field sources and, elsewhere in this package, for
    Earth-albedo emission. There is no atmosphere shell.
    """

    def __init__(self, radius = None):
        """
        Parameters
        ----------
        radius : Quantity or None
            Earth radius `R_E`, length units. Defaults to astropy's nominal
            `R_earth` constant.
        """

        self.radius = radius if radius is not None else R_earth.to(u.km)

    def _check_orbit_radius(self, orbit_radius_value, orbit_radius_quantity):
        """
        Raise if any of the plain-float `orbit_radius_value` (in
        `self.radius`'s unit) does not exceed `self.radius.value`.
        `orbit_radius_quantity` is only used to render a readable message.
        """

        if np.any(np.asarray(orbit_radius_value) <= self.radius.value):
            raise ValueError(
                f"orbit_radius ({orbit_radius_quantity}) must be strictly "
                f"greater than the Earth's radius ({self.radius}); the "
                "spacecraft would be at or below the surface.")

    def _angular_radius_rad(self, orbit_radius_km):
        """
        Plain-float core of `angular_radius`: `rho = arcsin(R_E / r)`.

        This is the *only* place the arcsin formula appears in this file;
        both `angular_radius` and `_is_occulted` route through it so the
        geometry cannot drift between the two.

        Parameters
        ----------
        orbit_radius_km : float or numpy.ndarray of float
            Distance `r` from the Earth's centre, as a plain float (or
            array of floats) in the *same unit as `self.radius`* (km by
            default, but whatever unit this `Earth` was constructed with).
            Not a `Quantity`. Despite the `_km` suffix (kept to match the
            unit `Earth()`'s default `self.radius` uses), the actual
            required unit is `self.radius.unit`.

        Returns
        -------
        float or numpy.ndarray of float
            Angular radius `rho`, in **radians**.

        No validation. `orbit_radius_km` is assumed to already exceed
        `self.radius.value` (`arcsin` of a value > 1 would otherwise
        silently produce `nan`, not raise) — this is a private, hot-path
        helper, and its callers are responsible for validating.
        """

        return np.arcsin(self.radius.value / orbit_radius_km)

    def angular_radius(self, orbit_radius):
        """
        Angular radius of the Earth as seen from a spacecraft at distance
        `orbit_radius` from the Earth's centre: `rho = arcsin(R_E / r)`.

        Parameters
        ----------
        orbit_radius : Quantity
            Distance `r` from the Earth's centre to the spacecraft, length
            units. Scalar or array; every value must exceed `self.radius`.

        Returns
        -------
        Quantity
            Angular radius `rho`, in degrees. Always below 90 deg, since
            `orbit_radius` must exceed `self.radius`.

        Raises
        ------
        ValueError
            If any `orbit_radius` does not exceed `self.radius` (the
            spacecraft would be at or below the Earth's surface).
        """

        r = np.asarray(orbit_radius.to_value(self.radius.unit))

        self._check_orbit_radius(r, orbit_radius)

        rho_rad = self._angular_radius_rad(r)

        return (rho_rad * u.rad).to(u.deg)

    def _is_occulted(self, sky_angle_rad, orbit_angle_rad, orbit_radius_km):
        """
        Plain-float, unit-stripped core of `is_occulted`. This is the hot
        path: PR 3's `InertialSimulator` calls it once per photon, so it
        takes plain floats (or numpy arrays of floats) throughout and does
        no `Quantity` work at all.

        **Units, strictly:**

        - `sky_angle_rad`, `orbit_angle_rad` : **radians**, not degrees,
          not `Quantity`.
        - `orbit_radius_km` : a plain float (or array), in the *same unit
          as `self.radius`* (km by default). Despite the name, NOT
          necessarily kilometres if this `Earth` was built with
          `radius` in another unit — matches `_angular_radius_rad`.

        Passing degrees, or a `Quantity`, or an `orbit_radius` in the
        wrong unit, will not raise here: it will silently produce a wrong
        boolean. This method exists purely for speed; every other caller
        should go through the public `is_occulted`, which converts once at
        the boundary and delegates here (see Section 3.5 of
        `docs/dev/inertial_sim_plan.md`).

        Same geometry as `is_occulted`: `nadir = orbit_angle + pi`,
        `occulted <=> |wrap(sky_angle - nadir)| < rho`, `wrap(...)` to
        `[-pi, pi)`, `rho` from `_angular_radius_rad`.

        No validation: `orbit_radius_km` is assumed to already exceed
        `self.radius.value`. Skipped deliberately for speed on this
        per-photon hot path (that check, plus the `Quantity` conversions
        it would need, is exactly the overhead this method exists to
        avoid) — validating `orbit_radius` is the caller's job, done once
        by the public `is_occulted`.

        Returns
        -------
        bool or numpy.ndarray of bool
            True wherever the source is occulted by the Earth.
        """

        rho = self._angular_radius_rad(orbit_radius_km)

        nadir = orbit_angle_rad + np.pi

        # Wrap (sky_angle - nadir) to [-pi, pi).
        delta = (sky_angle_rad - nadir + np.pi) % (2 * np.pi) - np.pi

        return np.abs(delta) < rho

    def is_occulted(self, sky_angle, orbit_angle, orbit_radius):
        """
        Whether a far-field photon arriving from `sky_angle` is blocked by
        the Earth, for a spacecraft at orbital position `orbit_angle` and
        distance `orbit_radius` from the Earth's centre.

        A far-field source is occulted iff its sky direction falls within
        the Earth's angular radius `rho` of nadir:
        `nadir = orbit_angle + 180 deg`,
        `occulted <=> |wrap(sky_angle - nadir)| < rho`, where `wrap(...)`
        wraps the angle difference to `[-180, 180)` deg.

        This converts its `Quantity` arguments once and delegates the
        actual geometry to `_is_occulted`; see that method if you are
        calling this once per photon in a tight loop and the `Quantity`
        overhead matters.

        Parameters
        ----------
        sky_angle : Quantity
            Source direction on the inertial sky, `lambda`, CCW from
            inertial +X, angle units. Scalar or array.
        orbit_angle : Quantity
            Spacecraft orbital position angle `theta`, angle units. Scalar
            or array, broadcastable against `sky_angle`.
        orbit_radius : Quantity
            Spacecraft orbital radius `r`, length units. Scalar or array,
            broadcastable against `sky_angle`.

        Returns
        -------
        bool or numpy.ndarray of bool
            True wherever the source is occulted by the Earth.

        Raises
        ------
        ValueError
            If any `orbit_radius` does not exceed `self.radius`.
        """

        r = np.asarray(orbit_radius.to_value(self.radius.unit))

        self._check_orbit_radius(r, orbit_radius)

        sky_angle_rad = sky_angle.to_value(u.rad)
        orbit_angle_rad = orbit_angle.to_value(u.rad)

        return self._is_occulted(sky_angle_rad, orbit_angle_rad, r)

    def plot(self, ax = None):
        """
        Draw the Earth as a filled circle centred on the origin of the
        inertial frame.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to draw into. A new figure and axes are created if None.

        Returns
        -------
        matplotlib.axes.Axes
            The axes the Earth was drawn into.
        """

        if ax is None:
            fig, ax = plt.subplots()

        length_unit = u.km
        radius = self.radius.to_value(length_unit)

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.fill(radius * np.cos(theta), radius * np.sin(theta),
                color = 'tab:blue', alpha = .3, label = 'Earth')

        ax.set_xlabel(f'X [{length_unit}]')
        ax.set_ylabel(f'Y [{length_unit}]')
        ax.set_aspect('equal')

        return ax
