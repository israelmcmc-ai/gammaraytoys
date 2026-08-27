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
        r_e = self.radius.value

        if np.any(r <= r_e):
            raise ValueError(
                f"orbit_radius ({orbit_radius}) must be strictly greater than "
                f"the Earth's radius ({self.radius}); the spacecraft would be "
                "at or below the surface.")

        rho = np.arcsin(r_e / r)

        return (rho * u.rad).to(u.deg)

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

        rho = self.angular_radius(orbit_radius)

        nadir = orbit_angle + 180 * u.deg

        # Wrap (sky_angle - nadir) to [-180, 180) deg.
        delta = (sky_angle - nadir + 180 * u.deg) % (360 * u.deg) - 180 * u.deg

        return np.abs(delta) < rho

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
