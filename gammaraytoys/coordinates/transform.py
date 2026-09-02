"""
The inertial <-> detector frame transformations of Section 3.4 of
`docs/dev/inertial_sim_plan.md`, derived once here so that every consumer
(sources, the inertial simulator, the notebooks) shares one implementation
and cannot drift.

Frames, in the plan's notation:

- **Inertial frame**: origin at the Earth's centre, angles CCW from +X. The
  spacecraft sits at `(r cos theta, r sin theta)` and a source's direction on
  the sky -- pointing *toward* the source -- is `lambda`.
- **Detector frame**: the `ToyTracker2D` frame. Its "zenith" is +y, and a
  source at off-axis angle `Nu` lies along `(sin Nu, cos Nu)` while the photon
  it emits flies along `270 deg - Nu`.

The spacecraft attitude `A` is the inertial angle of the detector's +y axis,
so in inertial components

```
y_detector = ( cos A,  sin A)
x_detector = ( sin A, -cos A)      # +y rotated by -90 deg
```

Every function here takes and returns `astropy` `Quantity` objects at its
boundary and does its arithmetic on plain floats internally (Section 3.5).
"""

import numpy as np
import astropy.units as u

from .twodim import Cartesian2D

__all__ = ['sky_angle_to_offaxis', 'offaxis_to_sky_angle',
           'inertial_to_detector_position', 'inertial_to_detector_direction',
           'spacecraft_position']


def _wrap_deg(angle_deg):
    """
    Wrap an angle, in plain-float degrees, to `[-180, 180)`.

    Parameters
    ----------
    angle_deg : float or numpy.ndarray of float
        Angle(s) in degrees. Not a `Quantity`.

    Returns
    -------
    float or numpy.ndarray of float
        The same angle(s), in degrees, wrapped to `[-180, 180)`.
    """

    return (angle_deg + 180.0) % 360.0 - 180.0


def sky_angle_to_offaxis(sky_angle, attitude):
    """
    Off-axis angle of a far-field source, from its inertial sky angle and the
    spacecraft attitude: `Nu = A - lambda`, wrapped to `[-180, 180) deg`.

    The wrap lives here rather than in `SpacecraftHistory`, which deliberately
    leaves `attitude` (and `orbit_angle`) unwrapped so they grow smoothly past
    360 deg with time.

    Sanity check (Section 3.4): `A = 90 deg` points the detector's +y along
    inertial +Y, so a source at `lambda = 90 deg` is on-axis (`Nu = 0`) and one
    at `lambda = 0 deg` sits at `Nu = 90 deg`, i.e. on the detector's +x side,
    which is inertial +X.

    Parameters
    ----------
    sky_angle : `astropy.units.Quantity`
        Source direction on the inertial sky, `lambda`, CCW from inertial +X
        and pointing *toward* the source (angle units). Scalar or array.
    attitude : `astropy.units.Quantity`
        Spacecraft attitude `A`, the inertial angle of the detector's +y axis
        (angle units). Scalar or array, broadcastable against `sky_angle`. May
        be unwrapped (e.g. 450 deg).

    Returns
    -------
    `astropy.units.Quantity`
        Off-axis angle `Nu`, in degrees, wrapped to `[-180, 180)`.
    """

    nu_deg = _wrap_deg(attitude.to_value(u.deg) - sky_angle.to_value(u.deg))

    return nu_deg * u.deg


def offaxis_to_sky_angle(offaxis_angle, attitude):
    """
    Inertial sky angle of a far-field source, from its detector-frame off-axis
    angle and the spacecraft attitude: `lambda = A - Nu`, wrapped to
    `[-180, 180) deg`.

    The exact inverse of `sky_angle_to_offaxis` (up to the wrap, which is
    idempotent): `Nu = A - lambda` rearranges to `lambda = A - Nu`, so the two
    functions are the same formula read in opposite directions.

    Parameters
    ----------
    offaxis_angle : `astropy.units.Quantity`
        Off-axis angle `Nu` in the detector frame (angle units), CCW from the
        detector's zenith (+y). Scalar or array.
    attitude : `astropy.units.Quantity`
        Spacecraft attitude `A` (angle units). Scalar or array, broadcastable
        against `offaxis_angle`. May be unwrapped.

    Returns
    -------
    `astropy.units.Quantity`
        Sky angle `lambda`, in degrees, wrapped to `[-180, 180)`.
    """

    sky_angle_deg = _wrap_deg(attitude.to_value(u.deg) - offaxis_angle.to_value(u.deg))

    return sky_angle_deg * u.deg


def inertial_to_detector_position(position, spacecraft_position, attitude):
    """
    Convert an inertial-frame position into the detector frame.

    With `d = P - C`, `P` the point of interest and `C` the spacecraft's
    inertial position (Section 3.4):

    ```
    x_det = d_X sin A - d_Y cos A
    y_det = d_X cos A + d_Y sin A
    ```

    which is just `d` projected onto the detector's own axes,
    `x_detector = (sin A, -cos A)` and `y_detector = (cos A, sin A)`.

    Parameters
    ----------
    position : `Cartesian2D`
        The point `P`, in the inertial frame (length units).
    spacecraft_position : `Cartesian2D`
        The spacecraft's inertial position `C` (length units), e.g. from
        `spacecraft_position(orbit_radius, orbit_angle)`.
    attitude : `astropy.units.Quantity`
        Spacecraft attitude `A` (angle units). May be unwrapped.

    Returns
    -------
    `Cartesian2D`
        The same point in the detector frame, in the length unit of
        `position.x`.
    """

    length_unit = position.x.unit

    dx = position.x.to_value(length_unit) - spacecraft_position.x.to_value(length_unit)
    dy = position.y.to_value(length_unit) - spacecraft_position.y.to_value(length_unit)

    attitude_rad = attitude.to_value(u.rad)
    sin_a = np.sin(attitude_rad)
    cos_a = np.cos(attitude_rad)

    x_det = dx * sin_a - dy * cos_a
    y_det = dx * cos_a + dy * sin_a

    return Cartesian2D(x_det * length_unit, y_det * length_unit)


def inertial_to_detector_direction(direction, attitude):
    """
    Convert an inertial-frame flight direction into the detector frame:
    `direction_det = direction_inertial - A + 90 deg`.

    Deliberately **not** wrapped, to match the existing detector-frame
    convention, which is itself unwrapped: `PointSource` flies its photons
    along `270 deg - Nu`, which for `Nu` in `[0, 360) deg` runs over
    `(-90, 270] deg`. Wrapping here would give the same physical direction but
    a different number, and would break the identity below on the nose.

    The consistency check of Section 3.4: a far-field photon from sky angle
    `lambda` flies at `direction_inertial = lambda + 180 deg`, which this
    formula sends to

    ```
    lambda + 180 - A + 90 = 270 - (A - lambda) = 270 deg - Nu
    ```

    exactly the existing detector-frame convention. (`Nu` here is the
    *unwrapped* `A - lambda`; `sky_angle_to_offaxis` wraps it, so with wrapping
    the identity holds modulo 360 deg.)

    Parameters
    ----------
    direction : `astropy.units.Quantity`
        Flight direction in the inertial frame, CCW from inertial +X (angle
        units). Scalar or array.
    attitude : `astropy.units.Quantity`
        Spacecraft attitude `A` (angle units). Scalar or array, broadcastable
        against `direction`. May be unwrapped.

    Returns
    -------
    `astropy.units.Quantity`
        Flight direction in the detector frame, in degrees, unwrapped.
    """

    direction_det_deg = (direction.to_value(u.deg) - attitude.to_value(u.deg) + 90.0)

    return direction_det_deg * u.deg


def spacecraft_position(orbit_radius, orbit_angle):
    """
    Inertial position of the spacecraft, `C = (r cos theta, r sin theta)`.

    Parameters
    ----------
    orbit_radius : `astropy.units.Quantity`
        Distance `r` from the Earth's centre (length units). Scalar or array.
    orbit_angle : `astropy.units.Quantity`
        Orbital position angle `theta`, CCW from inertial +X (angle units).
        Scalar or array, broadcastable against `orbit_radius`. May be
        unwrapped.

    Returns
    -------
    `Cartesian2D`
        The spacecraft's inertial position, in the length unit of
        `orbit_radius`.
    """

    length_unit = orbit_radius.unit

    r = orbit_radius.to_value(length_unit)
    theta_rad = orbit_angle.to_value(u.rad)

    return Cartesian2D(r * np.cos(theta_rad) * length_unit,
                       r * np.sin(theta_rad) * length_unit)
