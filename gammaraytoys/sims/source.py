from abc import ABC, abstractmethod
from gammaraytoys.coordinates import Cartesian2D
import numpy as np
import astropy.units as u
from .event import Photon
from .spectrum import MonoenergeticSpectrum
from .transform import sky_angle_to_offaxis, offaxis_to_sky_angle
from copy import copy
import matplotlib.pyplot as plt
from histpy import Histogram, Axis

# `ToyTracker2D.plot()` hardcodes its data coordinates to this unit -- every
# source marker drawn on top of it must match, or it lands in the right
# place numerically but the wrong place on the figure.
_PLOT_LENGTH_UNIT = u.cm

# How far outside the sky circle (itself `2 x` the surrounding-circle
# radius, see `FarFieldSource.plot_sky_circle`) a source's marker/arc sits,
# as a multiple of the sky circle's radius. Shared by every far-field
# `plot_sky_*` helper so a star and an arc drawn for different sources line
# up on the same ring.
_SKY_MARKER_RADIUS_FACTOR = 1.08


def _expand_axes_limits(ax, center, radius, length_unit = _PLOT_LENGTH_UNIT):
    """
    Grow `ax`'s x/y limits, if necessary, to contain a square bounding box
    of half-width `radius` centered on `center`.

    `ToyTracker2D.plot()` sizes its axes to `1.5 x` its own surrounding
    circle -- far smaller than the sky circle a far-field source draws at
    `2 x` that radius -- so plotting a source on top of it would otherwise
    leave the sky circle and its marker entirely outside the visible axes.
    Limits are only ever widened here, never narrowed, so plotting several
    sources in sequence on the same axes keeps every earlier one visible.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes to expand.
    center : `Cartesian2D`
        Centre of the bounding box, in the detector frame.
    radius : `astropy.units.Quantity`
        Half-width of the bounding box (length units).
    length_unit : `astropy.units.Unit`
        Unit `ax`'s data coordinates are already in. Defaults to
        `_PLOT_LENGTH_UNIT` (cm), matching `ToyTracker2D.plot()`.
    """

    x_center = center.x.to_value(length_unit)
    y_center = center.y.to_value(length_unit)
    r = radius.to_value(length_unit)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.set_xlim(min(xlim[0], x_center - r), max(xlim[1], x_center + r))
    ax.set_ylim(min(ylim[0], y_center - r), max(ylim[1], y_center + r))


class Source(ABC):
    """
    Abstract base class for every photon source.

    Holds the pieces that are common to every source regardless of how it is
    normalized or where it sits: the energy spectrum, the total normalization
    used to scale it (`normalization`), and the helpers that plot or
    discretize the spectrum. Concrete geometry -- where the source is and how
    photons are drawn from it -- lives in the two abstract subclasses below
    it:

    - `FarFieldSource`: normalized by a flux, in `1/cm/s` (a source at
      infinite distance, e.g. a point on the sky). Exposes that flux through
      `flux(pose)`.
    - `NearFieldSource`: normalized by a rate, in `1/s` (a source at a fixed
      position near the detector). Exposes that rate through `rate`.
    """

    @property
    @abstractmethod
    def normalization(self):
        """
        Total (spectrum-integrated) normalization used to scale the
        spectrum, whatever quantity that is for this source's family.

        `diff_flux`, `integrate_flux`, `discretize_spectrum` and
        `plot_spectrum` are all built on this instead of on `flux` directly,
        so they work unchanged for both source families: `FarFieldSource`
        returns its `flux` (`1/cm/s`) here, `NearFieldSource` returns its
        `rate` (`1/s`).

        Returns
        -------
        `astropy.units.Quantity` or None
            The source's normalization, in whatever unit its family uses, or
            `None` if the source has no normalization set.
        """
        pass

    @property
    @abstractmethod
    def spectrum(self):
        """
        The source's energy spectrum.

        Returns
        -------
        `Spectrum`
            The (unit-normalized) shape of the source's energy distribution.
        """
        pass

    def diff_flux(self, energy):
        """
        Differential flux at a given photon energy.

        Parameters
        ----------
        energy : `astropy.units.Quantity`
            Photon energy, in energy units (e.g. MeV).

        Returns
        -------
        `astropy.units.Quantity`
            `normalization * spectrum.pdf(energy)`, i.e. the flux (far-field)
            or rate (near-field) per unit energy at `energy`.
        """
        return self.normalization * self.spectrum.pdf(energy)

    def integrate_flux(self, lo_energy, hi_energy):
        """
        Flux integrated over an energy interval.

        Parameters
        ----------
        lo_energy, hi_energy : `astropy.units.Quantity`
            Lower and upper bounds of the energy interval (energy units,
            broadcastable against each other for multiple intervals).

        Returns
        -------
        `astropy.units.Quantity`
            `normalization * spectrum.integrate(lo_energy, hi_energy)`.
        """
        return self.normalization * self.spectrum.integrate(lo_energy, hi_energy)

    def discretize_spectrum(self, axis):
        """
        Bin the source's flux onto an energy axis.

        Parameters
        ----------
        axis : `histpy.Axis`
            Energy bin edges to integrate the flux over.

        Returns
        -------
        `histpy.Histogram`
            One-dimensional histogram over `axis`, with each bin holding the
            flux (or, for a near-field source, rate) integrated across that
            bin's energy range.
        """

        binned_spec = Histogram(axis,
                                unit = self.normalization.unit,
                                contents = self.integrate_flux(axis.lower_bounds,
                                                               axis.upper_bounds)
                                )

        return binned_spec

    def plot_spectrum(self, ax = None, e2 = False,
                      energy_units = None, y_units = None,
                      discretize_axis = None,
                      **kwargs):
        """
        Plot the source's differential energy spectrum.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            Axes to plot on. A new figure is created if not given.
        e2 : bool
            If True, plot `E^2 dN/dE` instead of `dN/dE`.
        energy_units : `astropy.units.Unit`, optional
            Units for the energy axis. Defaults to MeV.
        y_units : `astropy.units.Unit`, optional
            Units for the y axis. Derived from `self.normalization`'s unit
            when not given: for a far-field source (`flux`, `1/cm/s`) that
            is `1/(erg cm s)` for `dN/dE`, `erg/(cm s)` for `E^2 dN/dE`; for
            a near-field source (`rate`, `1/s`) that is `1/(erg s)` and
            `erg/s` respectively.
        discretize_axis : `histpy.Axis`, optional
            If given, plot the spectrum binned onto this energy axis (via
            `discretize_spectrum`) instead of a smooth curve.
        **kwargs
            Passed through to the underlying plot call.

        Returns
        -------
        `matplotlib.axes.Axes`
            The axes the spectrum was plotted on.
        """

        if self.normalization is None:
            raise RuntimeError("Set a flux or rate before plotting the spectrum")

        if ax is None:
            fig,ax = plt.subplots()

        if isinstance(self.spectrum, MonoenergeticSpectrum):
            raise RuntimeError("Can't plot monoenergetic spectrum")

        if energy_units is None:
            energy_units = u.MeV
        else:
            energy_units = u.Unit(energy_units)

        if discretize_axis is None:
            energy = np.geomspace(self.spectrum.min_energy, self.spectrum.max_energy, 1000).to(energy_units)
            y = self.diff_flux(energy)
        else:
            discretize_axis = Axis(discretize_axis)
            energy = discretize_axis.centers
            binned_spec = self.discretize_spectrum(discretize_axis)
            y = binned_spec / binned_spec.axis.widths

        if e2:
            if y_units is None:
                y_units = u.Unit(self.normalization.unit * u.erg)
            else:
                y_units = u.Unit(y_units)

            y *= energy**2
            y_label = f'$E^2 dN/dE$ [{y_units}]'
        else:
            if y_units is None:
                y_units = u.Unit(self.normalization.unit / u.erg)
            else:
                y_units = u.Unit(y_units)

            y_label = f'$dN/dE$ [{y_units}]'

        y = y.to(y_units)

        if discretize_axis is None:
            ax.plot(energy.value, y.value, **kwargs)
        else:
            y.plot(ax, **kwargs)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(f'Energy [{energy_units}]')

        ax.set_ylabel(y_label)

        return ax

    @abstractmethod
    def random_photon(self, detector, pose = None):
        """
        Draw one random photon aimed at the detector.

        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector the photon is thrown at. Used to size and place the
            throwing plane.
        pose : `SpacecraftInterval` or None
            Spacecraft pose to evaluate the source in. `None` (the default)
            means pure detector-frame mode -- the source's `random_photon`
            behaves exactly as it did before the inertial simulator existed,
            which is the path the tutorials take. A non-`None` pose puts the
            source in inertial mode: it is aimed through the spacecraft's
            attitude (see `gammaraytoys.sims.transform`) and its photons are
            subject to Earth occultation (see `FarFieldSource.occultable`).

        Returns
        -------
        `Photon` or None
            A photon in the detector frame, ready for
            `detector.simulate_event()`, or `None` if the photon was
            occulted by the Earth.
        """
        pass

    @abstractmethod
    def simulated_rate(self, detector, pose = None):
        """
        Expected rate of photons launched at the detector.

        This is the rate *before* occultation and before any time-dependent
        scaling -- purely "how many photons per second does this source
        throw at the detector's throwing plane". It is what `Simulator` (and,
        later, `InertialSimulator`) uses to mix sources normalized by a flux
        and sources normalized by a rate in the same run: everything is
        summed as a rate.

        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector the photons are thrown at. Its
            `throwing_plane_size` sets the overall scale.
        pose : `SpacecraftInterval` or None
            Spacecraft pose to evaluate the source in. `None` means pure
            detector-frame mode. Every source in this package has a
            pose-independent rate today -- the pose is threaded through for
            the (later) Earth albedo, whose apparent flux depends on the
            orbital radius.

        Returns
        -------
        `astropy.units.Quantity`
            Rate in `1/s`, or `None` if the source has no normalization set.
        """
        pass

class FarFieldSource(Source):
    """
    Abstract base class for sources at effectively infinite distance from
    the detector -- point sources, the isotropic sky, and (later) extended
    and Earth-albedo sources.

    Normalized by a flux in `1/cm/s`: `IsotropicSource` and `PointSource`
    already use `flux` to mean the flux **integrated over the whole sky**,
    not a per-unit-angle brightness -- every far-field source must match
    that convention, since it is what makes `simulated_rate()` a plain
    product with the detector's `throwing_plane_size` (see `flux`).
    """

    def flux(self, pose = None):
        """
        Flux integrated over the whole sky, in `1/cm/s`.

        The default implementation simply returns `self._flux` and ignores
        `pose` -- true for every far-field source in this codebase except
        the (future) Earth-albedo source, whose apparent flux depends on the
        spacecraft's orbital radius. Subclasses that need pose-dependence
        override this method; `normalization` and `simulated_rate()` then
        follow automatically. Concrete subclasses set `self._flux` in their
        constructor via the `flux` keyword.

        Parameters
        ----------
        pose : `SpacecraftInterval` or None
            Spacecraft pose. Ignored by this default implementation.

        Returns
        -------
        `astropy.units.Quantity` or None
            Flux in `1/cm/s`, or `None` if the source has no normalization
            set.
        """
        return self._flux

    @property
    def normalization(self):
        """
        Total normalization used to scale the spectrum: the flux.

        This is what `Source.diff_flux`, `integrate_flux`,
        `discretize_spectrum` and `plot_spectrum` use polymorphically; for a
        far-field source it is simply `flux()` (evaluated with no pose, i.e.
        the pure detector-frame flux).

        Returns
        -------
        `astropy.units.Quantity` or None
            `flux()`, in `1/cm/s`, or `None` if the source has no
            normalization set.
        """
        return self.flux()

    def simulated_rate(self, detector, pose = None):
        """
        Expected rate of photons launched at the detector.

        For every far-field source this is uniformly the sky-integrated
        flux times the detector's throwing-plane size:
        `flux(pose) * detector.throwing_plane_size`.

        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector the photons are thrown at. `throwing_plane_size` is
            `2a`, where `a` is the surrounding-circle radius.
        pose : `SpacecraftInterval` or None
            Spacecraft pose, forwarded to `flux`. `None` means pure
            detector-frame mode.

        Returns
        -------
        `astropy.units.Quantity`
            Rate in `1/s`, or `None` if the source has no normalization set.
        """
        flux = self.flux(pose)

        if flux is None:
            return None

        return flux * detector.throwing_plane_size

    @property
    def occultable(self):
        """
        Whether this source's photons can be blocked by the Earth.

        `True` for every source whose photons genuinely arrive from the
        distant sky, which is every far-field source in this package except
        the Earth albedo: `EarthAlbedoSource` (added in a later PR)
        overrides this to `False`, because its photons come from the
        Earth's direction *by construction* and a blanket occultation test
        would reject all of them (Section 8.1 of the plan).

        Only consulted when a source is drawn with a `pose`; in pure
        detector-frame mode there is no Earth and nothing to occult
        against.

        Returns
        -------
        bool
            `True` -- Earth occultation applies to this source.
        """
        return True

    def _occulted(self, sky_angle, pose):
        """
        Whether a photon arriving from inertial sky angle `sky_angle` is
        blocked by the Earth at this pose.

        This is the per-photon hot path, so it goes straight to
        `Earth._is_occulted` -- plain floats, radians, `orbit_radius` in the
        Earth's own radius unit -- rather than through the public,
        `Quantity`-converting `Earth.is_occulted`, which is roughly an order
        of magnitude more expensive. The public method is a thin wrapper
        over the same private one, so there is a single implementation of
        the geometry and no risk of the two drifting apart.

        Parameters
        ----------
        sky_angle : `astropy.units.Quantity`
            Inertial direction the photon arrives *from*, `lambda`, CCW from
            inertial +X (angle units).
        pose : `SpacecraftInterval`
            The spacecraft pose, supplying `orbit_angle`, `orbit_radius` and
            the `earth` to test against.

        Returns
        -------
        bool
            `True` if the photon is occulted and must be discarded. Always
            `False` when `self.occultable` is `False`.
        """

        if not self.occultable:
            return False

        earth = pose.earth

        return bool(earth._is_occulted(
            sky_angle.to_value(u.rad),
            pose.orbit_angle.to_value(u.rad),
            pose.orbit_radius.to_value(earth.radius.unit)))

    def _sky_radius(self, detector):
        """
        Radius of the "sky", drawn by `plot_sky_circle` as twice the
        detector's own surrounding circle.

        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector to size the sky circle against.

        Returns
        -------
        `astropy.units.Quantity`
            `2 * detector.surrounding_circle_radius`, in length units.
        """
        return 2 * detector.surrounding_circle_radius

    def plot_sky_circle(self, ax, detector, length_unit = _PLOT_LENGTH_UNIT, **kwargs):
        """
        Draw the sky as a faint dotted circle around the detector.

        The sky circle has radius `2 x` the detector's surrounding-circle
        radius, centered on `detector.surrounding_circle_center`, styled
        like the existing surrounding-circle drawn by
        `ToyTracker2D.plot(draw_surrounding_circle = True)`. Expands `ax`'s
        limits (only ever growing them, see `_expand_axes_limits`) so the
        circle -- entirely outside the plain `detector.plot()` limits -- is
        actually visible.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes already showing the detector, typically from
            `detector.plot()`.
        detector : `ToyTracker2D`
            The detector this source's sky circle is drawn against.
        length_unit : `astropy.units.Unit`
            Unit for the plotted data coordinates. Defaults to cm, matching
            `ToyTracker2D.plot()`; only change this if `ax` was set up with
            a different unit.
        **kwargs
            Passed through to `ax.plot`, overriding the default faint
            dotted style.

        Returns
        -------
        `matplotlib.axes.Axes`
            The axes the sky circle was plotted on.
        """

        center = detector.surrounding_circle_center
        radius = self._sky_radius(detector)

        theta = np.linspace(0, 360, 200) * u.deg
        x = (center.x + radius * np.cos(theta)).to_value(length_unit)
        y = (center.y + radius * np.sin(theta)).to_value(length_unit)

        style = dict(ls = ':', color = 'black', alpha = .3)
        style.update(kwargs)
        ax.plot(x, y, **style)

        _expand_axes_limits(ax, center, _SKY_MARKER_RADIUS_FACTOR * radius, length_unit)

        return ax

    def plot_sky_marker(self, ax, detector, offaxis_angle,
                        length_unit = _PLOT_LENGTH_UNIT,
                        marker_radius_factor = _SKY_MARKER_RADIUS_FACTOR,
                        **kwargs):
        """
        Draw a single red star just outside the sky circle, marking a point
        source's direction.

        The star sits along the unit vector `(sin Nu, cos Nu)` from
        `detector.surrounding_circle_center` (plan section 3.3: `Nu = 0` is
        the detector's `+y`, `Nu = 90 deg` is `+x`), at `marker_radius_factor
        x` the sky circle's radius -- just outside it. Used by
        `PointSource.plot`; the same convention underlies `plot_sky_arc`
        below.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes already showing the detector and, typically, its sky
            circle (`plot_sky_circle`).
        detector : `ToyTracker2D`
            The detector this source is plotted against.
        offaxis_angle : `astropy.units.Quantity`
            Off-axis angle Nu (angle units) the star is placed at.
        length_unit : `astropy.units.Unit`
            Unit for the plotted data coordinates. Defaults to cm.
        marker_radius_factor : float
            How far outside the sky circle's radius to place the star, as a
            multiple of that radius. Defaults to `_SKY_MARKER_RADIUS_FACTOR`
            (1.08).
        **kwargs
            Passed through to `ax.plot`, overriding the default red-star
            style.

        Returns
        -------
        `matplotlib.axes.Axes`
            The axes the star was plotted on.
        """

        center = detector.surrounding_circle_center
        marker_radius = marker_radius_factor * self._sky_radius(detector)

        x = (center.x + marker_radius * np.sin(offaxis_angle)).to_value(length_unit)
        y = (center.y + marker_radius * np.cos(offaxis_angle)).to_value(length_unit)

        style = dict(marker = '*', color = 'red', markersize = 15, linestyle = 'none')
        style.update(kwargs)
        ax.plot(x, y, **style)

        _expand_axes_limits(ax, center, marker_radius, length_unit)

        return ax

    def plot_sky_arc(self, ax, detector, center_angle, extent,
                     length_unit = _PLOT_LENGTH_UNIT,
                     marker_radius_factor = _SKY_MARKER_RADIUS_FACTOR,
                     **kwargs):
        """
        Draw an arc just outside the sky circle, spanning `extent` centered
        on `center_angle`.

        This is the shared primitive behind `IsotropicSource.plot` (called
        with `extent = 360 deg`, tracing a full circle) and, in later PRs,
        `ExtendedSource` and `EarthAlbedoSource`, which will call it with
        their own characteristic angular extent (a von Mises width, or the
        Earth's angular radius `rho`) instead of the whole sky. The arc uses
        the same `(sin Nu, cos Nu)` convention and marker radius as
        `plot_sky_marker`.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes already showing the detector and, typically, its sky
            circle (`plot_sky_circle`).
        detector : `ToyTracker2D`
            The detector this source is plotted against.
        center_angle : `astropy.units.Quantity`
            Off-axis angle Nu (angle units) at the centre of the arc.
        extent : `astropy.units.Quantity`
            Full angular width of the arc (angle units). `360 deg` traces a
            closed circle.
        length_unit : `astropy.units.Unit`
            Unit for the plotted data coordinates. Defaults to cm.
        marker_radius_factor : float
            How far outside the sky circle's radius to draw the arc, as a
            multiple of that radius. Defaults to `_SKY_MARKER_RADIUS_FACTOR`
            (1.08).
        **kwargs
            Passed through to `ax.plot`, overriding the default red-line
            style.

        Returns
        -------
        `matplotlib.axes.Axes`
            The axes the arc was plotted on.
        """

        center = detector.surrounding_circle_center
        marker_radius = marker_radius_factor * self._sky_radius(detector)

        nu = center_angle + np.linspace(-0.5, 0.5, 200) * extent

        x = (center.x + marker_radius * np.sin(nu)).to_value(length_unit)
        y = (center.y + marker_radius * np.cos(nu)).to_value(length_unit)

        style = dict(color = 'red', lw = 2)
        style.update(kwargs)
        ax.plot(x, y, **style)

        _expand_axes_limits(ax, center, marker_radius, length_unit)

        return ax

class NearFieldSource(Source):
    """
    Abstract base class for sources at a fixed position near the detector,
    rather than at infinite distance on the sky (see `NearPointSource`,
    added in a later PR).

    Normalized by a total emission rate in `1/s` rather than a flux: "flux"
    -- a brightness per unit length of sky -- is not a meaningful quantity
    for a source close enough that its distance to the detector matters. A
    near-field source therefore has no `flux` at all (that method only
    exists on `FarFieldSource`); its normalization is `rate` instead.

    Unlike `FarFieldSource`, there is no shared formula for `simulated_rate()`
    across near-field geometries -- each source's acceptance depends on its
    own position relative to the detector -- so it stays abstract here and is
    implemented by each concrete subclass.
    """

    @property
    @abstractmethod
    def rate(self):
        """
        Total emission rate of the source, in `1/s`.

        The near-field analogue of `FarFieldSource.flux`: the source's
        overall normalization, integrated over its full emission angle,
        before any geometric acceptance onto the detector's throwing plane
        is applied (that acceptance is folded in by `simulated_rate`).
        Implemented by concrete near-field sources, e.g. `NearPointSource`.

        Returns
        -------
        `astropy.units.Quantity` or None
            Rate in `1/s`, or `None` if the source has no normalization set.
        """
        pass

    @property
    @abstractmethod
    def position(self):
        """
        Position of the source in the detector frame.

        Unlike a far-field source, a near-field source's location is a
        genuine detector-frame position rather than a direction -- it can
        sit inside the detector's surrounding circle entirely. Implemented
        by concrete near-field sources, e.g. `NearPointSource` (added in a
        later PR).

        Returns
        -------
        `Cartesian2D`
            The source's position, in detector-frame length units (e.g. cm).
        """
        pass

    @property
    def normalization(self):
        """
        Total normalization used to scale the spectrum: the rate.

        This is what `Source.diff_flux`, `integrate_flux`,
        `discretize_spectrum` and `plot_spectrum` use polymorphically; for a
        near-field source it is `rate` (there is no `flux` to fall back on
        -- see the class docstring).

        Returns
        -------
        `astropy.units.Quantity` or None
            `rate`, in `1/s`, or `None` if the source has no normalization
            set.
        """
        return self.rate

    def plot(self, ax, detector, length_unit = _PLOT_LENGTH_UNIT, **kwargs):
        """
        Draw this source's location on axes already showing
        `detector.plot()`.

        Near-field sources sit at a fixed detector-frame position -- inside
        or close to the detector itself -- so they're marked with a red
        star drawn directly at `self.position`, unlike a far-field source's
        marker on the sky circle (see `FarFieldSource.plot_sky_marker`).

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes already showing the detector, typically from
            `detector.plot()`.
        detector : `ToyTracker2D`
            The detector this source is being plotted against. Present for
            interface symmetry with the far-field `plot` methods; this
            source's own position does not depend on it.
        length_unit : `astropy.units.Unit`
            Unit for the plotted data coordinates. Defaults to cm, matching
            `ToyTracker2D.plot()`.
        **kwargs
            Passed through to `ax.plot`, overriding the default red-star
            style.

        Returns
        -------
        `matplotlib.axes.Axes`
            The axes the source was plotted on.
        """

        style = dict(marker = '*', color = 'red', markersize = 15, linestyle = 'none')
        style.update(kwargs)
        ax.plot(self.position.x.to_value(length_unit),
               self.position.y.to_value(length_unit),
               **style)

        return ax

class PointSource(FarFieldSource):
    """
    A far-field source at a single fixed direction.

    That direction is given in exactly one of two frames, and which one is
    used decides how the source behaves for the rest of its life:

    - **detector-frame** (`offaxis_angle`): the source sits at a fixed
      off-axis angle Nu, the detector is the centre of the universe, and
      `pose` is ignored entirely. This is the original behaviour and the one
      the tutorials and `Simulator` use.
    - **inertial** (`sky_angle`): the source sits at a fixed direction
      `lambda` on the inertial sky. Its off-axis angle is then whatever the
      spacecraft's attitude makes it, `Nu = A - lambda`, recomputed from the
      `pose` for every photon, and the photon is discarded when the Earth is
      in the way. This is the mode `InertialSimulator` uses.

    Give one or the other, never both and never neither.

    Photons are thrown from a plane tangent to the detector's surrounding
    circle, perpendicular to the direction to the source, and fly along that
    direction.
    """

    def __init__(self, offaxis_angle = None, spectrum = None,
                 flux = None, flux_pivot = None, pivot_energy = None,
                 chirality = None, chirality_degree = 0,
                 sky_angle = None):
        """
        Parameters
        ----------
        offaxis_angle : `astropy.units.Quantity`, optional
            Off-axis angle Nu in the detector frame (angle units), CCW from
            detector zenith (+y). See the module-level convention in
            `ToyTracker2D`. Mutually exclusive with `sky_angle`: give
            exactly one of the two.
        spectrum : `Spectrum`
            The source's energy spectrum shape. Required -- it only carries
            a `None` default so that `offaxis_angle` could gain one without
            reordering the existing positional arguments.
        flux : `astropy.units.Quantity`, optional
            Total flux integrated over the whole sky, in `1/cm/s`. Needed
            for normalization (`simulated_rate`, `diff_flux`, ...); either
            this or `flux_pivot`/`pivot_energy` may be given, or neither, in
            which case `flux` is `None`.
        flux_pivot : `astropy.units.Quantity`, optional
            Differential flux at `pivot_energy` (`1/cm/s/energy`), used
            together with `pivot_energy` to derive `flux` from the spectrum
            shape when `flux` itself is not given directly.
        pivot_energy : `astropy.units.Quantity`, optional
            Energy at which `flux_pivot` is specified.
        chirality : int or None
            Dominant chirality (+1 or -1) of the photons this source emits,
            or `None` for no chirality preference.
        chirality_degree : float
            Degree of polarization, in `[0, 1]`: 0 draws chirality with no
            preference (50/50 between the two values), 1 always draws the
            dominant `chirality`, and values in between interpolate --
            the fraction of photons actually drawn with the dominant
            `chirality` is `0.5 + chirality_degree/2`, not
            `chirality_degree` itself. Defaults to 0 (unpolarized), so a
            source is unpolarized unless asked otherwise. Ignored if
            `chirality` is `None`, which is itself the default -- the
            photon then picks its own chirality at random.
        sky_angle : `astropy.units.Quantity`, optional
            Direction `lambda` on the inertial sky (angle units), CCW from
            inertial +X and pointing *toward* the source. Mutually exclusive
            with `offaxis_angle`: give exactly one of the two. A source
            given a `sky_angle` can only be drawn from with a `pose` (see
            `random_photon`).

        Raises
        ------
        ValueError
            If both or neither of `offaxis_angle` and `sky_angle` are given,
            or if no `spectrum` is given.
        """

        if spectrum is None:
            raise ValueError("PointSource requires a spectrum.")

        if (offaxis_angle is None) == (sky_angle is None):
            raise ValueError(
                "A PointSource is aimed either in the detector frame, with "
                "`offaxis_angle`, or on the inertial sky, with `sky_angle`. "
                "Give exactly one of the two; got "
                f"offaxis_angle={offaxis_angle}, sky_angle={sky_angle}.")

        self._spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree

        self.sky_angle = sky_angle

        # For an inertial (sky_angle) source this starts out None and is
        # re-aimed from the pose on every draw, exactly the way
        # IsotropicSource re-aims its own internal PointSource; the throwing
        # plane cache below is keyed on the off-axis angle, so re-aiming is
        # safe.
        self.offaxis_angle = offaxis_angle

        if flux is not None:
            self._flux = flux
        else:
            if flux_pivot is None or pivot_energy is None:
                self._flux = None
            else:
                self._flux = (flux_pivot/spectrum.pdf(pivot_energy)).to(1/u.cm/u.s)

        # Cache for the throwing plane, which is a pure function of the detector
        # and the off-axis angle. Both can change between photons -- IsotropicSource
        # points a single PointSource somewhere new for every draw -- so the cache
        # is keyed on both, not on the detector alone.
        self._detector = None
        self._cached_offaxis_angle = None
        self._plane_origin = None
        self._throw_parallel = None

    @property
    def spectrum(self):
        """The source's energy spectrum."""
        return self._spectrum

    def random_injection_position(self, detector):
        """
        Draw a random photon starting position on the detector's throwing
        plane at the current `offaxis_angle`.

        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector to throw at.

        Returns
        -------
        `Cartesian2D`
            A position uniformly distributed along the throwing plane, in
            the detector frame.
        """

        if detector is not self._detector or self.offaxis_angle != self._cached_offaxis_angle:
            self._detector = detector
            self._cached_offaxis_angle = self.offaxis_angle
            self._plane_origin, self._throw_parallel = detector.throwing_plane(self.offaxis_angle)

        perp_norm_dist = np.random.uniform(-1,1)
        return  Cartesian2D(self._plane_origin.x + self._throw_parallel.x * perp_norm_dist,
                            self._plane_origin.y + self._throw_parallel.y * perp_norm_dist)

    def random_photon(self, detector, pose = None):
        """
        Draw one random photon aimed at the detector.

        For a detector-frame source (built with `offaxis_angle`) this simply
        throws from that fixed off-axis angle and `pose` is ignored
        entirely, occultation included -- there is no sky direction to
        occult.

        For an inertial source (built with `sky_angle`) the source is first
        re-aimed at `Nu = A - lambda` using the pose's attitude, and the
        photon is then discarded if the Earth is between the spacecraft and
        the source.

        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector the photon is thrown at.
        pose : `SpacecraftInterval` or None
            Spacecraft pose. Ignored for a detector-frame source; required
            for an inertial one, whose off-axis angle is undefined without
            an attitude.

        Returns
        -------
        `Photon` or None
            A photon starting on the throwing plane, flying along
            `270 deg - offaxis_angle`, with an energy drawn from `spectrum`
            and a chirality drawn per `chirality`/`chirality_degree`;
            `None` if the source was occulted by the Earth at this pose.

        Raises
        ------
        ValueError
            If this source was given a `sky_angle` and no `pose`.
        """

        if self.sky_angle is not None:

            if pose is None:
                raise ValueError(
                    "This PointSource is aimed on the inertial sky "
                    f"(sky_angle = {self.sky_angle}), so it needs a "
                    "spacecraft pose to know where that is in the detector "
                    "frame. Pass `pose`, or build the source with "
                    "`offaxis_angle` instead for pure detector-frame use.")

            # Re-aim at Nu = A - lambda for this pose (Section 3.4).
            self.offaxis_angle = sky_angle_to_offaxis(self.sky_angle, pose.attitude)

            if self._occulted(self.sky_angle, pose):
                return None

        chirality = copy(self.chirality)
        if chirality is not None:
            if np.random.uniform() > 0.5 + self.chirality_degree/2:
                # Flip to non-dominant chirality
                chirality *= -1

        return Photon(position = self.random_injection_position(detector),
                      direction = 270*u.deg - self.offaxis_angle,
                      energy = self.spectrum.random_energy(),
                      chirality = chirality)

    def plot(self, ax, detector, **kwargs):
        """
        Draw this source's sky location on axes already showing
        `detector.plot()`.

        Draws the sky circle (`plot_sky_circle`) and a single red star just
        outside it (`plot_sky_marker`), at this source's `offaxis_angle`.

        The plot is in the detector frame, so an inertial (`sky_angle`)
        source can only be drawn once it has been aimed at a pose -- i.e.
        after at least one `random_photon(detector, pose)` call, which is
        what sets its `offaxis_angle`.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes already showing the detector, typically from
            `detector.plot()`.
        detector : `ToyTracker2D`
            The detector this source is being plotted against; sizes the
            sky circle and its marker radius.
        **kwargs
            Passed through to the star's `ax.plot` call (`plot_sky_marker`),
            overriding its default red-star style. The sky circle keeps its
            own default style regardless.

        Returns
        -------
        `matplotlib.axes.Axes`
            The axes the source was plotted on.

        Raises
        ------
        RuntimeError
            If this is an inertial source that has not been aimed at a pose
            yet, so it has no detector-frame direction to draw.
        """

        if self.offaxis_angle is None:
            raise RuntimeError(
                "This PointSource is aimed on the inertial sky "
                f"(sky_angle = {self.sky_angle}) and has not been evaluated "
                "at a spacecraft pose yet, so it has no detector-frame "
                "off-axis angle to plot. Draw a photon with a pose first, or "
                "plot it at an off-axis angle of your choosing with "
                "`plot_sky_marker`.")

        self.plot_sky_circle(ax, detector)
        self.plot_sky_marker(ax, detector, self.offaxis_angle, **kwargs)

        return ax

class IsotropicSource(FarFieldSource):
    """
    A far-field source uniform over the whole sky.

    Internally re-aims a single reusable `PointSource` to a new random
    off-axis angle for every photon, rather than building a fresh one per
    draw.
    """

    def __init__(self, spectrum, flux = None, chirality = None, chirality_degree = 0):
        """
        Parameters
        ----------
        spectrum : `Spectrum`
            The source's energy spectrum shape.
        flux : `astropy.units.Quantity`, optional
            Total flux integrated over the whole sky, in `1/cm/s`. `None`
            leaves the source unnormalized.
        chirality : int or None
            Dominant chirality (+1 or -1) of the photons this source emits,
            or `None` for no chirality preference.
        chirality_degree : float
            Degree of polarization, in `[0, 1]`: 0 draws chirality with no
            preference (50/50 between the two values), 1 always draws the
            dominant `chirality`, and values in between interpolate --
            the fraction of photons actually drawn with the dominant
            `chirality` is `0.5 + chirality_degree/2`, not
            `chirality_degree` itself. Defaults to 0 (unpolarized), so a
            source is unpolarized unless asked otherwise. Ignored if
            `chirality` is `None`, which is itself the default -- the
            photon then picks its own chirality at random.
        """

        self._spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree
        self._flux = flux

        # A single point source that gets aimed somewhere new for every photon,
        # rather than a throw-away PointSource per photon.
        self._point_source = PointSource(offaxis_angle = 0*u.deg,
                                         spectrum = spectrum,
                                         chirality = chirality,
                                         chirality_degree = chirality_degree)

    @property
    def spectrum(self):
        """The source's energy spectrum."""
        return self._spectrum

    def random_photon(self, detector, pose = None):
        """
        Draw one random photon from a uniformly random direction.

        A uniform sky is uniform in either frame, so the off-axis angle is
        drawn the same way with or without a pose. What a pose adds is
        occultation: the drawn direction is converted back to an inertial
        sky angle, `lambda = A - Nu`, and the photon is discarded if the
        Earth is in the way. Drawing the direction *first* and then
        rejecting is deliberate -- it needs no bespoke truncated-sky
        sampling, and it is what makes the simulator's Poisson mean the
        unocculted one (Section 6 of the plan).

        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector the photon is thrown at.
        pose : `SpacecraftInterval` or None
            Spacecraft pose. `None` (the default) means pure detector-frame
            mode, with no occultation.

        Returns
        -------
        `Photon` or None
            A photon thrown from a uniformly random off-axis angle in
            `[0, 360) deg`, with an energy drawn from `spectrum` and a
            chirality drawn per `chirality`/`chirality_degree`; `None` if
            that direction was occulted by the Earth at this pose.
        """

        # Re-sync in case these were changed after construction
        self._point_source.chirality = self.chirality
        self._point_source.chirality_degree = self.chirality_degree

        offaxis_angle = np.random.uniform(0,360)*u.deg
        self._point_source.offaxis_angle = offaxis_angle

        if pose is not None:
            sky_angle = offaxis_to_sky_angle(offaxis_angle, pose.attitude)

            if self._occulted(sky_angle, pose):
                return None

        return self._point_source.random_photon(detector = detector)

    def plot(self, ax, detector, **kwargs):
        """
        Draw this source's sky coverage on axes already showing
        `detector.plot()`.

        Draws the sky circle (`plot_sky_circle`) and a full 360 deg arc
        just outside it (`plot_sky_arc`), representing uniform coverage of
        the whole sky -- the same arc primitive `ExtendedSource` and
        `EarthAlbedoSource` will reuse in later PRs with a narrower extent.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes already showing the detector, typically from
            `detector.plot()`.
        detector : `ToyTracker2D`
            The detector this source is being plotted against; sizes the
            sky circle and its arc radius.
        **kwargs
            Passed through to the arc's `ax.plot` call (`plot_sky_arc`),
            overriding its default red-line style. The sky circle keeps its
            own default style regardless.

        Returns
        -------
        `matplotlib.axes.Axes`
            The axes the source was plotted on.
        """

        self.plot_sky_circle(ax, detector)
        self.plot_sky_arc(ax, detector, center_angle = 0*u.deg, extent = 360*u.deg, **kwargs)

        return ax
