from abc import ABC, abstractmethod
from gammaraytoys.coordinates import Cartesian2D
import numpy as np
import astropy.units as u
from .event import Photon
from .spectrum import MonoenergeticSpectrum
from copy import copy
import matplotlib.pyplot as plt
from histpy import Histogram, Axis

class Source(ABC):
    """
    Abstract base class for every photon source.

    Holds the pieces that are common to every source regardless of how it is
    normalized or where it sits: the energy spectrum, the total normalization
    used to scale it (`flux`), and the helpers that plot or discretize the
    spectrum. Concrete geometry -- where the source is and how photons are
    drawn from it -- lives in the two abstract subclasses below it:

    - `FarFieldSource`: normalized by a flux, in `1/cm/s` (a source at
      infinite distance, e.g. a point on the sky).
    - `NearFieldSource`: normalized by a rate, in `1/s` (a source at a fixed
      position near the detector).
    """

    @property
    @abstractmethod
    def flux(self):
        """
        Total (spectrum-integrated) normalization of the source.

        Returns
        -------
        `astropy.units.Quantity` or None
            Flux in `1/cm/s` for a far-field source, or `None` if either the
            source has no normalization set, or the source is a near-field
            source (whose normalization is a rate, not a flux -- see
            `NearFieldSource`).
        """
        pass

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
            behaves exactly as it did before the inertial simulator existed.
            A non-`None` pose is meaningful starting with the inertial
            simulator (see `InertialSimulator`); it is accepted here so the
            signature is uniform across sources, but has no effect yet.

        Returns
        -------
        `Photon` or None
            A photon in the detector frame, ready for
            `detector.simulate_event()`, or `None` if the photon was
            occulted (not possible yet -- occultation is introduced by the
            inertial simulator).
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
            detector-frame mode. Meaningful starting with the inertial
            simulator; has no effect yet.

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
    product with the detector's `throwing_plane_size` (see `sky_integrated_flux`).
    """

    def sky_integrated_flux(self, pose = None):
        """
        Flux integrated over the whole sky, in `1/cm/s`.

        The default implementation simply returns `flux` and ignores `pose`
        -- true for every far-field source in this codebase except the
        (future) Earth-albedo source, whose apparent flux depends on the
        spacecraft's orbital radius. Subclasses that need pose-dependence
        override this method; `flux` and `simulated_rate()` then follow
        automatically.

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
        return self.flux

    @property
    def normalization(self):
        """
        Total normalization used to scale the spectrum: the flux.

        This is what `Source.diff_flux`, `integrate_flux`,
        `discretize_spectrum` and `plot_spectrum` use polymorphically; for a
        far-field source it is simply `flux`.

        Returns
        -------
        `astropy.units.Quantity` or None
            `flux`, in `1/cm/s`, or `None` if the source has no
            normalization set.
        """
        return self.flux

    def simulated_rate(self, detector, pose = None):
        """
        Expected rate of photons launched at the detector.

        For every far-field source this is uniformly the sky-integrated
        flux times the detector's throwing-plane size:
        `sky_integrated_flux(pose) * detector.throwing_plane_size`.

        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector the photons are thrown at. `throwing_plane_size` is
            `2a`, where `a` is the surrounding-circle radius.
        pose : `SpacecraftInterval` or None
            Spacecraft pose, forwarded to `sky_integrated_flux`. `None`
            means pure detector-frame mode.

        Returns
        -------
        `astropy.units.Quantity`
            Rate in `1/s`, or `None` if the source has no normalization set.
        """
        flux = self.sky_integrated_flux(pose)

        if flux is None:
            return None

        return flux * detector.throwing_plane_size

class NearFieldSource(Source):
    """
    Abstract base class for sources at a fixed position near the detector,
    rather than at infinite distance on the sky (see `NearPointSource`,
    added in a later PR).

    Normalized by a total emission rate in `1/s` rather than a flux: "flux"
    -- a brightness per unit length of sky -- is not a meaningful quantity
    for a source close enough that its distance to the detector matters.
    `flux` is therefore always `None` for a near-field source;
    `Simulator.total_flux` uses that to report `None` whenever a near-field
    source is mixed into a run, since a single flux no longer describes it.

    Unlike `FarFieldSource`, there is no shared formula for `simulated_rate()`
    across near-field geometries -- each source's acceptance depends on its
    own position relative to the detector -- so it stays abstract here and is
    implemented by each concrete subclass.
    """

    @property
    def flux(self):
        """
        Always `None`.

        A near-field source is normalized by a rate (`1/s`), not a flux
        (`1/cm/s`) -- see the class docstring.
        """
        return None

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
    def normalization(self):
        """
        Total normalization used to scale the spectrum: the rate.

        This is what `Source.diff_flux`, `integrate_flux`,
        `discretize_spectrum` and `plot_spectrum` use polymorphically; for a
        near-field source it is `rate`, not `flux` (which is always `None`
        here -- see `flux`).

        Returns
        -------
        `astropy.units.Quantity` or None
            `rate`, in `1/s`, or `None` if the source has no normalization
            set.
        """
        return self.rate

class PointSource(FarFieldSource):
    """
    A far-field source at a fixed off-axis angle in the detector frame.

    Photons are thrown from a plane tangent to the detector's surrounding
    circle, perpendicular to the direction to the source, and fly along that
    direction.
    """

    def __init__(self, offaxis_angle, spectrum,
                 flux = None, flux_pivot = None, pivot_energy = None,
                 chirality = None, chirality_degree = 1):
        """
        Parameters
        ----------
        offaxis_angle : `astropy.units.Quantity`
            Off-axis angle Nu in the detector frame (angle units), CCW from
            detector zenith (+y). See the module-level convention in
            `ToyTracker2D`.
        spectrum : `Spectrum`
            The source's energy spectrum shape.
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
            `chirality_degree` itself. Defaults to 1 (fully polarized),
            since a `PointSource` is normally used to model a single
            polarized beam. Ignored if `chirality` is `None`.
        """

        self._spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree

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
    def flux(self):
        """Total flux integrated over the whole sky, in `1/cm/s`, or `None`."""
        return self._flux

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
        Draw one random photon aimed at the detector from `offaxis_angle`.

        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector the photon is thrown at.
        pose : `SpacecraftInterval` or None
            Ignored. `PointSource` is aimed by a fixed detector-frame
            `offaxis_angle`; pose-dependent aiming arrives with the inertial
            simulator, which re-aims a `PointSource` by `sky_angle` instead.

        Returns
        -------
        `Photon`
            A photon starting on the throwing plane, flying along
            `270 deg - offaxis_angle`, with an energy drawn from `spectrum`
            and a chirality drawn per `chirality`/`chirality_degree`.
        """

        chirality = copy(self.chirality)
        if chirality is not None:
            if np.random.uniform() > 0.5 + self.chirality_degree/2:
                # Flip to non-dominant chirality
                chirality *= -1

        return Photon(position = self.random_injection_position(detector),
                      direction = 270*u.deg - self.offaxis_angle,
                      energy = self.spectrum.random_energy(),
                      chirality = chirality)

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
            `chirality_degree` itself. Defaults to 0 (unpolarized), since an
            `IsotropicSource` is normally used to model an unpolarized
            diffuse background. Ignored if `chirality` is `None`.
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
    def flux(self):
        """Total flux integrated over the whole sky, in `1/cm/s`, or `None`."""
        return self._flux

    @property
    def spectrum(self):
        """The source's energy spectrum."""
        return self._spectrum

    def random_photon(self, detector, pose = None):
        """
        Draw one random photon from a uniformly random direction.

        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector the photon is thrown at.
        pose : `SpacecraftInterval` or None
            Ignored. See `Source.random_photon`.

        Returns
        -------
        `Photon`
            A photon thrown from a uniformly random off-axis angle in
            `[0, 360) deg`, with an energy drawn from `spectrum` and a
            chirality drawn per `chirality`/`chirality_degree`.
        """

        # Re-sync in case these were changed after construction
        self._point_source.chirality = self.chirality
        self._point_source.chirality_degree = self.chirality_degree

        self._point_source.offaxis_angle = np.random.uniform(0,360)*u.deg

        return self._point_source.random_photon(detector = detector)
