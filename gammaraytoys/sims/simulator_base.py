from histpy import Histogram, Axes, Axis
from astropy import units as u
import numpy as np


class SimulatorBase:
    """
    The machinery shared by `Simulator` (detector-frame) and
    `InertialSimulator` (inertial-frame).

    The two simulators differ only in how they decide *which* source throws a
    photon *when*: the detector-frame one draws a source at random until a
    photon or trigger count is reached, while the inertial one walks a
    spacecraft history interval by interval and draws a Poisson count per
    source. Everything downstream of that decision is identical -- drawing the
    photon, walking it through the detector, reconstructing it, and binning
    the result -- and lives here so the two cannot drift apart.

    This is deliberately *not* a physics base class: `InertialSimulator` does
    not reach the detector-frame physics by subclassing `Simulator`, it reaches
    it through the source transformations (Section 6 of the plan). What is
    shared here is bookkeeping -- the Compton Data Space axes and the
    per-photon simulate/reconstruct/fill sequence.
    """

    def __init__(self, detector, reconstructor, doppler_broadening = True):
        """
        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector photons are thrown at and walked through.
        reconstructor : `Reconstructor`
            Used to reconstruct each simulated event.
        doppler_broadening : bool
            Whether to apply the detector's energy-resolution Doppler
            broadening to the first interaction of each event (see
            `ToyTracker2D.simulate_event`).
        """

        self.detector = detector
        self.reconstructor = reconstructor
        self.doppler_broadening = doppler_broadening

        # Defaults, can be changed
        self._photon_energy_axis = Axis(np.geomspace(.2,50,200)*u.MeV,
                                          label = 'Ei',
                                          scale = 'log')
        self._offaxis_angle_axis = Axis(np.linspace(-180, 180, 360)*u.deg, label = 'Nu')
        self._chirality_axis = Axis([-2,0,2], label = 'k')
        self._measured_energy_axis = Axis(np.geomspace(.1,60,200)*u.MeV,
                                          label = 'Em',
                                          scale = 'log')
        self._phi_axis = Axis(np.linspace(0,180, 180)*u.deg, label = 'Phi')
        self._psi_axis = Axis(np.linspace(-180,180, 360)*u.deg, label = 'Psi')

    @property
    def measured_energy_axis(self):
        """`histpy.Axis`: measured (reconstructed) energy binning, label 'Em'."""
        return self._measured_energy_axis

    @measured_energy_axis.setter
    def measured_energy_axis(self, new):
        # Do not change scale
        self._measured_energy_axis = Axis(new, label = 'Em',
                                          scale = (new.axis_scale
                                                   if isinstance(new, Axis)
                                                   else
                                                   'log'))

    @property
    def photon_energy_axis(self):
        """`histpy.Axis`: thrown-photon energy binning, label 'Ei'."""
        return self._photon_energy_axis

    @photon_energy_axis.setter
    def photon_energy_axis(self, new):
        # Do not change scale
        self._photon_energy_axis = Axis(new, label = 'Ei',
                                          scale = (new.axis_scale
                                                   if isinstance(new, Axis)
                                                   else
                                                   'log'))

    @property
    def phi_axis(self):
        """`histpy.Axis`: Compton scattering angle binning, label 'Phi'."""
        return self._phi_axis

    @phi_axis.setter
    def phi_axis(self, new):
        self._phi_axis = Axis(new, label = 'Phi')

    @property
    def psi_axis(self):
        """`histpy.Axis`: Compton scatter direction binning, label 'Psi'."""
        return self._psi_axis

    @psi_axis.setter
    def psi_axis(self, new):
        self._psi_axis = Axis(new, label = 'Psi')

    @property
    def offaxis_angle_axis(self):
        """`histpy.Axis`: thrown-photon off-axis angle binning, label 'Nu'."""
        return self._offaxis_angle_axis

    @offaxis_angle_axis.setter
    def offaxis_angle_axis(self, new):
        self._offaxis_angle_axis = Axis(new, label = 'Nu')

    @property
    def chirality_axis(self):
        """`histpy.Axis`: thrown-photon chirality binning, label 'k'."""
        return self._chirality_axis

    @chirality_axis.setter
    def chirality_axis(self, new):
        self._chirality_axis = Axis(new, label = 'k')

    @property
    def compton_data_axes(self):
        """`histpy.Axes`: the reconstructed Compton Data Space axes
        ('Em', 'Phi', 'Psi')."""
        return Axes([self.measured_energy_axis,
                     self.phi_axis,
                     self.psi_axis])

    @property
    def photon_axes(self):
        """`histpy.Axes`: the thrown-photon axes ('Ei', 'Nu', 'k')."""
        return Axes([self.photon_energy_axis,
                     self.offaxis_angle_axis,
                     self.chirality_axis])

    @property
    def compton_axes(self):
        """`histpy.Axes`: the thrown-photon axes followed by the reconstructed
        Compton Data Space axes ('Ei', 'Nu', 'k', 'Em', 'Phi', 'Psi')."""
        return Axes([self.photon_energy_axis,
                     self.offaxis_angle_axis,
                     self.chirality_axis,
                     self.measured_energy_axis,
                     self.phi_axis,
                     self.psi_axis])

    def _simulate_one(self, source, pose = None, earth = None):
        """
        Draw one photon from `source`, walk it through the detector and
        reconstruct it.

        This is the whole per-photon sequence, shared by both simulators so
        that neither can quietly grow its own variant of it.

        Parameters
        ----------
        source : `Source`
            The source to draw the photon from.
        pose : `SpacecraftInterval` or None
            Spacecraft pose, forwarded to `source.random_photon`. `None`
            (the default) is pure detector-frame mode, in which no source
            can be occulted and this method therefore never returns `None`.
        earth : `Earth` or None
            The Earth to test occultation against, forwarded to
            `source.random_photon`. Ignored when `pose` is `None`; required
            (raises otherwise) for an occultable far-field source given a
            `pose` (see `FarFieldSource._occulted`).

        Returns
        -------
        (`Photon`, `RecoEvent`) or None
            The simulated event (its `hits` carry what the detector
            recorded) and its reconstruction, or `None` if the source was
            occulted at this pose and no photon was launched at all.
        """

        primary = source.random_photon(self.detector, pose, earth)

        if primary is None:
            # Occulted: nothing was ever launched at the detector.
            return None

        sim_event = self.detector.simulate_event(primary,
                                                 doppler_broadening = self.doppler_broadening)

        reco_event = self.reconstructor.reconstruct(sim_event)

        return sim_event, reco_event

    def _run_binned(self, events, axes = None, photon_axes = None):
        """
        Consume a stream of simulated events and fill reconstructed (and,
        optionally, thrown-photon) histograms from it.

        The shared body of both simulators' `run_binned`; they differ only in
        the `events` stream they hand it.

        Parameters
        ----------
        events : iterable of (`Photon`, `RecoEvent`)
            One `(sim_event, reco_event)` pair per photon actually launched
            at the detector. Occulted photons are never launched and so must
            not appear here.
        axes : str or list of str, optional
            Which of the reconstructed Compton Data Space axes ('Em', 'Phi',
            'Psi') to bin `h_data` over. Defaults to all three
            (`self.compton_data_axes`).
        photon_axes : bool, str or list of str, optional
            Whether to also bin the *thrown* photons, and over which of
            'Ei', 'Nu', 'k'. `None` or `False` (the default) skips this
            entirely and only `h_data` is returned. `True` bins over all
            three (`self.photon_axes`); a str or list of str bins over that
            subset. Whenever this is not `None`/`False`, `h_data` is
            additionally binned jointly over the requested photon axes (so
            each reconstructed bin can be sliced by the thrown quantities
            that produced it), and a second histogram `h_sim` records every
            *launched* photon (triggered or not) over the requested photon
            axes alone.

        Returns
        -------
        `histpy.Histogram` or (`histpy.Histogram`, `histpy.Histogram`)
            `h_data` alone if `photon_axes` is `None`/`False`; otherwise
            `(h_data, h_sim)`.
        """

        if axes is None:
            data_axes = self.compton_data_axes
        else:
            if isinstance(axes, str):
                axes = [axes]
            data_axes = self.compton_data_axes[axes]

        if isinstance(photon_axes, str):
                photon_axes = [photon_axes]

        if photon_axes is True:
            sim_hist = True
            photon_axes = self.photon_axes
        elif photon_axes is not False and photon_axes is not None:
            sim_hist = True
            photon_axes = self.photon_axes[photon_axes]
        else:
            sim_hist = False

        if sim_hist:
            h_data = Histogram(list(photon_axes) + list(data_axes))
            h_sim = Histogram(photon_axes)
        else:
            h_data = Histogram(data_axes)

        for sim_event, reco_event in events:

            if sim_hist:
                photon_data = {'Ei': sim_event.energy,
                               'Nu': 270*u.deg - sim_event.direction,
                               'k': sim_event.chirality}

                photon_data = [photon_data[k] for k in photon_axes.labels]

                h_sim.fill(*photon_data)

            if reco_event.triggered:

                reco_data = {'Em': reco_event.energy,
                             'Phi': reco_event.phi,
                             'Psi': reco_event.psi}

                reco_data = [reco_data[k] for k in data_axes.labels]

                if sim_hist:
                    reco_data = photon_data + reco_data

                h_data.fill(*reco_data)

        if sim_hist:
            return h_data, h_sim
        else:
            return h_data
