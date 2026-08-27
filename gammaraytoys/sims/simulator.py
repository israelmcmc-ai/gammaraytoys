from histpy import Histogram, Axes, Axis
from astropy import units as u
import numpy as np
from .source import Source, FarFieldSource
from tqdm import tqdm

class Simulator:
    """
    Detector-frame photon simulator.

    Throws photons from a mix of sources directly at a fixed detector --
    the detector is the centre of the universe, nothing moves, and there is
    no notion of spacecraft pose or occultation. Sources are drawn with
    probability proportional to their `simulated_rate()`, and the run is
    sized (given a `duration`) from the sources' total simulated rate. See
    `InertialSimulator` (added in a later PR) for the pose-aware counterpart.
    """

    def __init__(self, detector, sources, reconstructor,
                 duration = None, nsim = None, ntrig = None,
                 doppler_broadening = True):
        """
        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector photons are thrown at and walked through.
        sources : `Source` or list of `Source`
            One source, or a list of sources to mix. Each source's
            `simulated_rate(detector)` sets both the overall event rate and
            its relative weight in source selection.
        reconstructor : `Reconstructor`
            Used to reconstruct each simulated event.
        duration : `astropy.units.Quantity`, optional
            If given, run for this simulated time (see `run_events`).
        nsim : int, optional
            If given, run for this many launched photons.
        ntrig : int, optional
            If given, run until this many triggers are recorded.
        doppler_broadening : bool
            Whether to apply the detector's energy-resolution Doppler
            broadening to the first interaction of each event (see
            `ToyTracker2D.simulate_event`).
        """

        self.detector = detector

        if isinstance(sources, Source):
            self.sources = [sources]
            self.total_rate = sources.simulated_rate(self.detector)
            self._relative_rate = [1]
        else:
            # Multiple sources
            self.sources = sources

            rates = u.Quantity([s.simulated_rate(self.detector) for s in self.sources])

            self.total_rate = np.sum(rates)
            self._relative_rate = (rates/self.total_rate).to_value('')

        self.reconstructor = reconstructor

        self.duration = 0*u.s
        self.nsim = 0
        self.ntrig = 0
        
        # Default, can be changed
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
        
        # Other
        self.doppler_broadening = doppler_broadening

    def _standarize_termination(self, nsim = None, ntrig = None, duration = None):
        """
        Convert whichever single finishing condition was given
        (`nsim`/`ntrig`/`duration`) into the full triple, filling the other
        two in as `np.inf` (unbounded) where they are not the driving
        condition.

        Parameters
        ----------
        nsim : int, optional
            Target number of launched photons.
        ntrig : int, optional
            Target number of triggers.
        duration : `astropy.units.Quantity`, optional
            Target simulated live time.

        Returns
        -------
        (int or None, int or float, int or float)
            `(nsim, ntrig, duration)`, standardized. `nsim` (or `duration`)
            comes back `None` instead of a number if `self.total_rate` is
            `None` (some source has no normalization set), since it cannot
            be computed in that case.
        """

        if np.sum([duration is not None,
                   nsim is not None,
                   ntrig is not None]) != 1:
            raise ValueError("Specify one and only one finishing condition")

        if duration is not None:
            if self.total_rate is not None:
                nsim = int(np.round((self.total_rate*duration).to_value('')))
            else:
                nsim = None
            # TBD after sims
            ntrig = np.inf

        elif nsim is not None:
            if self.total_rate is not None:
                duration = nsim/self.total_rate
            else:
                duration = None
            # TBD after sims
            ntrig = np.inf

        elif ntrig is not None:
            # TBD after sims
            nsim = np.inf
            duration = np.inf

        else:
            raise RuntimeError("This should happen")

        return nsim, ntrig, duration

    @property
    def total_flux(self):
        """
        Total flux (`1/cm/s`) summed over the simulator's far-field sources.

        Kept for backward compatibility with detector-frame-only runs, where
        every source is far-field and this reduces to "total sky-integrated
        flux". Returns `None` if any source in the run is a near-field
        source (whose normalization is a rate, not a flux) or has an unset
        (`None`) flux -- in which case use `total_rate` instead.
        """

        fluxes = []
        for source in self.sources:
            if not isinstance(source, FarFieldSource) or source.flux is None:
                return None
            fluxes.append(source.flux)

        return np.sum(u.Quantity(fluxes))

    @property
    def nsources(self):
        """Number of sources mixed into this simulator."""
        return len(self.sources)

    @property
    def measured_energy_axis(self):
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
        return self._phi_axis

    @phi_axis.setter
    def phi_axis(self, new):
        self._phi_axis = Axis(new, label = 'Phi')

    @property
    def psi_axis(self):
        return self._psi_axis

    @psi_axis.setter
    def psi_axis(self, new):
        self._psi_axis = Axis(new, label = 'Psi')

    @property
    def offaxis_angle_axis(self):
        return self._offaxis_angle_axis

    @offaxis_angle_axis.setter
    def offaxis_angle_axis(self, new):
        self._offaxis_angle_axis = Axis(new, label = 'Nu')

    @property
    def chirality_axis(self):
        return self._chirality_axis

    @chirality_axis.setter
    def chirality_axis(self, new):
        self._chirality_axis = Axis(new, label = 'k')

    @property
    def compton_data_axes(self):
        return Axes([self.measured_energy_axis,
                     self.phi_axis,
                     self.psi_axis])

    @property
    def photon_axes(self):
        return Axes([self.photon_energy_axis,
                     self.offaxis_angle_axis,
                     self.chirality_axis])

    @property
    def compton_axes(self):
        return Axes([self.photon_energy_axis,
                     self.offaxis_angle_axis,
                     self.chirality_axis,
                     self.measured_energy_axis,
                     self.phi_axis,
                     self.psi_axis])
        

    def run_binned(self, nsim = None, ntrig = None, duration = None,
                   axes = None, photon_axes = None):

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

        for sim_event, reco_event in self.run_events(nsim, ntrig, duration):

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
        
    def run_events(self, nsim = None, ntrig = None, duration = None):

        nsim_target, ntrig_target, duration_target = self._standarize_termination(nsim, ntrig, duration)
        
        nsim = 0
        ntrig = 0

        terminate = False

        with tqdm(total = nsim_target if np.isfinite(nsim_target) else ntrig_target) as pbar:
        
            while True:

                if terminate:
                    self.nsim += nsim
                    self.ntrig += ntrig
                    if self.total_rate is not None:
                        self.duration += (nsim/self.total_rate).to(u.s)
                    else:
                        self.duration = None

                    break

                nsim += 1

                source = self.sources[np.random.choice(range(self.nsources),
                                                       p = self._relative_rate)]

                primary = source.random_photon(self.detector)

                sim_event = self.detector.simulate_event(primary,
                                                         doppler_broadening = self.doppler_broadening)

                reco_event = self.reconstructor.reconstruct(sim_event)

                if np.isfinite(nsim_target) or reco_event.triggered:
                    pbar.update()
                
                if nsim >= nsim_target:
                    terminate = True

                if reco_event.triggered:
                    ntrig += 1

                    if ntrig >= ntrig_target:
                        terminate = True

                yield sim_event, reco_event

        
