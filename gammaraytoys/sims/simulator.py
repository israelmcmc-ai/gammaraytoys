from astropy import units as u
import numpy as np
from .source import Source
from .simulator_base import SimulatorBase
from tqdm import tqdm

class Simulator(SimulatorBase):
    """
    Detector-frame photon simulator.

    Throws photons from a mix of sources directly at a fixed detector --
    the detector is the centre of the universe, nothing moves, and there is
    no notion of spacecraft pose or occultation. Sources are drawn with
    probability proportional to their `simulated_rate()`, and the run is
    sized (given a `duration`) from the sources' total simulated rate. See
    `InertialSimulator` for the pose-aware counterpart.
    """

    def __init__(self, detector, sources, reconstructor,
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
        doppler_broadening : bool
            Whether to apply the detector's energy-resolution Doppler
            broadening to the first interaction of each event (see
            `ToyTracker2D.simulate_event`).
        """

        super().__init__(detector = detector,
                         reconstructor = reconstructor,
                         doppler_broadening = doppler_broadening)

        if isinstance(sources, Source):
            self.sources = [sources]
            self.total_simulated_rate = sources.simulated_rate(self.detector)
            self._relative_rate = [1]
        else:
            # Multiple sources
            self.sources = sources

            rates = u.Quantity([s.simulated_rate(self.detector) for s in self.sources])

            self.total_simulated_rate = np.sum(rates)
            self._relative_rate = (rates/self.total_simulated_rate).to_value('')

        self.duration = 0*u.s
        self.nsim = 0
        self.ntrig = 0

    def _standardize_termination(self, nsim = None, ntrig = None, duration = None):
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
        (int or float or None, int or float, `astropy.units.Quantity` or float or None)
            `(nsim, ntrig, duration)`, standardized. Whichever of the three
            was not the driving condition comes back as `np.inf`
            (unbounded) -- a plain, unitless float, even for `duration`.
            `nsim` (or `duration`) comes back `None` instead of a number if
            `self.total_simulated_rate` is `None` (some source has no normalization
            set), since it cannot be computed in that case. Otherwise
            `duration` is an `astropy.units.Quantity` in time units (the
            given, or rate-derived, live time) and `nsim`/`ntrig` are `int`
            (the given, or rate-derived, target).
        """

        if np.sum([duration is not None,
                   nsim is not None,
                   ntrig is not None]) != 1:
            raise ValueError("Specify one and only one finishing condition")

        if duration is not None:
            if self.total_simulated_rate is not None:
                nsim = int(np.round((self.total_simulated_rate*duration).to_value('')))
            else:
                nsim = None
            # TBD after sims
            ntrig = np.inf

        elif nsim is not None:
            if self.total_simulated_rate is not None:
                duration = nsim/self.total_simulated_rate
            else:
                duration = None
            # TBD after sims
            ntrig = np.inf

        elif ntrig is not None:
            # TBD after sims
            nsim = np.inf
            duration = np.inf

        else:
            raise RuntimeError("This should not happen")

        return nsim, ntrig, duration

    @property
    def nsources(self):
        """Number of sources mixed into this simulator."""
        return len(self.sources)

    def run_binned(self, nsim = None, ntrig = None, duration = None,
                   axes = None, photon_axes = None):
        """
        Run `run_events` to completion, filling reconstructed (and,
        optionally, thrown-photon) histograms instead of yielding events.

        `nsim`, `ntrig` and `duration` are forwarded to `run_events`
        unchanged -- see its docstring for the finishing conditions and for
        how a `duration` is converted to a photon count up front rather
        than tracked as elapsed time. Every launched photon is walked
        through the run; only triggered events are filled into `h_data`.

        Parameters
        ----------
        nsim : int, optional
            Stop after launching this many photons. See `run_events`.
        ntrig : int, optional
            Stop after recording this many triggers. See `run_events`.
        duration : `astropy.units.Quantity`, optional
            Stop after launching the number of photons expected in this
            much simulated live time. See `run_events`.
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

        return self._run_binned(self.run_events(nsim, ntrig, duration),
                                axes = axes, photon_axes = photon_axes)

    def run_events(self, nsim = None, ntrig = None, duration = None):
        """
        Throw photons at the detector and yield one (simulated, reconstructed)
        pair per launched photon, until a finishing condition is met.

        Exactly one of `nsim`, `ntrig`, `duration` must be given; it sets the
        finishing condition. Under the hood a `duration` is converted to a
        target `nsim` up front -- `nsim = round(total_simulated_rate * duration)`,
        via `self.total_simulated_rate` -- rather than tracked as elapsed simulated
        time while the loop runs: the run always finishes on a launched- or
        triggered-photon count, never on a live-time check mid-run. This
        matters if `total_simulated_rate` changes between calls (e.g. the source list
        is edited): a `duration` given here is translated to a photon count
        using the rate *at call time*, not re-evaluated as photons are
        thrown.

        For each photon launched (whether or not it triggers), one event is
        drawn from a source (selected with probability proportional to its
        `simulated_rate`), walked through `self.detector`, and reconstructed.
        The generator yields for every launched photon; check
        `reco_event.triggered` to filter to triggers only.

        Finishing conditions:

        - `nsim` given (directly, or derived from `duration`): stop once
          this many photons have been launched.
        - `ntrig` given: stop once this many *triggered* events have been
          recorded (photons keep launching, and are yielded, even though
          they do not count toward this target unless they trigger).

        After the run, `self.nsim`, `self.ntrig` and `self.duration` are
        incremented by this run's totals (`self.duration` again derived
        from the launched-photon count and `self.total_simulated_rate`, not measured
        -- `None` if `self.total_simulated_rate` is `None`).

        Parameters
        ----------
        nsim : int, optional
            Stop after launching this many photons.
        ntrig : int, optional
            Stop after recording this many triggers.
        duration : `astropy.units.Quantity`, optional
            Stop after launching the number of photons expected in this
            much simulated live time, given `self.total_simulated_rate` (time units).

        Yields
        ------
        (`Photon`, `RecoEvent`)
            `(sim_event, reco_event)` for every launched photon: the photon
            returned by `self.detector.simulate_event` (its `hits` carry
            what the detector recorded) and its reconstruction from
            `self.reconstructor`. Check `reco_event.triggered`.
        """

        nsim_target, ntrig_target, duration_target = self._standardize_termination(nsim, ntrig, duration)
        
        nsim = 0
        ntrig = 0

        terminate = False

        with tqdm(total = nsim_target if np.isfinite(nsim_target) else ntrig_target) as pbar:
        
            while True:

                if terminate:
                    self.nsim += nsim
                    self.ntrig += ntrig
                    if self.total_simulated_rate is not None:
                        self.duration += (nsim/self.total_simulated_rate).to(u.s)
                    else:
                        self.duration = None

                    break

                nsim += 1

                source = self.sources[np.random.choice(range(self.nsources),
                                                       p = self._relative_rate)]

                sim_event, reco_event = self._simulate_one(source)

                if np.isfinite(nsim_target) or reco_event.triggered:
                    pbar.update()
                
                if nsim >= nsim_target:
                    terminate = True

                if reco_event.triggered:
                    ntrig += 1

                    if ntrig >= ntrig_target:
                        terminate = True

                yield sim_event, reco_event

        
