from astropy import units as u
import numpy as np
from tqdm import tqdm

from .source import Source
from .simulator_base import SimulatorBase


class InertialSimulator(SimulatorBase):
    """
    Inertial-frame photon simulator: the sky stands still and the spacecraft
    moves.

    Where `Simulator` makes the detector the centre of the universe, this
    simulator walks a `SpacecraftHistory` interval by interval. In each
    interval the spacecraft's pose -- orbital radius, orbital angle and
    attitude -- is frozen, and every source is evaluated at that pose: a
    fixed-sky-angle source is re-aimed to whatever off-axis angle the attitude
    puts it at, and any photon arriving from behind the Earth is thrown away.

    The detector-frame physics is untouched and is reached *through* the
    sources' own coordinate transformations (see
    `gammaraytoys.sims.transform`), not by subclassing `Simulator`.

    Per interval and per source the photon count is drawn from a Poisson
    distribution with mean `simulated_rate(detector, pose) * livetime` --
    always Poisson, never a rounded expectation -- and each photon's timestamp
    is drawn uniformly over the interval's **full span**, not just its live
    fraction. Livetime only scales how many photons there are; spreading them
    over the whole span keeps deadtime distributed through the interval rather
    than parked as an artificial gap at its end.

    Occultation is a per-photon rejection drawn from the *unocculted* mean:
    `N` photons are drawn as if the Earth were not there, and each is then
    tested and discarded if blocked. For a point source this degenerates
    exactly to "the source is on or off for the whole interval"; unlike an
    analytic visible-fraction it also handles isotropic and extended sources
    with no bespoke truncation maths per source class.
    """

    def __init__(self, detector, sources, reconstructor, spacecraft_history,
                 earth, doppler_broadening = True):
        """
        Parameters
        ----------
        detector : `ToyTracker2D`
            The detector photons are thrown at and walked through.
        sources : `Source` or list of `Source`
            One source, or a list of sources to mix. Every source must be
            normalized (have a flux or a rate set), and every far-field
            source must be aimed on the inertial sky rather than at a fixed
            detector-frame off-axis angle -- an `offaxis_angle` source would
            silently ignore the spacecraft's attitude and sit welded to the
            detector while the sky rotated past it.
        reconstructor : `Reconstructor`
            Used to reconstruct each simulated event.
        spacecraft_history : `SpacecraftHistory`
            The orbit, attitude and livetime history to simulate over. Its
            intervals define the run: iterating it already excludes the
            terminator row.
        earth : `Earth`
            The Earth used for occultation. Required, and deliberately not
            defaulted: this simulator passes it explicitly to every
            `source.random_photon` call it makes, rather than reading it off
            the pose, so that a history built against one Earth and a
            simulation run against another cannot silently disagree about
            what is blocked. Checked at construction (see `_validate_earth`)
            against every `orbit_radius` in `spacecraft_history`, since an
            inconsistent pair would otherwise disagree silently: `Earth`'s
            hot occultation path is deliberately unvalidated, so a
            mismatched Earth would turn into `nan`s and switch occultation
            off for the whole run without raising anything.
        doppler_broadening : bool
            Whether to apply the detector's energy-resolution Doppler
            broadening to the first interaction of each event (see
            `ToyTracker2D.simulate_event`).

        Raises
        ------
        ValueError
            If `earth`'s radius does not leave every interval's
            `orbit_radius` in `spacecraft_history` strictly above it (see
            `_validate_earth`); if any source has no normalization set (its
            `simulated_rate()` is `None`); or if any far-field source is
            aimed by a detector-frame `offaxis_angle` instead of a
            `sky_angle`. The latter two name the offending source: without
            the normalization check the missing normalization would only
            surface much later, as a `TypeError` deep inside the Poisson
            mean.
        """

        super().__init__(detector = detector,
                         reconstructor = reconstructor,
                         doppler_broadening = doppler_broadening)

        self.sources = [sources] if isinstance(sources, Source) else list(sources)
        self.spacecraft_history = spacecraft_history
        self.earth = earth

        self._validate_earth()
        self._validate_sources()

        self.nsim = 0
        self.ntrig = 0
        self.noccult = 0

    def _validate_earth(self):
        """
        Check that `self.earth` agrees with every `orbit_radius` in
        `self.spacecraft_history`, and raise if not.

        `self.earth` is taken separately from `spacecraft_history` (see the
        `earth` parameter above) precisely so the two can disagree -- a
        history built against one `Earth` and a run against another. Nothing
        else catches that: `Earth._is_occulted`, the per-photon hot path, is
        deliberately unvalidated (Section 4.5's `_angular_radius_rad`), so a
        mismatched Earth whose radius exceeds some interval's `orbit_radius`
        would make `arcsin(R_E / r)` a silent `nan`, `abs(delta) < nan` a
        silent `False` for every photon, and occultation would vanish from
        the run without so much as an exception -- only a numpy
        `RuntimeWarning` deep in the loop.

        This walks every interval once, up front, rather than leaving the
        check to fire on whichever photon happens to hit it first (it might
        never fire at all, since `_is_occulted` is never wrong loudly --
        only silently).

        Raises
        ------
        ValueError
            If any interval's `orbit_radius` does not exceed `self.earth`'s
            radius. Delegates to `Earth.angular_radius`, which already
            raises with a clear message naming both radii -- there is no
            new error text to maintain here.
        """

        min_radius = None

        for interval in self.spacecraft_history:
            if min_radius is None or interval.orbit_radius < min_radius:
                min_radius = interval.orbit_radius

        if min_radius is not None:
            # Raises ValueError, with a message naming both radii, if
            # min_radius does not exceed self.earth.radius.
            self.earth.angular_radius(min_radius)

    def _validate_sources(self):
        """
        Check every source is usable in an inertial run, and raise naming the
        offender if not.

        Two things can go wrong, and both are much cheaper to catch here than
        halfway through a run:

        - a source with no flux or rate set, whose `simulated_rate()` is
          `None` and would otherwise blow up as a `TypeError` inside the
          Poisson mean, thousands of photons into the run;
        - a far-field source aimed by a fixed detector-frame `offaxis_angle`,
          which would quietly ignore the spacecraft's attitude entirely.

        Raises
        ------
        ValueError
            Naming the offending source, its index in `self.sources`, and
            which of the two problems it has.
        """

        # A pose the sources can actually be evaluated at: the flux of a
        # far-field source is pose-independent today, but the Earth albedo
        # (a later PR) has a flux that depends on the orbital radius, and
        # would have no answer at all for `pose = None`.
        first_pose = next(iter(self.spacecraft_history), None)

        for i, source in enumerate(self.sources):

            label = f"sources[{i}] ({type(source).__name__})"

            # A far-field source aimed in the detector frame has an
            # `offaxis_angle` and no `sky_angle`. A source with neither
            # (e.g. IsotropicSource, which covers the whole sky, or a
            # near-field source, which lives at a detector-frame position by
            # definition) is fine.
            if (getattr(source, 'sky_angle', None) is None
                    and getattr(source, 'offaxis_angle', None) is not None):
                raise ValueError(
                    f"{label} is aimed at a fixed detector-frame off-axis "
                    f"angle ({source.offaxis_angle}), so it would stay welded "
                    "to the detector while the spacecraft rotated underneath "
                    "it. A source in an InertialSimulator must be placed on "
                    "the inertial sky: build it with `sky_angle` instead of "
                    "`offaxis_angle`.")

            if source.simulated_rate(self.detector, first_pose) is None:
                raise ValueError(
                    f"{label} has no normalization set, so there is no rate "
                    "to draw a photon count from. Give it a `flux` (far-field) "
                    "or a `rate` (near-field) before simulating.")

    @property
    def nsources(self):
        """Number of sources mixed into this simulator."""
        return len(self.sources)

    def _poses(self, tstart = None, tstop = None):
        """
        Walk the spacecraft history, yielding one entry per interval that
        contributes to the run.

        The `.ori` file's own time range is the run's extent; `tstart` and
        `tstop` only narrow it. An interval that overlaps the requested window
        only partly is clipped to the overlap: its livetime is scaled by the
        overlapping fraction of its span, and its photons are timestamped
        within the overlap. With no window given nothing is clipped and this
        is exactly "every interval, whole".

        Parameters
        ----------
        tstart, tstop : `astropy.units.Quantity` or None
            Time window to narrow the run to (time units). `None` means "no
            narrowing at that end".

        Yields
        ------
        (`SpacecraftInterval`, float, float, float)
            `(pose, start_s, stop_s, livetime_s)`: the interval's pose,
            followed by the clipped start time, stop time and livetime as
            plain floats in seconds. `pose` alone does not carry an Earth --
            see `self.earth`, passed separately to every `random_photon`
            call in `run_events`.
        """

        window_start = -np.inf if tstart is None else tstart.to_value(u.s)
        window_stop = np.inf if tstop is None else tstop.to_value(u.s)

        for interval in self.spacecraft_history:

            start = interval.start_time.to_value(u.s)
            stop = interval.stop_time.to_value(u.s)

            lo = max(start, window_start)
            hi = min(stop, window_stop)

            if hi <= lo:
                continue

            # Timestamps are spread over the whole (clipped) span, while the
            # livetime -- which only scales the count -- is scaled by the same
            # fraction of the interval that survived the clip.
            live = interval.livetime.to_value(u.s) * (hi - lo) / (stop - start)

            yield interval, lo, hi, live

    def _expected_counts(self, tstart = None, tstop = None):
        """
        Total expected number of photons launched at the detector over the
        run, summed over every interval and source.

        This is what the progress bar counts against: with a Poisson draw per
        (source, interval) the run's total is not knowable from a photon count
        alone, so it has to be summed up front. It is the *unocculted*
        expectation, matching the means the Poisson draws are taken from --
        the occulted photons are drawn and then rejected, so they are counted
        here too.

        Parameters
        ----------
        tstart, tstop : `astropy.units.Quantity` or None
            Time window narrowing the run, as in `run_events`.

        Returns
        -------
        float
            Expected number of photons, dimensionless.
        """

        total = 0.0

        for pose, _, _, livetime in self._poses(tstart, tstop):
            for source in self.sources:
                total += source.simulated_rate(self.detector, pose).to_value(u.Hz) * livetime

        return total

    def run_events(self, tstart = None, tstop = None):
        """
        Walk the spacecraft history and yield one event per photon actually
        launched at the detector.

        For every interval, and every source in it, the expected count
        `mu = simulated_rate(detector, pose) * livetime` is formed and `N` is
        drawn from `Poisson(mu)`. Each of those `N` photons gets a timestamp
        uniform over the interval's full span and is drawn from the source at
        that pose; a photon the Earth blocks is silently dropped and never
        yielded, since it was never launched at all.

        Parameters
        ----------
        tstart : `astropy.units.Quantity` or None
            Start of the time window to simulate (time units). `None` (the
            default) starts at the beginning of the spacecraft history.
        tstop : `astropy.units.Quantity` or None
            End of the time window to simulate (time units). `None` (the
            default) runs to the end of the spacecraft history. Intervals
            overlapping the window only partly are clipped to the overlap
            (see `_poses`).

        Yields
        ------
        (`astropy.units.Quantity`, `Source`, `Photon`, `RecoEvent`)
            `(time, source, sim_event, reco_event)`: the photon's timestamp
            in seconds, the source that emitted it, the simulated event (its
            `hits` carry what the detector recorded) and its reconstruction.
            Check `reco_event.triggered` to filter to triggers only.
        """

        nsim = 0
        ntrig = 0
        noccult = 0

        with tqdm(total = self._expected_counts(tstart, tstop)) as pbar:

            for pose, start, stop, livetime in self._poses(tstart, tstop):

                for source in self.sources:

                    mu = source.simulated_rate(self.detector, pose).to_value(u.Hz) * livetime

                    for _ in range(np.random.poisson(mu)):

                        # Uniform over the full span, not just the live part:
                        # livetime only scales the count.
                        time = np.random.uniform(start, stop)

                        pbar.update()

                        event = self._simulate_one(source, pose, self.earth)

                        if event is None:
                            # Occulted by the Earth before it was ever thrown.
                            noccult += 1
                            continue

                        sim_event, reco_event = event

                        nsim += 1
                        if reco_event.triggered:
                            ntrig += 1

                        yield time * u.s, source, sim_event, reco_event

        self.nsim += nsim
        self.ntrig += ntrig
        self.noccult += noccult

    def run_binned(self, axes = None, photon_axes = None,
                   tstart = None, tstop = None):
        """
        Run `run_events` to completion, filling reconstructed (and,
        optionally, thrown-photon) histograms instead of yielding events.

        Every launched photon is walked through the run; only triggered
        events are filled into `h_data`. Occulted photons appear in neither
        histogram -- they were never launched.

        Parameters
        ----------
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
            additionally binned jointly over the requested photon axes, and a
            second histogram `h_sim` records every *launched* photon
            (triggered or not) over the requested photon axes alone. Note
            that 'Nu' is the photon's off-axis angle in the detector frame at
            the pose it was thrown from, not its fixed inertial sky angle.
        tstart : `astropy.units.Quantity` or None
            Start of the time window to simulate. See `run_events`.
        tstop : `astropy.units.Quantity` or None
            End of the time window to simulate. See `run_events`.

        Returns
        -------
        `histpy.Histogram` or (`histpy.Histogram`, `histpy.Histogram`)
            `h_data` alone if `photon_axes` is `None`/`False`; otherwise
            `(h_data, h_sim)`.
        """

        events = ((sim_event, reco_event)
                  for _, _, sim_event, reco_event in self.run_events(tstart, tstop))

        return self._run_binned(events, axes = axes, photon_axes = photon_axes)
