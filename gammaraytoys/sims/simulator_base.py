from collections.abc import Sequence

from histpy import Histogram, Axes, Axis
from astropy import units as u
from astropy.coordinates import Angle
import numpy as np

from .config import load_config, _config_base_dir
from .config_utils import _as_mapping, _boolean, _check_keys, _integer, _text
from .earth import Earth
from .reco import Reconstructor, SimpleTraditionalReconstructor
from .source import Source
from .spacecraft_history import SpacecraftHistory


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

    Attributes
    ----------
    source_names : dict or None
        Maps each named `Source` in this run to its name, in the shape
        `write_event_csv`'s own `source_names` argument wants. Set by
        `from_config` from the `name` key of each source block; `None` for
        a simulator built directly, which labels events by class name.
    random_seed : int or None
        The seed `from_config` seeded numpy's global generator with, kept
        so that a run can say how it was seeded. `None` when no seed was
        given.

    Both carry a class-level default so that a simulator built directly --
    or a third-party subclass that never runs this `__init__` -- still has
    them rather than raising `AttributeError` from inside `write_event_csv`
    or `to_config`.
    """

    source_names = None
    random_seed = None

    # The `earth` block a configuration named, for `to_config`. Only the
    # inertial simulator has an `earth` of its own; the detector-frame one
    # may still have been given one, for an `EarthAlbedoSource` to emit
    # from, and would otherwise lose it on the way back out.
    _config_earth = None

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

    @classmethod
    def _from_config(cls, config, inertial):
        """
        Build a simulator from a configuration -- the body of both
        `Simulator.from_config` and `InertialSimulator.from_config`, which
        differ only in the `inertial` flag and in what they say about it.

        Parameters
        ----------
        config : str, path-like or mapping
            A path to a YAML file, or an already-parsed mapping. Relative
            paths the configuration names for files beside it -- an `.ori`
            spacecraft history, a `TabulatedScaling`'s CSV -- are taken
            relative to the directory of that YAML file, so a configuration in
            `runs/` can be loaded from anywhere. A mapping came from no file
            and has no directory, so relative paths in one fall back to the
            working directory. Absolute paths are used as written, always.
        inertial : bool
            Whether `cls` is the inertial simulator, and so takes (and
            requires) a `spacecraft_history` and an `earth`. Passed by the
            subclass rather than read off `cls`: which of the two is calling is
            a fact each one already knows about itself.

        Returns
        -------
        `SimulatorBase`
            An instance of `cls`, carrying `source_names` and `random_seed`,
            and the provenance `to_config` needs.

        Raises
        ------
        ValueError
            On any unknown or missing key, any bad value, or anything the
            simulator itself rejects at construction (an unnormalized source,
            a far-field source aimed in the detector frame, an Earth that
            disagrees with the orbit).
        TypeError
            If `config` is neither a path nor a mapping.
        """

        where = 'config'
        block = load_config(config)

        # Where a relative sibling path in this configuration points. `None`
        # for a mapping, which has no file and so no directory of its own.
        base_dir = _config_base_dir(config)

        allowed = _COMMON_TOP_KEYS + (('spacecraft_history',) if inertial else ())

        if not inertial and 'spacecraft_history' in block:
            raise ValueError(
                f"{where}: 'spacecraft_history' belongs to "
                f"InertialSimulator.from_config -- the detector-frame Simulator "
                f"has no spacecraft, no orbit and no occultation. Use "
                f"InertialSimulator.from_config for a configuration with a "
                f"spacecraft history.")

        required = ('detector', 'sources') + (('spacecraft_history',) if inertial else ())
        _check_keys(block, where, allowed, required = required)

        # The Earth first: it is shared by the history, by TargetedPointing and
        # by EarthAlbedoSource, and every one of them must get *this* one. A run
        # with two different planets is a bug this project has shipped before.
        earth = (Earth.from_config(block['earth'], f"{where}.earth")
                 if 'earth' in block else Earth())

        # Imported here, not at module level: `gammaraytoys.detectors` imports
        # `gammaraytoys.sims` for `Photon`, so a module-level import would close
        # a cycle and break `import gammaraytoys` outright.
        from ..detectors import ToyTracker2D

        detector = ToyTracker2D.from_config(block['detector'], f"{where}.detector")

        reconstructor = (Reconstructor.from_config(block['reconstructor'],
                                                   f"{where}.reconstructor")
                         if 'reconstructor' in block
                         else SimpleTraditionalReconstructor())

        doppler_broadening = _boolean(block, 'doppler_broadening', where, default = True)

        seed = _integer(block, 'random_seed', where, minimum = 0, maximum = _MAX_SEED)

        sources, names = cls._sources_from_config(
            block['sources'], f"{where}.sources", earth, base_dir)

        kwargs = {'detector': detector,
                  'sources': sources,
                  'reconstructor': reconstructor,
                  'doppler_broadening': doppler_broadening}

        history_block = None

        if inertial:
            history, history_block = SpacecraftHistory._from_config_and_block(
                block['spacecraft_history'], f"{where}.spacecraft_history", earth,
                base_dir)
            kwargs['spacecraft_history'] = history
            kwargs['earth'] = earth

        try:
            simulator = cls(**kwargs)
        except Exception as err:
            raise ValueError(
                f"{where}: could not build a {cls.__name__} from this "
                f"configuration ({type(err).__name__}: {err}).") from err

        simulator.source_names = names
        simulator.random_seed = seed
        simulator._config_earth = earth if 'earth' in block else None

        if inertial:
            simulator._config_spacecraft_history = history_block

        # Seeded last, after every object is built, so that what follows the
        # call -- the run itself -- always starts from the same RNG state, no
        # matter what construction did or did not draw. Two `from_config` calls
        # with the same seed, each followed by a run, give byte-identical runs.
        if seed is not None:
            np.random.seed(seed)

        return simulator
    def _to_config(self):
        """
        Write this simulator back out as a configuration -- the body of both
        `Simulator.to_config` and `InertialSimulator.to_config`, which differ
        only in what they say about it.

        Its detector, sources, reconstructor and Earth are read off the objects
        themselves; the source names and the spacecraft history's provenance --
        neither of which the objects keep -- come from what `from_config`
        recorded.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            A configuration `from_config` reads back into an equal simulator,
            holding only plain strings, numbers, lists and dictionaries, so it
            can be written straight out with `yaml.safe_dump`.

        Raises
        ------
        ValueError
            If any part of this simulator is not something a configuration can
            describe, or if it has a spacecraft history but was not built by
            `from_config` and so cannot say where that history came from.
        """

        config = {'detector': self.detector.to_config()}

        earth = getattr(self, 'earth', None)

        if earth is None:
            earth = self._config_earth

        if earth is not None:
            config['earth'] = earth.to_config()

        config['reconstructor'] = self.reconstructor.to_config()

        if not self.doppler_broadening:
            config['doppler_broadening'] = False

        if hasattr(self, 'spacecraft_history'):
            history_block = self._config_spacecraft_history

            if history_block is None:
                raise ValueError(
                    "this simulator's spacecraft history was not built by "
                    "`from_config`, so there is nothing to write for it: a "
                    "generated `SpacecraftHistory` keeps its sampled rows, not "
                    "the orbital elements that produced them, and one read from "
                    "a file does not remember the file.")

            config['spacecraft_history'] = history_block

        if self.random_seed is not None:
            config['random_seed'] = self.random_seed

        names = self.source_names or {}

        config['sources'] = [source.to_config(names.get(source))
                             for source in self.sources]

        return config
    @staticmethod
    def _sources_from_config(config, where, earth, base_dir = None):
        """
        Build every source in the `sources` list, and collect their names.

        Parameters
        ----------
        config : sequence
            The `sources` list.
        where : str
            Label for the list in error messages.
        earth : `Earth`
            The run's single Earth, handed to every source that needs one.
        base_dir : `pathlib.Path` or None
            The directory a relative path inside a source's `scaling` block
            resolves against. See `SourceScaling.from_config`.

        Returns
        -------
        sources : list of `Source`
            The sources, in the order they were written.
        names : dict
            Maps each named `Source` to its name, in the shape
            `write_event_csv`'s `source_names` argument wants. Sources with no
            `name` key are absent from it, and fall back to their class name in
            an event file.

        Raises
        ------
        ValueError
            If the list is missing or empty, if a name is not a non-empty
            string, if a name contains `'#'` (which `write_event_csv` refuses,
            because the reader would treat it as the start of a comment and
            silently drop the rest of the row), if two sources share a name, or
            on anything `Source.from_config` rejects.
        """

        if (not isinstance(config, Sequence) or isinstance(config, (str, bytes))
                or len(config) == 0):
            raise ValueError(
                f"{where}: must be a non-empty list of source blocks, got "
                f"{config!r}.")

        sources = []
        names = {}

        for index, item in enumerate(config):
            block = _as_mapping(item, f"{where}[{index}]")

            # The source's own name goes into the label, so that a mistake in
            # the third of five sources says which one it was.
            label = f"{where}[{index}]"
            if isinstance(block.get('name'), str):
                label = f"{label} ({block['name']})"

            source = Source.from_config(block, label, earth, base_dir)
            sources.append(source)

            if 'name' in block:
                name = _text(block, 'name', label, required = True)

                if not name:
                    raise ValueError(f"{label}: key 'name' must not be empty.")

                if '#' in name:
                    raise ValueError(
                        f"{label}: key 'name' = {name!r} must not contain '#'. "
                        f"`write_event_csv` refuses such a label: its reader "
                        f"treats '#' as the start of a comment and would silently "
                        f"drop the rest of every row this source wrote.")

                if name in names.values():
                    raise ValueError(
                        f"{label}: two sources are both named {name!r}; names "
                        f"label events in an event file and must be unique.")

                names[source] = name

        return sources, names

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
                               'Nu': Angle(270*u.deg - sim_event.direction).wrap_at(180*u.deg),
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


# ---------------------------------------------------------------------------
# The top level of the schema (`docs/dev/inertial_sim_plan.md`, Section 7).
# Read by `SimulatorBase._from_config` above.
# ---------------------------------------------------------------------------


#: Top-level keys both simulators accept.
_COMMON_TOP_KEYS = ('detector', 'earth', 'sources', 'reconstructor',
                    'doppler_broadening', 'random_seed')

#: Largest seed `numpy.random.seed` accepts.
_MAX_SEED = 2 ** 32 - 1
