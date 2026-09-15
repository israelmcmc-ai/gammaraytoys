from abc import ABC, abstractmethod
from gammaraytoys.physics import ComptonPhysics2D
import numpy as np

from ..config_utils import _as_mapping, _check_keys, _dispatch_type_name

class Reconstructor(ABC):

    @abstractmethod
    def reconstruct(self, sim_event):
        pass

    @classmethod
    def from_config(cls, config, where = 'reconstructor'):
        """
        Build a `Reconstructor` from its configuration block.

        ```yaml
        {type: SimpleTraditionalReconstructor}
        ```

        Called on `Reconstructor`, the block's `type` chooses the class -- there
        is one to choose from today. Called on the concrete class, that class is
        what gets built: `type` may name it or be left out, and naming a
        different one raises rather than quietly handing back the other class.

        Parameters
        ----------
        config : mapping
            The `reconstructor` block.
        where : str
            Label for this block in error messages.

        Returns
        -------
        `Reconstructor`

        Raises
        ------
        ValueError
            On an unknown type or key.
        """

        block = _as_mapping(config, where)
        _dispatch_type_name(cls, Reconstructor, block, where,
                            _RECONSTRUCTOR_TYPES, _RECONSTRUCTOR_CLASSES,
                            'reconstructor')
        _check_keys(block, where, ('type',))

        return SimpleTraditionalReconstructor()
    def to_config(self):
        """
        Write this reconstructor back out as a configuration block.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            `{'type': 'SimpleTraditionalReconstructor'}`.

        Raises
        ------
        ValueError
            If this is not a type a configuration can name. Anything else that
            subclasses `Reconstructor` lands here, which is the only honest
            answer -- a configuration has no `type` name for it.
        """

        if isinstance(self, SimpleTraditionalReconstructor):
            return {'type': 'SimpleTraditionalReconstructor'}

        raise ValueError(
            f"{type(self).__name__} is not a reconstructor a "
            f"configuration can describe; the types that are: "
            f"{sorted(set(_RECONSTRUCTOR_TYPES.values()))}.")

class SimpleTraditionalReconstructor(Reconstructor):
    """
    Top layer is index 0. Assume only another bottom layer composed by everything else.

    Triggering requires at least one hit in the top layer (index 0) and at
    least one hit below it (layer > 0), since `psi` is reconstructed from
    the lever arm between a top hit and the mean position of the
    below-top hits -- an event with every hit confined to layer 0 has no
    such lever arm.
    """

    def reconstruct(self, sim_event):

        hits = sim_event.hits

        triggered = (hits.nhits >= 2
                     and hits.layer[0] == 0
                     and np.any(hits.layer > 0))

        if not triggered:
            # Didn't meet our trigger condition
            return RecoCompton()

        measured_energy = np.sum(hits.energy)

        # Energy and position
        energy_top = hits.energy[0]
        position_top = hits.position[0]
        position_bottom = np.mean(hits.position[hits.layer > 0])

        energy_out = measured_energy - energy_top

        #CDS
        phi = ComptonPhysics2D(measured_energy).scattering_angle(energy_out)

        if np.isnan(phi):
            # Unphysical. Likely a measurement error. Filter out
            return RecoCompton()
        
        psi = -np.arctan2(position_bottom.x - position_top.x,
                          position_top.y - position_bottom.y)

        return RecoCompton(energy = measured_energy,
                           phi = phi,
                           psi = psi)

class RecoEvent(ABC):
    """
    """

    @property
    @abstractmethod
    def triggered(self) -> bool:
        pass

class RecoCompton(RecoEvent):

    def __init__(self, energy = None, phi = None, psi = None):
        
        if energy is None and phi is None and psi is None:
            self._trig = False
        else:
            self._trig = True

        self.energy = energy
        self.phi = phi
        self.psi = psi

    @property
    def triggered(self):
        return self._trig
    

    
        
        


# ---------------------------------------------------------------------------
# What a configuration may call each of these classes.
#
# Both tables sit at the bottom of the file because the second one names the
# class above: `Reconstructor.from_config` looks it up when it runs, long
# after this module has finished importing.
# ---------------------------------------------------------------------------


#: Accepted spellings of every reconstructor type.
_RECONSTRUCTOR_TYPES = {'SimpleTraditionalReconstructor': 'SimpleTraditionalReconstructor'}

#: The class each canonical reconstructor type builds.
_RECONSTRUCTOR_CLASSES = {'SimpleTraditionalReconstructor':
                          SimpleTraditionalReconstructor}
