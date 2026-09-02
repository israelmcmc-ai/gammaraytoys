from abc import ABC, abstractmethod
from gammaraytoys.physics import ComptonPhysics2D
import numpy as np

class Reconstructor(ABC):

    @abstractmethod
    def reconstruct(self, sim_event):
        pass

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
    

    
        
        
