import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from gammaraytoys import Material
from astropy import units as u
from gammaraytoys.sims import Photon, Compton, Absorption
from gammaraytoys.physics import ComptonPhysics2D
from gammaraytoys.coordinates import Cartesian2D
from scipy.stats import norm, expon

class ToyTracker2D:

    def __init__(self, material, layer_length, layer_positions, layer_thickness, energy_resolution, energy_threshold):
        """
        
        """
        
        self._size = layer_length

        self._layer_pos = layer_positions

        # Cache these: they're pure functions of the (immutable) constructor
        # inputs above, but get accessed on every layer crossing during
        # event simulation, so recomputing them on every property access
        # (Quantity arithmetic / np.max/np.min over the layer array) adds up.
        self._left_bound = -self._size/2
        self._right_bound = self._size/2
        self._top_bound = np.max(self._layer_pos)
        self._bottom_bound = np.min(self._layer_pos)

        # Plain-float mirrors of the geometry above, in a fixed internal
        # length unit. simulate_event's hot loop walks the detector doing
        # scalar arithmetic on these on every layer crossing; using plain
        # floats there instead of Quantity avoids paying astropy's
        # unit-conversion machinery on every single operation.
        self._length_unit = self._layer_pos.unit
        self._layer_pos_value = self._layer_pos.to_value(self._length_unit)
        self._left_bound_value = self._left_bound.to_value(self._length_unit)
        self._right_bound_value = self._right_bound.to_value(self._length_unit)
        self._top_bound_value = self._top_bound.to_value(self._length_unit)
        self._bottom_bound_value = self._bottom_bound.to_value(self._length_unit)

        self._material = Material.from_name(material)

        self._layer_thickness = np.broadcast_to(layer_thickness, self.nlayers, subok=True)

        self._mthick = self._layer_thickness * self.material.density
        self._mthick_unit = self._mthick.unit
        self._mthick_value = self._mthick.value
        self._energy_res = np.broadcast_to(energy_resolution, self.nlayers)
        self._energy_thresh = np.broadcast_to(energy_threshold, self.nlayers, subok = True)

        self._npix = (layer_length/self._layer_thickness).to_value('').astype(int)
        self._pix_size = layer_length/self._npix
        self._pix_size_value = self._pix_size.to_value(self._length_unit)

        det_edges = Cartesian2D(u.Quantity([self.left_bound, self.right_bound, self.left_bound,    self.right_bound]),
                                u.Quantity([self.top_bound,  self.top_bound,   self.bottom_bound,  self.bottom_bound]))
        
        self._det_center = np.mean(det_edges)
        
        self._surr_radius = np.sqrt(np.max(np.sum(np.power(self._det_center.xyz[:,None] - det_edges.xyz, 2), axis = 0)))
        
        # Checks

        # Overlaps
        argsort_pos = np.argsort(layer_positions)
        sort_layer_pos = layer_positions[argsort_pos]
        sort_layer_thickness = self._layer_thickness[argsort_pos]

        gaps  = ((sort_layer_pos[1:]  - sort_layer_thickness[1:]/2) -
                 (sort_layer_pos[:-1] + sort_layer_thickness[:-1]/2))

        if np.any(gaps < 0):
            raise ValueError("Overlaps detected. Increase the space between layers or make them thinner.")
        
    @property
    def nlayers(self):
        return self._layer_pos.size

    @property
    def size(self):
        return self._size
    
    @property
    def material(self):
        return self._material

    @property
    def layer_positions(self):
        return self._layer_pos

    @property
    def mass_thickness(self):
        return self._mthick

    @property
    def position_resolution(self):
        return self._pix_size

    @property
    def energy_resolution(self):
        return self._energy_res

    @property
    def energy_threshold(self):
        return self._energy_thresh

    @property
    def left_bound(self):
        return self._left_bound

    @property
    def right_bound(self):
        return self._right_bound

    @property
    def top_bound(self):
        return self._top_bound

    @property
    def bottom_bound(self):
        return self._bottom_bound

    @property
    def surrounding_circle_radius(self):
        return self._surr_radius

    @property
    def surrounding_circle_center(self):
        return self._det_center
    
    def throwing_plane(self, offaxis_angle):

        surr_center = self.surrounding_circle_center
        surr_radius = self.surrounding_circle_radius

        cart_angle = 90*u.deg - offaxis_angle
        
        norm_vector = Cartesian2D(surr_radius*np.cos(cart_angle),
                                  surr_radius*np.sin(cart_angle))

        plane_origin = Cartesian2D(surr_center.x + norm_vector.x,
                                   surr_center.y + norm_vector.y)
        
        throw_parallel = Cartesian2D(-norm_vector.y,
                                     norm_vector.x)

        return plane_origin, throw_parallel

    @property
    def throwing_plane_size(self):
        return 2*self.surrounding_circle_radius
    
    @property
    def height(self):
        return self.top_bound - self.bottom_bound
    
    def plot(self, ax = None, event = None, draw_surrounding_circle = False, **kwargs):

        if ax is None:
            fig,ax = plt.subplots()

        length_unit = u.cm

        voxels = []
        for pos,layer_thickness,npix, pix_size in zip(self._layer_pos, self._layer_thickness, self._npix, self._pix_size):
            pos = pos.to_value(length_unit)
            layer_thickness = layer_thickness.to_value(length_unit)
            pix_size = pix_size.to_value(length_unit)
            for i in range(npix): 
                voxels.append(mpl.patches.Rectangle((-self.size.to_value(length_unit)/2 + i*pix_size, pos - layer_thickness/2),
                                                      pix_size, layer_thickness,
                                                     edgecolor = '.5',
                                                    facecolor = '.9', lw = 1)
                              )

        ax.add_collection(mpl.collections.PatchCollection(voxels, match_original=True))

        surr_center = self.surrounding_circle_center
        surr_radius = self.surrounding_circle_radius
        
        if draw_surrounding_circle:
            theta_plot = np.linspace(0,2*np.pi)
            x = (surr_center.x + surr_radius*np.cos(theta_plot)).to_value(length_unit)
            y = (surr_center.y + surr_radius*np.sin(theta_plot)).to_value(length_unit)
            ax.plot(x,y,ls = ':', color = 'black', alpha = .3)

            if event is not None:
                dist_plot = np.linspace(-1,1)

                plane_origin, plane_parallel = self.throwing_plane(270*u.deg - event.direction)
                x = (plane_origin.x + plane_parallel.x*dist_plot).to_value(length_unit)
                y = (plane_origin.y + plane_parallel.y*dist_plot).to_value(length_unit)
                ax.plot(x,y,ls = '--', color = 'black', alpha = .3)
                

        if event is not None:
            hits = event.hits
            ax.text(.03,.95,f"$\gamma(E = {event.energy:.1f}, k = {event.chirality})$",
                    transform=ax.transAxes)
            ax.text(.03,.03,f"Nhits = {hits.nhits}\nMeasured energy = {np.sum(hits.energy):.2f}",
                    transform=ax.transAxes)
            event.plot(ax, length_unit, **kwargs)

        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")

        ax.set_xlim((surr_center.x - 1.5*surr_radius).to_value(length_unit),
                    (surr_center.x + 1.5*surr_radius).to_value(length_unit))
        ax.set_ylim((surr_center.y - 1.5*surr_radius).to_value(length_unit),
                    (surr_center.y + 1.5*surr_radius).to_value(length_unit))
 
        ax.set_aspect('equal')
        
        return ax

    def simulate_event(self, particle, doppler_broadening = True):

        # particle.direction is fixed for the whole walk (only position changes
        # between layer crossings), so these only need computing once instead
        # of on every iteration of the loop below.
        flying_up = particle.direction < 180*u.deg
        flying_down = not flying_up
        flying_right = particle.direction < 90*u.deg or particle.direction > 270*u.deg
        flying_left = not flying_right

        if particle.direction == 0*u.deg or particle.direction == 180*u.deg:
            # Horizontal particles never cross a layer boundary
            return particle

        tan_direction = np.tan(particle.direction).value
        abs_sin_direction = np.abs(np.sin(particle.direction)).value

        # particle.energy is also fixed for the whole walk (it only changes
        # when a new particle is created after an interaction, which ends
        # this walk), so the attenuation coefficient it determines doesn't
        # need recomputing on every layer crossing either. Converted once to
        # a plain float compatible with the cached mass_thickness values (see
        # below) so the hot loop can do this arithmetic without Quantity
        # overhead.
        total_attenuation_coeff = self.material.total_attenuation(particle.energy)
        total_attenuation_coeff_value = total_attenuation_coeff.to_value(1/self._mthick_unit)

        # Track position as plain floats (in the detector's internal length
        # unit) while walking between layers, instead of a Cartesian2D/
        # Quantity: rewrapping every intermediate step is unnecessary
        # overhead when we only need an actual Cartesian2D once we record an
        # interaction (or never, if the particle exits without interacting).
        pos_x = particle.position.x.to_value(self._length_unit)
        pos_y = particle.position.y.to_value(self._length_unit)

        while True:

            # Terminate events flying out of boundaries
            if ((pos_x >= self._right_bound_value and flying_right)
                or
                (pos_x <= self._left_bound_value  and flying_left)
                or
                (pos_y >= self._top_bound_value and flying_up)
                or
                (pos_y <= self._bottom_bound_value and flying_down)):
                break

            # Determine interaction location
            new_pos_x = pos_x + (self._layer_pos_value - pos_y)/tan_direction

            # Check only the crosses within the detector, along the flying direction,
            # and excluding the current layer (if the particle starts exactly at a layer)
            y_dist_to_layers = self._layer_pos_value - pos_y

            crossed_tracker_idx = np.where((new_pos_x < self._right_bound_value) &
                                           (new_pos_x > self._left_bound_value) &
                                           (y_dist_to_layers > 0 if flying_up else y_dist_to_layers < 0)
                                           )[0]

            y_dist_to_crosses = y_dist_to_layers[crossed_tracker_idx] * (-1 if flying_down else 1)

            if y_dist_to_crosses.size == 0:
                # No interactions, flew in between layers
                break

            layer_idx_crossed = np.argmin(y_dist_to_crosses)

            layer_idx = crossed_tracker_idx[layer_idx_crossed].item()

            new_pos_x = new_pos_x[layer_idx]
            new_pos_y = self._layer_pos_value[layer_idx]

            # Determine if it interacted based on the total attenuation coefficient
            # (Beer-Lambert law: survival probability = exp(-optical depth))
            optical_depth = self._mthick_value[layer_idx] * total_attenuation_coeff_value / abs_sin_direction
            interaction_prob = 1 - np.exp(-optical_depth)

            if np.random.uniform() > interaction_prob:
                # Didn't interact. Continues flying
                pos_x, pos_y = new_pos_x, new_pos_y
                continue

            new_pos = Cartesian2D(new_pos_x*self._length_unit, new_pos_y*self._length_unit)

            # Add measurement error to position
            pix_size = self._pix_size_value[layer_idx]
            measured_x = (np.floor(new_pos_x/pix_size) + 1/2)*pix_size
            measured_y = new_pos_y
            measured_pos = Cartesian2D(measured_x*self._length_unit,
                                       measured_y*self._length_unit)

            # Determined which interaction type we have. Only Compton or total absorption for now.

            int_type = 'absorption'
            
            # If pair and compton, we assume that the e- and e+ are fully absorbed.
            compton_attenuation_coeff = self.material.compton_attenuation(particle.energy)
            
            if np.random.uniform() < compton_attenuation_coeff / total_attenuation_coeff:
                # Compton.
                compton_physics = ComptonPhysics2D(particle.energy)
                
                # Get random direction
                scattering_angle = compton_physics.random_scattering_angle(chirality = particle.chirality)
                new_direction = particle.direction + scattering_angle

                # Fudge doppler broadening by getting the corresponding outgoing
                # energy for the obtained angle given a non-zero elecontr momentum.
                # This is not really realistic, since the non-free electron
                # also changes the theta distribution, and the electron are also bound
                # 10 keV is not the bounding energy of an electron! But it gives a
                # reasonable broadening that illustrates the effect
                p_electron = None
                if doppler_broadening:
                    p_electron = np.random.uniform(-1,1)*expon.rvs(scale = 10)*u.keV
                
                # Derive the deposited energy from kinematics
                energy_out = compton_physics.energy_out(scattering_angle, p_electron)

                deposited_energy = particle.energy - energy_out

                if deposited_energy > 0:
                    # Rarely, for low energy photons, due to our fudge doppler
                    # broadening, the photon can actually gain energy.
                    # in that case consider it fully absorbed
                    
                    # Add measurement errors energy
                    energy_res = self.energy_resolution[layer_idx] * deposited_energy.value
                    measured_energy = norm.rvs(deposited_energy.value,
                                               scale = energy_res)
                    measured_energy = np.maximum(0, measured_energy)
                    measured_energy *= deposited_energy.unit

                    # Add interaction to tree
                    int_type = 'compton'
                    compton = Compton(position = new_pos,
                                      energy = deposited_energy)

                    compton.add_parent(particle)

                    if deposited_energy > self.energy_threshold[layer_idx]:
                        compton.set_measurement(layer = layer_idx,
                                                position = measured_pos,
                                                energy = measured_energy)

                    # Add child particles (no electron, assumed fully absorbed for now)
                    photon = Photon(position = new_pos,
                                    direction = new_direction,
                                    energy = energy_out,
                                    chirality = particle.chirality)

                    photon.add_parent(compton)

                    # Continue simulation, iterative (mutates photon.interaction in place).
                    # The doppler flag has to be passed down explicitly, otherwise
                    # every scattered photon falls back to the default and gets
                    # broadened even when the caller asked for no broadening.
                    self.simulate_event(photon, doppler_broadening = doppler_broadening)


            
            # If the event reached this point consider it fully absorbed
            if int_type == 'absorption':
            
                # Add interaction to tree
                absorption = Absorption(position = new_pos,
                                        energy = particle.energy)

                absorption.add_parent(particle)

                # Add measurement errors energy
                measured_energy = norm.rvs(particle.energy.value,
                                           scale = self.energy_resolution[layer_idx] * particle.energy.value)
                measured_energy *= particle.energy.unit

                absorption.set_measurement(layer = layer_idx,
                                           position = measured_pos,
                                           energy = measured_energy)

            # Terminate
            break

        return particle






