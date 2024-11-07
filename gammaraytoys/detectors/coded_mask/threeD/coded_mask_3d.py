import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, UnitSphericalRepresentation
from astropy.units import Quantity
from scipy.stats import poisson
from histpy import Histogram, Axis, Axes
from scipy.stats import poisson, norm
from tqdm import tqdm

class ToyCodedMaskDetector3D:

    def __init__(self, detector_axes, mask, mask_separation, detector_efficiency, shielding = 1):
        """
        Shielding = shield efficiency (to the signal)
        """
        
        self._mask_sep = mask_separation
        self._det_axes = detector_axes
        self._mask = mask
        self._det_eff = detector_efficiency
        self._sky_axes = None
        self._response = None
        self.shielding = shielding

    @classmethod
    @u.quantity_input(mask_size = u.m, mask_separation = u.m, detector_size = u.m)
    def create_random_mask(cls, mask_size, mask_npix, mask_separation, open_fraction, detector_size, detector_npix, detector_efficiency, shielding = 1):

        # Go to 2D if needed
        if mask_size.size == 1:
            mask_size = u.Quantity([mask_size,mask_size])

        if detector_size.size == 1:
            detector_size = u.Quantity([detector_size, detector_size])

        if np.isscalar(mask_npix):
            mask_npix = (mask_npix, mask_npix)

        if np.isscalar(detector_npix):
            detector_npix = (detector_npix, detector_npix)

        # Init
        return cls(detector_axes = Axes([Axis(np.linspace(-detector_size[0]/2, detector_size[0]/2, detector_npix[0]+1),
                                              label = 'x'),
                                         Axis(np.linspace(-detector_size[1]/2, detector_size[1]/2, detector_npix[1]+1),
                                              label = 'y')]), 
                   mask = Histogram([np.linspace(-mask_size[0]/2, mask_size[0]/2, mask_npix[0]+1), np.linspace(-mask_size[1]/2, mask_size[1]/2, mask_npix[1]+1)],
                                    (np.random.uniform(size = mask_npix) < open_fraction).astype(int),
                                    labels = ['x','y']), 
                   mask_separation = mask_separation, 
                   detector_efficiency = detector_efficiency,
                   shielding = shielding)

    @property
    def mask(self):
        return self._mask

    @property
    def detector_axes(self):
        return self._det_axes

    @property
    def detector_efficiency(self):
        return self._det_eff

    @property
    def mask_separation(self):
        return self._mask_sep

    @property
    def angular_resolution(self):
        return np.arctan(np.min(u.Quantity([ax.widths for ax in self.mask.axes]))/self.mask_separation)

    @property
    def mask_size(self):
        return u.Quantity([self.mask.axes[0].hi_lim - self.mask.axes[0].lo_lim,
                           self.mask.axes[1].hi_lim - self.mask.axes[1].lo_lim])

    @property
    def detector_size(self):
        return u.Quantity([self.detector_axes[0].hi_lim - self.detector_axes[0].lo_lim,
                           self.detector_axes[1].hi_lim - self.detector_axes[1].lo_lim])
    
    
    @property
    def partially_coded_fov(self):
        return u.Quantity([np.arctan((self.mask_size[0]/2+self.detector_size[0]/2)/self.mask_separation),
                           np.arctan((self.mask_size[1]/2+self.detector_size[1]/2)/self.mask_separation)])

    @property
    def sky_axes(self):

        if self._sky_axes is None:
            # Compute and cache

            fov = self.partially_coded_fov

            self._sky_axes =  Axes([np.arange(-fov[0].to_value(u.degree),
                                              fov[0].to_value(u.degree),
                                              self.angular_resolution.to_value(u.degree)/5) * u.degree,
                                    np.arange(-fov[1].to_value(u.degree),
                                              fov[1].to_value(u.degree),
                                              self.angular_resolution.to_value(u.degree)/5) * u.degree],
                                   labels = ['lon','lat'])

        return self._sky_axes
            
    @property
    def fully_coded_fov(self):
        return u.Quantity([np.arctan((self.mask_size[0]/2-self.detector_size[0]/2)/self.mask_separation),
                           np.arctan((self.mask_size[0]/2-self.detector_size[0]/2)/self.mask_separation)])

    # ======== Tested in 3D above this line ======

    def _get_mask_axis_geom_weights(self, det_axis, mask_axis, angle):

        mask_proj_edges = mask_axis.edges - self._mask_sep * np.tan(angle)

        mask_proj_idx = det_axis.find_bin(mask_proj_edges)

        # Each elemet will be a list of mask_bins,geom_weight pairs
        mask_bins_weights = [[] for i in range(det_axis.nbins)]
        
        for mask_bin,(det_bin_i,det_bin_f) in enumerate(zip(mask_proj_idx[:-1], mask_proj_idx[1:])):

            if det_bin_f < 0 or det_bin_i >= det_axis.nbins: 
                continue

            # Middle bins. Full
            for det_bin,width in zip(range(det_bin_i+1, det_bin_f),
                                     det_axis.widths[det_bin_i+1: det_bin_f]):
                mask_bins_weights[det_bin] += [(mask_bin, width)]
            
            # Lower edge
            if det_bin_i >= 0:
                upper_bound = det_axis.upper_bounds[det_bin_i]
                mask_bins_weights[det_bin_i] += [(mask_bin, upper_bound - mask_proj_edges[mask_bin])]
            
            # Upper edge
            if det_bin_f < det_axis.nbins:
                lower_bound = det_axis.lower_bounds[det_bin_f]
                mask_bins_weights[det_bin_f] += [(mask_bin, mask_proj_edges[mask_bin+1] - lower_bound)]

        return mask_bins_weights
                
    @u.quantity_input(flux = u.Unit()/u.m/u.m/u.s, duration = u.s, angle = u.rad)
    def point_source_response(self, flux, duration, coord, fluctuate = False):

        # Standarize coordinate
        coord = coord.represent_as(UnitSphericalRepresentation)

        lon = coord.lon.to_value(u.rad)
        lat = coord.lat.to_value(u.rad)

        # Init expectation to 0
        # Will remove units when multiplying by flux and duration
        expectation = Histogram(self._det_axes,
                                unit = 1/flux.unit/duration.unit)

        # Coded mask projection edges
        right_edge =  self.mask.axes['x'].hi_lim - np.tan(lon)*self.mask_separation
        left_edge =   self.mask.axes['x'].lo_lim - np.tan(lon)*self.mask_separation
        top_edge =    self.mask.axes['y'].hi_lim - np.tan(lat)*self.mask_separation
        bottom_edge = self.mask.axes['y'].lo_lim - np.tan(lat)*self.mask_separation
        
        # Get geometrical weight along each axis
        mask_bin_weights_x = self._get_mask_axis_geom_weights(self._det_axes['x'], self._mask.axes['x'], lon)
        mask_bin_weights_y = self._get_mask_axis_geom_weights(self._det_axes['y'], self._mask.axes['y'], lat)

        # Loop through all x,y weights combinations
        for det_bin_x,(det_bin_left, det_bin_right) in tqdm(enumerate(zip(expectation.axes[0].lower_bounds, expectation.axes[0].upper_bounds)), total = expectation.axes[0].nbins):
            for det_bin_y,(det_bin_bottom, det_bin_top) in enumerate(zip(expectation.axes[1].lower_bounds, expectation.axes[1].upper_bounds)):

                # From mask
                for mask_bin_x, geom_weight_x in mask_bin_weights_x[det_bin_x]:
                    for mask_bin_y, geom_weight_y in mask_bin_weights_y[det_bin_y]:
                        expectation[det_bin_x, det_bin_y] += self.mask[mask_bin_x, mask_bin_y] * geom_weight_x * geom_weight_y

                # Outside mask, through shielding
                shield_leak = 1-self.shielding

                if left_edge <= det_bin_left or det_bin_right <= right_edge:
                    shield_leak *= 0
                else:
                    shield_leak *= np.min(left_edge, det_bin_right) - np.max(right_edge, det_min_left)
                    
                if bottom_edge <= det_bin_bottom or det_bin_top <= top_edge:
                    shield_leak *= 0
                else:
                    shield_leak *= np.min(bottom_edge, det_bin_top) - np.max(top_edge, det_min_bottom)
                
                expectation[det_bin_x, det_bin_y] += shield_leak
                        
        # Weight by exposure
        off_axis_angle = np.arccos(coord.to_cartesian().x)
        
        expectation *= flux * duration * np.cos(off_axis_angle) * self._det_eff

        expectation.clear_underflow_and_overflow()

        # Convert Quantity to np array (it should be already unitless)
        expectation = Histogram(expectation.axes, expectation.contents.to('').value)
        
        if fluctuate:
            expectation[:] = poisson.rvs(mu = expectation.contents)
        
        return expectation

    
    # ======== 2D bwlow this line =======

    
    @property
    def response(self):
        if self._response is None:
            # Compute and cache

            flux = 1/u.cm/u.s
            duration = 1*u.s
            
            response = Histogram.concatenate(self.sky_axis,
                                             [self.point_source_response(flux = flux,
                                                                         angle = a,
                                                                         duration = duration,
                                                                         fluctuate = False)
                                              for a in self.sky_axis.centers])
            
            response = response.project(1,0) # Transpose

            # Give correct area units
            response = Histogram(response.axes, response.contents/flux/duration)

            self._response = response
                            
        return self._response

    def effective_area(self, angle):

        if np.abs(angle > self.fully_coded_fov):
            return 0*u.cm
        
        return np.sum(self.response[:, self.sky_axis.find_bin(angle)])
    
    @u.quantity_input(model = u.Unit()/u.m/u.s, duration = u.s)
    def convolve_model(self, model, duration, fluctuate = True):

        expectation = duration*np.dot(self.response.contents, model.contents)
        
        expectation = Histogram(self._det_axis,
                                contents = expectation.to('').value)

        return expectation

    @u.quantity_input(flux = u.Unit()/u.m/u.s, angle = u.rad, width = u.rad)
    def gaussian_model(self, flux, angle, width):

        #Prevent numerical error from point sources
        width = np.maximum(self.angular_resolution/1e6, width) 
        
        # Factor 5 ang res, somewhat arbitrary
        model = Histogram(self.sky_axis, unit = flux.unit)

        norm_cdf = norm.cdf(model.axis.edges.to_value(u.rad),
                            loc = angle.to_value(u.rad),
                            scale = width.to_value(u.rad))
        model[:] = flux * (norm_cdf[1:] - norm_cdf[:-1])

        return model

    @u.quantity_input(rate = u.Hz, duration = u.s)
    def uniform_bkg(self, rate, duration):

        bkg = Histogram(self.detector_axis)

        bkg[:] = bkg.axis.widths.value
        
        bkg *= (rate*duration).to('').value / np.sum(bkg)

        return bkg
