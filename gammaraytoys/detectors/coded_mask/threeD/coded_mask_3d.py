import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle, UnitSphericalRepresentation
from astropy.units import Quantity
from scipy.stats import poisson
from histpy import Histogram, Axis, Axes
from scipy.stats import poisson, norm
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad
from tqdm import tqdm
import sparse
import h5py as h5

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

    def write(self, filename):

        Histogram(self.detector_axes).write(filename, 'detector_axes')
        self.mask.write(filename, 'mask')

        if self.response is not None:
            self.response.write(filename, 'response')

        with h5.File(filename, 'a') as f:

            f.attrs['mask_separation'] = str(self.mask_separation)
            f.attrs['detector_efficiency'] = str(self.detector_efficiency)
            f.attrs['shielding'] = str(self.shielding)

    @classmethod
    def open(cls, filename):

        detector_axes = Histogram.open(filename, 'detector_axes').axes
        mask = Histogram.open(filename, 'mask')
        
        with h5.File(filename, 'r') as f:

            mask_separation = u.Quantity(f.attrs['mask_separation'])
            detector_efficiency = u.Quantity(f.attrs['detector_efficiency'])
            shielding = u.Quantity(f.attrs['shielding'])

            has_response = 'response' in f

        new = cls(detector_axes = detector_axes,
                  mask = mask,
                  mask_separation = mask_separation,
                  detector_efficiency = detector_efficiency,
                  shielding = shielding)

        if has_response:
            new._response = Histogram.open(filename, 'response')

        return new
            
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
    def open_fraction(self):
        return np.sum(self.mask) / np.prod(self.mask.nbins)

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
    
    def _get_mask_axis_geom_weights(self, det_axis, mask_axis, angle):

        mask_proj_edges = mask_axis.edges - self._mask_sep * np.tan(angle)

        mask_proj_idx = det_axis.find_bin(mask_proj_edges)

        # Each elemet will be a list of mask_bins,geom_weight pairs
        mask_bins = []
        mask_weights = []

        for mask_bin,(det_bin_i,det_bin_f) in enumerate(zip(mask_proj_idx[:-1], mask_proj_idx[1:])):

            if det_bin_f < 0 or det_bin_i >= det_axis.nbins: 
                continue

            # Middle bins. Full
            for det_bin,width in zip(range(det_bin_i+1, det_bin_f),
                                     det_axis.widths[det_bin_i+1: det_bin_f]):

                mask_bins += [[det_bin, mask_bin]]
                mask_weights += [width.value]

            # Lower edge
            if det_bin_i >= 0:
                upper_bound = det_axis.upper_bounds[det_bin_i]

                mask_bins += [[det_bin_i, mask_bin]]
                mask_weights += [(upper_bound - mask_proj_edges[mask_bin]).value]

            # Upper edge
            if det_bin_f < det_axis.nbins:
                lower_bound = det_axis.lower_bounds[det_bin_f]

                mask_bins += [[det_bin_f, mask_bin]]
                mask_weights += [(mask_proj_edges[mask_bin+1] - lower_bound).value]

        mask_bins_weights = sparse.COO(coords = np.transpose(mask_bins),
                                       data = mask_weights,
                                       shape = [det_axis.nbins, mask_axis.nbins])

        #return mask_bins_weights.tocsr()
        return sparse.GCXS.from_coo(mask_bins_weights)
    
    @u.quantity_input(flux = u.Unit()/u.m/u.m/u.s, duration = u.s, angle = u.rad)
    def point_source_response(self, flux, duration, coord, fluctuate = False, imaging = True, use_cache = True):

        # Standarize coordinate
        coord = coord.represent_as(UnitSphericalRepresentation)

        lon = coord.lon
        lat = coord.lat

        if lon > 180*u.deg:
            lon -= 360*u.deg

        # Check if catched
        response = None
        if use_cache and self.response is not None:
            if (lon >= self.response.axes['lon'].lo_lim and lon < self.response.axes['lon'].hi_lim and
                lat >= self.response.axes['lat'].lo_lim and lat < self.response.axes['lat'].hi_lim):

                response = self.response[self.response.axes['lon'].find_bin(lon),
                                         self.response.axes['lat'].find_bin(lat)]

        # Compute response if not cached
        if response is None:

            # Faster without units, we'll get them later
            lon = lon.to_value(u.rad)
            lat = lat.to_value(u.rad)
            
            # Init response to 0
            response = np.zeros(self._det_axes.nbins)

            # Coded mask geometry
            # Weight by pixel area
            det_x_widths = self.detector_axes[0].widths.value
            det_y_widths = self.detector_axes[1].widths.value
            det_x_lbounds = self.detector_axes[0].lower_bounds.value
            det_y_lbounds = self.detector_axes[1].lower_bounds.value
            det_x_ubounds = self.detector_axes[0].upper_bounds.value
            det_y_ubounds = self.detector_axes[1].upper_bounds.value

            pix_area = det_x_widths[:,None] * det_y_widths[None,:] 

            # Coded mask projection edges
            right_edge =  np.minimum(np.maximum(self.mask.axes['x'].hi_lim - np.tan(lon)*self.mask_separation, self.detector_axes[0].lo_lim), self.detector_axes[0].hi_lim)
            left_edge =   np.minimum(np.maximum(self.mask.axes['x'].lo_lim - np.tan(lon)*self.mask_separation, self.detector_axes[0].lo_lim), self.detector_axes[0].hi_lim)
            top_edge =    np.minimum(np.maximum(self.mask.axes['y'].hi_lim - np.tan(lat)*self.mask_separation, self.detector_axes[1].lo_lim), self.detector_axes[1].hi_lim)
            bottom_edge = np.minimum(np.maximum(self.mask.axes['y'].lo_lim - np.tan(lat)*self.mask_separation, self.detector_axes[1].lo_lim), self.detector_axes[1].hi_lim)

            # Shield

            # Get mask projection bins
            right_edge_bin =  np.minimum(np.maximum(self.detector_axes[0].find_bin(right_edge),  0), self.detector_axes[0].nbins-1)
            left_edge_bin =   np.minimum(np.maximum(self.detector_axes[0].find_bin(left_edge),   0), self.detector_axes[0].nbins-1)
            top_edge_bin =    np.minimum(np.maximum(self.detector_axes[1].find_bin(top_edge),    0), self.detector_axes[1].nbins-1)
            bottom_edge_bin = np.minimum(np.maximum(self.detector_axes[1].find_bin(bottom_edge), 0), self.detector_axes[1].nbins-1)

            # Unit applied later
            right_edge  = right_edge.value
            left_edge   = left_edge.value
            top_edge    = top_edge.value
            bottom_edge = bottom_edge.value

            shield_leak = 1-self.shielding
            response[:] = shield_leak

            if imaging:
                mask_factor = 0
            else:
                mask_factor = self.open_fraction

            # Pixels fully contains
            response[left_edge_bin+1:right_edge_bin, bottom_edge_bin+1:top_edge_bin] = mask_factor

            # Mask edge
            diff_factor = mask_factor-shield_leak
            left_factor   = (det_x_ubounds[left_edge_bin] - left_edge)     / det_x_widths[left_edge_bin]
            right_factor  = (right_edge - det_x_lbounds[right_edge_bin])   / det_x_widths[right_edge_bin]
            bottom_factor = (det_y_ubounds[bottom_edge_bin] - bottom_edge) / det_y_widths[bottom_edge_bin]
            top_factor    = (top_edge - det_y_lbounds[top_edge_bin])       / det_y_widths[top_edge_bin]

            # Sides
            response[left_edge_bin,  bottom_edge_bin+1:top_edge_bin] += diff_factor * left_factor
            response[right_edge_bin, bottom_edge_bin+1:top_edge_bin] += diff_factor * right_factor

            response[left_edge_bin+1:right_edge_bin, bottom_edge_bin] += diff_factor * bottom_factor
            response[left_edge_bin+1:right_edge_bin, top_edge_bin]    += diff_factor * top_factor

            # Corners
            response[left_edge_bin,  bottom_edge_bin] += diff_factor * left_factor  * bottom_factor
            response[right_edge_bin, bottom_edge_bin] += diff_factor * right_factor * bottom_factor
            response[left_edge_bin,  top_edge_bin]    += diff_factor * left_factor  * top_factor
            response[right_edge_bin, top_edge_bin]    += diff_factor * right_factor * top_factor

            # Area overal weights
            response *= pix_area

            # Coded Mask
            if imaging:

                # Get geometrical weight along each axis
                mask_bin_weights_x = self._get_mask_axis_geom_weights(self._det_axes['x'], self._mask.axes['x'], lon)
                mask_bin_weights_y = self._get_mask_axis_geom_weights(self._det_axes['y'], self._mask.axes['y'], lat)

                # Equiv to
                # response[det_bin_x, det_bin_y] += mask[mask_bin_x, mask_bin_y] * geom_weight_x * geom_weight_y
                # Over all det and mask bins
                mask = sparse.GCXS.from_numpy(self.mask)

                response += sparse.einsum('jn,in', mask_bin_weights_y, sparse.einsum('im,mn', mask_bin_weights_x, mask)).todense()

            # Weight by exposure and
            off_axis_angle = np.arccos(coord.to_cartesian().x)

            response = response * self._det_axes['x'].unit * self._det_axes['y'].unit * np.cos(off_axis_angle) * self._det_eff

        # Multiply by exposuse
        expectation = response * flux * duration

        # Convert Quantity to np array (it should be already unitless)
        expectation = Histogram(self.detector_axes, expectation.to('').value)
        
        if fluctuate:
            expectation[:] = poisson.rvs(mu = expectation.contents)
        
        return expectation

    def compute_response(self, lon_range = None, lat_range = None):

        # Standarize input
        if lon_range is None:
            min_lon = 0
            max_lon = self.sky_axes['lon'].nbins
        else:
            min_lon,max_lon = self.sky_axes['lon'].find_bin(u.Quantity(lon_range))

        if lat_range is None:
            min_lat = 0
            max_lat = self.sky_axes['lat'].nbins
        else:
            min_lat,max_lat = self.sky_axes['lat'].find_bin(u.Quantity(lat_range))

        response = Histogram([self.sky_axes['lon'][min_lon:max_lon+1],
                              self.sky_axes['lat'][min_lat:max_lat+1]] +
                              list(self.detector_axes),
                              track_overflow = False)
                             
        flux = 1/u.cm/u.cm/u.s
        duration = 1*u.s
        
        for nLon, lon in tqdm(enumerate(response.axes['lon'].centers), total = response.axes['lon'].nbins):
            for nLat, lat in enumerate(response.axes['lat'].centers):

                coord = UnitSphericalRepresentation(lon = lon, lat = lat)

                response[nLon, nLat] = self.point_source_response(flux = flux,
                                                                  coord = coord,
                                                                  duration = duration,
                                                                  fluctuate = False).contents
                                
        # Normalize and given area units
        response /= flux*duration

        self._response = response
            
    @property
    def response(self):
        return self._response

    def effective_area(self, coord):
        
        # Standarize coordinate
        coord = coord.represent_as(UnitSphericalRepresentation)
        
        if (np.abs(coord.lon) > self.fully_coded_fov[0]) or (np.abs(coord.lon) > self.fully_coded_fov[1]) :
            return 0*u.cm

        # TODO: obtain from 
        if self._response is not None:
            # From cache
            psr = self.response[*self.response.axes['lon','lat'].find_bin(coord.lon, coord.lat)]
        else:
            # On the fly
            psr = self.point_source_response(flux = 1/u.cm/u.cm/u.s,
                                             coord = coord,
                                             duration = 1*u.s,
                                             fluctuate = False)
        
        return np.sum(psr) * u.cm * u.cm
    

    
    @u.quantity_input(flux = u.Unit()/u.cm/u.cm/u.s, angle = u.rad, width = u.rad)
    def gaussian_model(self, flux, coord, width_lon, width_lat, max_sigma = 5):
        """
        0 beyond max_sigma
        """

        width_lon = np.maximum(self.angular_resolution/1e6, width_lon)
        width_lat = np.maximum(self.angular_resolution/1e6, width_lat)

        model = Histogram(self.sky_axes, unit = flux.unit) 

        dist = multivariate_normal(mean = [coord.lon.to_value(u.rad), coord.lat.to_value(u.rad)],
                                   cov = [[width_lon.to_value(u.rad)**2,0],[0,width_lat.to_value(u.rad)**2]])

        # Mask at 3 sigma for speed
        lon_axis = self.sky_axes['lon']
        min_lon_bin = np.maximum(0,                  lon_axis.find_bin(coord.lon - max_sigma*width_lon))
        max_lon_bin = np.minimum(lon_axis.nbins - 1, lon_axis.find_bin(coord.lon + max_sigma*width_lon))
        lon_edges = lon_axis.edges[min_lon_bin:max_lon_bin+2].to_value(u.rad)
                                 
        lat_axis = self.sky_axes['lat']
        min_lat_bin = np.maximum(0,                  lat_axis.find_bin(coord.lat - max_sigma*width_lat))
        max_lat_bin = np.minimum(lat_axis.nbins - 1, lat_axis.find_bin(coord.lat + max_sigma*width_lat))
        lat_edges = lat_axis.edges[min_lat_bin:max_lat_bin+2].to_value(u.rad)

        if min_lon_bin >= lon_axis.nbins or max_lon_bin < 0 or min_lat_bin >= lat_axis.nbins or max_lat_bin < 0:
            # Fully outside
            return model
        
        # Compute
        LON,LAT = np.meshgrid(lon_edges, lat_edges, indexing='ij')
        
        LON *= np.cos(LAT) # cos for approx phase spase

        cdf = dist.cdf(np.transpose([LON,LAT], [1,2,0]))

        pdf_int = np.diff(np.diff(cdf, axis = 0), axis = 1)

        model[min_lon_bin:max_lon_bin+1, min_lat_bin:max_lat_bin+1] = flux * pdf_int

        return model

    @u.quantity_input(rate = u.Hz, duration = u.s)
    def uniform_bkg(self, rate, duration):

        bkg = Histogram(self.detector_axes)

        bkg[:] = bkg.axes[0].widths.value[:,None] * bkg.axes[1].widths.value[None,:] 
        
        bkg *= (rate*duration).to('').value / np.sum(bkg)

        return bkg

    # ======== Tested in 3D above this line ======

    @u.quantity_input(flux = u.Unit()/u.cm/u.cm/u.s/u.sr, angle = u.rad, width = u.rad)
    def isotropic_diffuse_model(self, flux):

        model = Histogram(self.sky_axes, unit = flux.unit * u.sr) 

        model[:] = flux * model.axes['lon'].widths[:,None] * (np.sin(model.axes['lat'].upper_bounds) - np.sin(model.axes['lat'].lower_bounds)) * u.rad

        return model
    
    # ======== 2D bwlow this line =======
    
    @u.quantity_input(model = u.Unit()/u.m/u.m/u.s, duration = u.s)
    def convolve_model(self, model, duration, fluctuate = False, imaging = True):

        expectation = Histogram(self.detector_axes)

        nLon_list, nLat_list = model.contents.nonzero()

        for nLon, nLat, lon, lat in tqdm(zip(nLon_list, nLat_list, model.axes['lon'].centers[nLon_list], model.axes['lat'].centers[nLat_list]),
                                         total = len(nLon_list)):
        
            flux = model[nLon, nLat]

            coord = UnitSphericalRepresentation(lon = lon, lat = lat)

            psr =  self.point_source_response(flux = flux,
                                              coord = coord,
                                              duration = duration,
                                              fluctuate = fluctuate,
                                              imaging = imaging)

            expectation += psr

        return expectation

