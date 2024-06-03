import xarray as xr
import numpy as np
import torch.fft
import torch
import logging
import os
import warnings
from scipy.signal import convolve2d
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


logger = logging.getLogger(__name__)


class WavePropagator(object):

    def __init__(self,
                 data_path,
                 n_bins=1000,
                 tile_size=512,
                 step_size=128,
                 marker_size=10,
                 transform_mode=None,
                 device="cpu"):

        self.h_ds = xr.open_dataset(data_path)

        if 'zMin' in self.h_ds.attrs:
            self.zMin = self.h_ds.attrs['zMin']  # minimum z in sample volume
            self.zMax = self.h_ds.attrs['zMax']
        else:  # some of the raw data does not have this parameter
            # should warn the user here through the logger
            self.zMin = 0.014
            self.zMax = 0.158

        self.n_bins = n_bins
        self.z_bins = np.linspace(
            self.zMin, self.zMax, n_bins+1)*1e6  # histogram bin edges
        self.z_centers = self.z_bins[:-1] + 0.5 * \
            np.diff(self.z_bins)  # histogram bin centers
        self.dz = np.mean(np.diff(self.z_bins))  # calculate plane increment size

        self.tile_size = tile_size  # size of tiled images in pixels
        self.step_size = step_size  # amount that we shift the tile to make a new tile
        # UNET gaussian marker width (standard deviation) in um
        self.marker_size = marker_size
        self.device = device

        # step_size is not allowed be be larger than the tile_size
        assert self.tile_size >= self.step_size

        self.dx = self.h_ds.attrs['dx']      # horizontal resolution
        self.dy = self.h_ds.attrs['dy']      # vertical resolution
        self.Nx = int(self.h_ds.attrs['Nx'])  # number of horizontal pixels
        self.Ny = int(self.h_ds.attrs['Ny'])  # number of vertical pixels
        self.lam = self.h_ds.attrs['lambda']  # wavelength
        self.image_norm = 255.0
        self.transform_mode = transform_mode

        logger.info(
            f"Wave propagation object loaded pre-wave-propagation transformation: {transform_mode}")

        self.x_arr = np.arange(-self.Nx//2, self.Nx//2)*self.dx
        self.y_arr = np.arange(-self.Ny//2, self.Ny//2)*self.dy

        self.tile_x_bins = np.arange(-self.Nx//2,
                                     self.Nx//2, self.step_size)*self.dx*1e6
        self.tile_y_bins = np.arange(-self.Ny//2,
                                     self.Ny//2, self.step_size)*self.dy*1e6

        self.fx = torch.fft.fftfreq(
            self.Nx, self.dx, device=self.device).unsqueeze(0).unsqueeze(2)
        self.fy = torch.fft.fftfreq(
            self.Ny, self.dy, device=self.device).unsqueeze(0).unsqueeze(1)

    def torch_holo_set(self,
                       Ein: torch.tensor,
                       z_tnsr: torch.tensor):
        """
        Propagates an electric field a distance z
        Ein complex torch.tensor
        - input electric field

        fx:real torch.tensor
        - x frequency axis (3D, setup to broadcast)

        fy: real torch.tensor
        - y frequency axis (3D, setup to broadcast)

        z_tnsr: torch.tensor
        - tensor of distances to propagate the wave Ein
            expected to have dims (Nz,1,1) where Nz is the number of z
            dimensions

        lam: float
        - wavelength

        returns: complex torch.tensor with dims (Nz,fy,fx)

        Note the torch.fft library uses dtype=torch.complex64
        This may be an issue for GPU implementation

        """

        if self.transform_mode == "standard":
            Ein = Ein.float()
            Ein -= torch.mean(Ein)
            Ein /= torch.std(Ein)

        elif self.transform_mode == "min-max":
            Ein = Ein.float()
            Ein -= torch.min(Ein)
            Ein /= torch.max(Ein)

        Etfft = torch.fft.fft2(Ein)
        Eofft = Etfft*torch.exp(1j*2*np.pi*z_tnsr/self.lam *
                                (torch.sqrt(1-self.lam**2*(self.fx**2+self.fy**2))-1))
        # the -1 term at the end of the line above is not coventional.  I added it to remove the
        # accumulation of linear phase with z propagation.  This keeps the overall phase from
        # changing between planes and should make it easier for the ML architecture to process.

        # It might be helpful if we could omit this step.  It would save an inverse fft.
        Eout = torch.fft.ifft2(Eofft)

        return Eout


class UpsamplingPropagator(WavePropagator):

    def get_reconstructed_sub_images(self, h_idx, part_per_holo=None, empt_per_holo=None):
        """
        Reconstruct a hologram at specific planes to provide training data
        with a specified number of sub images containing and not containing
        particles
        """

        with torch.no_grad():

            #### roughly half of the empty cases should be near in focus ####
            empt_near_cnt = empt_per_holo//2
            ####

            # locate particle information corresponding to this hologram
            particle_idx = np.where(self.h_ds['hid'].values == h_idx+1)

            x_part = self.h_ds['x'].values[particle_idx]
            y_part = self.h_ds['y'].values[particle_idx]
            z_part = self.h_ds['z'].values[particle_idx]
            # not used but here it is
            d_part = self.h_ds['d'].values[particle_idx]

            # create a 3D histogram
            in_data = np.stack((x_part, y_part, z_part)).T
            h_part = np.histogramdd(
                in_data, bins=[self.tile_x_bins, self.tile_y_bins, self.z_bins])[0]
            # specify the z bin locations of the particles
            z_part_bin_idx = np.digitize(z_part, self.z_bins)-1

            # smoothing kernel accounts for overlapping subimages when the
            # subimage is larger than the stride
            ratio = self.tile_size//self.step_size
            if self.step_size < self.tile_size:
                overlap_kernel = np.ones((ratio, ratio))
                for z_idx in range(h_part.shape[-1]):
                    h_part[:, :, z_idx] = convolve2d(h_part[:, :, z_idx], overlap_kernel)[
                        ratio-1:h_part.shape[0]+ratio-1, ratio-1:h_part.shape[1]+ratio-1]

            # locate all the cases where particles are and are not present
            # to sample from those cases
            if self.step_size < self.tile_size:
                # note that the last bin is ommitted from each to avoid edge cases where
                # the image is not complete

                edge_idx = ratio-1

                # find the locations where particles are in focus
                loc_idx = np.where(h_part[:-edge_idx, :-edge_idx, :] > 0)
                # find locations where particles are not in focus
                empt_idx = np.where(h_part[:-edge_idx, :-edge_idx, :] == 0)
                #### find locations where particles are nearly in focus  ####
                zdiff = np.diff(h_part[:-edge_idx, :-edge_idx, :], axis=2)
                zero_pad = np.zeros(
                    h_part[:-edge_idx, :-edge_idx, :].shape[0:2]+(1,))
                near_empt_idx = np.where((h_part[:-edge_idx, :-edge_idx, :] == 0) & ((np.concatenate(
                    [zdiff, zero_pad], axis=2) == 1) | (np.concatenate([zero_pad, zdiff], axis=2) == -1)))
                ####
            else:
                # find the locations where particles are in focus
                loc_idx = np.where(h_part > 0)
                # find locations where particles are not in focus
                empt_idx = np.where(h_part == 0)
                #### find locations where particles are nearly in focus ####
                zdiff = np.diff(h_part, axis=2)
                zero_pad = np.zeros(h_part.shape[0:2]+(1,))
                near_empt_idx = np.where((h_part == 0) & ((np.concatenate(
                    [zdiff, zero_pad], axis=2) == 1) | (np.concatenate([zero_pad, zdiff], axis=2) == -1)))
                ####

            # select sub images with particles in them
            if part_per_holo > loc_idx[0].size:
                # pick the entire set
                loc_x_idx = loc_idx[0]
                loc_y_idx = loc_idx[1]
                loc_z_idx = loc_idx[2]
            else:
                # randomly select particles from the set
                sel_part_idx = np.random.choice(
                    np.arange(loc_idx[0].size, dtype=int), size=part_per_holo, replace=False)
                loc_x_idx = loc_idx[0][sel_part_idx]
                loc_y_idx = loc_idx[1][sel_part_idx]
                loc_z_idx = loc_idx[2][sel_part_idx]

            # randomly select empties from the empty set
            #### Add nearly in focus cases to the training data ####
            sel_empt_idx = np.random.choice(np.arange(
                near_empt_idx[0].size, dtype=int), size=empt_near_cnt, replace=False)  # select nearly in focus cases
            ####
            sel_empt_idx = np.concatenate([np.random.choice(np.arange(empt_idx[0].size, dtype=int), size=(
                empt_per_holo-empt_near_cnt), replace=False), sel_empt_idx])  # select random out of focus cases
            empt_x_idx = empt_idx[0][sel_empt_idx]
            empt_y_idx = empt_idx[1][sel_empt_idx]
            empt_z_idx = empt_idx[2][sel_empt_idx]

            # full set of plane indices to reconstruct (empty and with particles)
            z_full_idx = np.unique(np.concatenate((loc_z_idx, empt_z_idx)))

            # build the torch tensor for reconstruction
            z_plane = torch.tensor(
                self.z_centers[z_full_idx]*1e-6, device=self.device).unsqueeze(-1).unsqueeze(-1)

            # create the torch tensor for propagation
            E_input = torch.tensor(self.h_ds['image'].isel(
                hologram_number=h_idx).values).to(self.device).unsqueeze(0)

            # reconstruct the selected planes
            E_out = self.torch_holo_set(
                E_input, z_plane).detach().cpu().numpy()

            # grab the sub images corresponding to the selected data points
            particle_in_focus_lst = []  # training labels for if particle is in focus
            particle_unet_labels_lst = []  # training labels for if particle is in focus
            image_lst = []  # sliced reconstructed image
            image_index_lst = []  # indices used to identify the image slice
            image_corner_coords = []  # coordinates of the corner of the image slice

            step_size = self.step_size
            tile_size = self.tile_size

            for sub_idx, z_idx in enumerate(z_full_idx):
                part_set_idx = np.where(loc_z_idx == z_idx)[0]
                empt_set_idx = np.where(empt_z_idx == z_idx)[0]

                # initialize the UNET mask
                unet_mask = np.zeros(E_out.shape[1:])
                part_in_plane_idx = np.where(z_part_bin_idx == z_idx)[
                    0]  # locate all particles in this plane

                # build the UNET mask for this z plane
                for part_idx in part_in_plane_idx:
                    #             unet_mask += np.exp(-(y_arr[None,:]*1e6-y_part[part_idx])**2/(2*marker_size**2) - (x_arr[:,None]*1e6-x_part[part_idx])**2/(2*marker_size**2) )
                    unet_mask += ((self.y_arr[None, :]*1e6-y_part[part_idx])**2 + (
                        self.x_arr[:, None]*1e6-x_part[part_idx])**2 < (d_part[part_idx]/2)**2).astype(float)

                for part_idx in part_set_idx:
                    x_idx = loc_x_idx[part_idx]
                    y_idx = loc_y_idx[part_idx]
                    image_lst.append(E_out[sub_idx, x_idx*step_size:(
                        x_idx*step_size+tile_size), y_idx*step_size:(y_idx*step_size+tile_size)])
                    image_index_lst.append([x_idx, y_idx, z_idx])
                    image_corner_coords.append(
                        [self.x_arr[x_idx*step_size], self.y_arr[y_idx*step_size]])
                    particle_in_focus_lst.append(1)
                    particle_unet_labels_lst.append(
                        unet_mask[x_idx*step_size:(x_idx*step_size+tile_size), y_idx*step_size:(y_idx*step_size+tile_size)])

                for empt_idx in empt_set_idx:
                    x_idx = empt_x_idx[empt_idx]
                    y_idx = empt_y_idx[empt_idx]
                    image_lst.append(E_out[sub_idx, x_idx*step_size:(
                        x_idx*step_size+tile_size), y_idx*step_size:(y_idx*step_size+tile_size)])
                    image_index_lst.append([x_idx, y_idx, z_idx])
                    image_corner_coords.append(
                        [self.x_arr[x_idx*step_size], self.y_arr[y_idx*step_size]])
                    particle_in_focus_lst.append(0)
                    particle_unet_labels_lst.append(
                        unet_mask[x_idx*step_size:(x_idx*step_size+tile_size), y_idx*step_size:(y_idx*step_size+tile_size)])

        return particle_in_focus_lst, image_lst, image_index_lst, image_corner_coords, particle_unet_labels_lst

