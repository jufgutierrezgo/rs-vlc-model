from constants import Constants as Kt

# numeric numpy library
import numpy as np

# Library to plot 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Scipy import
import scipy.signal as signal
from scipy.stats import norm
# Skiimage import
from skimage import data


import sys
sys.path.insert(0, './camera-models')

# import transmitter module
from transmitter import Transmitter as Transmitter
# import surface module 
from surface import Surface as Surface
# import camera module 
from camera import Camera

from typing import Optional



import logging

# logging.basicConfig(format=FORMAT)

class RollingShutter:
    """
    This class defines the rolling shutter adquisition properties
    """    

    def __init__(
        self,
        name: str,
        t_exposure: float,
        t_rowdelay: float,
        t_start: float,
        iso: float,
        transmitter: Transmitter,
        camera: Camera        
            ) -> None:

        self._name = name

        self._t_exposure = np.float32(t_exposure)        
        if self._t_exposure <= 0:
            raise ValueError("The exposure time must be a float non-negative.")
        
        self._t_rowdelay = np.float32(t_rowdelay)        
        if self._t_rowdelay <= 0:
            raise ValueError("The row delay time must be a float non-negative.")

        self._t_start = np.float32(t_start)        
        if self._t_rowdelay <= 0:
            raise ValueError(
                "The row adquisition start time must be non-negative."
                )

        self._iso = np.int(iso)        
        if self._iso <= 0:
            raise ValueError(
                "The ISO must be integer non-negative."
                )
        
        self._transmitter = transmitter
        if not type(transmitter) is Transmitter:
            raise ValueError(
                "Transmiyyer attribute must be an object type Transmitter.")
        
        self._camera = camera
        if not type(camera) is Camera:
            raise ValueError(
                "Camera attribute must be an object type Camera.")
        
        self._index_row_bins = self._compute_row_bins()
        self._current_image = self._compute_image_current(
            symbols_csk=self._transmitter._symbols_csk,
            im_bayern_crostalk=self._camera._image_bayern_crosstalk,
            im_gain=camera._power_image,
            index_bins=self._index_row_bins,
            height=self._camera._resolution_h,
            width=self._camera._resolution_w
        )

    def _compute_row_bins(self) -> np.ndarray:
        """ This function computes the row bins respect to each of symbols. """

        # compute the time of each symbol. It is equal to 1/f.
        t_symbol = 1/self._transmitter._frequency

        # compute the symbol corresponding to the last row 
        last_symbol = int(
                ((self._t_rowdelay * self._camera._resolution_h)
                + self._t_start) / t_symbol 
            ) + 1

        no_symbol = np.arange(0, last_symbol+1)
        
        index_bins = (
            ((no_symbol*t_symbol) - self._t_start - self._t_exposure/2)
            / self._t_rowdelay        
            ).astype(int) + 1
        index_bins[-1] = self._camera._resolution_h
        index_bins[0] = 0

        print("Row bins:")
        print(index_bins)

        return index_bins

    def _compute_image_current(
            self, 
            symbols_csk, 
            im_bayern_crostalk, 
            im_gain,
            index_bins,
            height,
            width) -> np.ndarray:
        """ This function computes the image with the transmitter symbols. """

        image_current = np.zeros((height, width))
        image_color = np.zeros((height, width))

        for symbol in np.arange(1,np.size(index_bins)):
            image_color[
                index_bins[symbol-1]: index_bins[symbol], :
                ] = np.sum(
                    np.multiply(
                        im_bayern_crostalk[
                            index_bins[symbol-1]: index_bins[symbol], :, :],
                        symbols_csk[:, symbol-1].reshape(1, 1, 3)
                    ),
                axis=2
                )
            # print(symbols_csk[:, symbol-1].reshape(1, 1, 3))

        image_current = im_gain * image_color
        # image_current = image_color

        return image_current

    def plot_current_image(self):
        """ Plot the image of the photocurrent by each pixel. """        
        
        # Plot power image
        plt.imshow(self._current_image, cmap='gray', interpolation='nearest')
        plt.title("Image of the normalized received power")        
        plt.show()