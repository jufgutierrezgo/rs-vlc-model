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
        
        self._compute_row_bins()

    def _compute_row_bins(self) -> None:
        """ This function computes the row bins respect to each of symbols. """

        t_symbol = 1/self._transmitter._frequency
        no_symbol = np.arange(1,self._transmitter._no_symbols+1)
        
        index_bins = (
            ((no_symbol*t_symbol) - self._t_start - self._t_exposure/2)
            / self._t_rowdelay        
            ).astype(int)
        
        print(index_bins)
        
            
