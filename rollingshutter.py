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

#import transmitter module
from transmitter import Transmitter as Transmitter

#import surface module
from surface import Surface as Surface

from typing import Optional

from camera import Camera

import logging

# logging.basicConfig(format=FORMAT)

class Camera:
    """
    This class defines the camera properties
    """

    _DECIMALS = 2  # how many decimal places to use in print

    def __init__(
        self,
        name: str,
        t_exposure: float,
        t_rowdelay: float,
        t_start: float,
        iso: float
            ) -> None:
    
