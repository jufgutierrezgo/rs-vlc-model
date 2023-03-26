from constants import Constants as Kt

# numeric numpy library
import numpy as np

# Library to plot the LED patter, SPD and responsivity
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/juanpc/python_phd/camera-models')

# import camera module
from camera_models import *  # our package


class Camera:
    """
    This class defines the camera properties
    """

    _DECIMALS = 2  # how many decimal places to use in print

    def __init__(
        self,
        name: str,
        focal_length: float,
        px: float,
        py: float,
        mx: float,
        my: float,
        theta_x: float,
        theta_y: float,
        theta_z: float,
        centre: np.ndarray,
        image_heigth: float,
        image_width: float
            ) -> None:

        self._name = name

        self._focal_length = np.float32(focal_length)        
        if self._focal_length <= 0:
            raise ValueError("The luminous flux must be non-negative.")
        
        # principal point x-coordinate
        self._px = px
        if self._px <= 0:
            raise ValueError("The PX must be non-negative.")

        # principal point y-coordinate
        self._py = py
        if self._py <= 0:
            raise ValueError("The PY must be non-negative.")

        # number of pixels per unit distance in image coordinates in x direction
        self._mx = mx
        if self._mx <= 0:
            raise ValueError("The MX must be non-negative.")

        # number of pixels per unit distance in image coordinates in y direction
        self._my = my
        if self._my <= 0:
            raise ValueError("The PX must be non-negative.")

        # roll angle
        self._theta_x = theta_x

        # pitch angle
        self._theta_y = theta_y

        # yaw angle
        self._theta_z = theta_z

        # camera centre
        self._centre = np.array(centre,  dtype=np.float32)
        if not (isinstance(self._centre, np.ndarray)) or self._centre.size != 3:
            raise ValueError("Camera centre must be an 1d-numpy array [x y z] dtype= float or int.")        

        # image heigth
        self._image_heigth = image_heigth
        if self._image_heigth <= 0:
            raise ValueError("The IMAGE LENGTH must be non-negative.")

        # image width
        self._image_width = image_width
        if self._image_width <= 0:
            raise ValueError("The IMAGE WIDTH must be non-negative.")
        

        self._calibration_kwargs = {"f": self._focal_length, "px": self._px, "py": self._py, "mx": self._mx, "my": self._my}
        self._rotation_kwargs = {"theta_x": self._theta_x, "theta_y": self._theta_y, "theta_z": self._theta_z}
        self._projection_kwargs = {**self._calibration_kwargs, **self._rotation_kwargs, "C": self._centre}

    def get_matrix(self) -> None:

        K = get_calibration_matrix(**self._calibration_kwargs)
        print("Calibration matrix (K):\n", K.round(self._DECIMALS))
        R = get_rotation_matrix(**self._rotation_kwargs)
        print("\nRotation matrix (R):\n", R.round(self._DECIMALS))
        P = get_projection_matrix(**self._projection_kwargs)
        print("\nProjection matrix (P):\n", P.round(self._DECIMALS))
        