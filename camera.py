from constants import Constants as Kt

# numeric numpy library
import numpy as np

# Library to plot the LED patter, SPD and responsivity
import matplotlib.pyplot as plt


class Camera:
    """
    This class defines the camera properties
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray
            ) -> None:

