from dataclasses import dataclass

import os

import numpy as np

@dataclass(frozen=True)
class Constants:
    """
    Global constants.
    """
    # Array with normal vectors for each wall.
    # TODO: consider using a named tuple
    NORMAL_VECTOR_WALL = [
        [0, 0, -1], [0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, 1]
        ]
    # directory root of the project
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # directory to save channel impulse response raw data
    SENSOR_PATH = ROOT_DIR + "/sensors/"
    # directory to save histograms and graphs
    REPORT_PATH = ROOT_DIR + "/report/"
    # directory of tests
    TEST_PATH = ROOT_DIR + "/tests/"
    # Numbers of LED (Transmission channels)
    NO_LEDS = 3
    # Numbers of Photodetector Channels
    NO_DETECTORS = 3
    # Speed of light
    SPEED_OF_LIGHT = 299792458
    # IEEE 16-CSK constellation
    IEEE_16CSK = np.transpose(
        np.array([
            [1/3, 1/3, 1/3],
            [1/9, 7/9, 1/9],
            [0, 2/3, 1/3],
            [1/3, 2/3, 0],
            [1/9, 4/9, 4/9],
            [0, 1, 0],
            [4/9, 4/9, 1/9],
            [4/9, 1/9, 4/9],
            [0, 1/3, 2/3],        
            [1/9, 1/9, 7/9],
            [0, 0, 1],
            [1/3, 0, 2/3],
            [2/3, 1/3, 0],
            [7/9, 1/9, 1/9],
            [2/3, 0, 1/3],
            [1, 0, 0]
            ])
        )
    # IEEE 8-CSK constellation
    IEEE_8CSK = np.transpose(
        np.array([
            [0, 1, 0],
            [0, 2/3, 1/3],
            [1/3, 2/3, 0],
            [11/18, 5/18, 2/18],
            [0, 0, 1],
            [2/18, 5/18, 11/18],
            [1/2, 0, 1/2],
            [1, 0, 0]
            ])
        )
    # IEEE 4-CSK constellation
    IEEE_4CSK = np.transpose(
        np.array([
            [1/3, 1/3, 1/3],            
            [0, 1, 0],            
            [0, 0, 1],            
            [1, 0, 0]
            ])
        )