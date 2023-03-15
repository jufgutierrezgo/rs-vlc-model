from constants import Constants as Kt

# Library to plot the LED patter, SPD and responsivity
import matplotlib.pyplot as plt

import numpy as np

from numpy import loadtxt


class Surface:
    """
    This class defines the transmitter features
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray,
        normal: np.ndarray,
        reflectance: str,
        size: np.ndarray        
            ) -> None:

        self._name = name

        self._position = np.array(position, dtype=np.float32)
        if self._position.size != 3:
            raise ValueError("Position must be an 1d-numpy array [x y z].")

        self._normal = np.array(normal,  dtype=np.float32)
        if not (isinstance(self._normal, np.ndarray)) or self._normal.size != 3:
            raise ValueError("Normal must be an 1d-numpy array [x y z] dtype= float or int.")        

        self._reflectance = reflectance        
        if self._reflectance == 'plaster':
            # load the spectral reflectance of Plaster material
            self._surface_reflectance = loadtxt(
                    Kt.REFLECTANCE_PATH+'Interp_ReflecPlaster.txt'
                )            
        elif self._reflectance == 'floor':
            # load the spectral reflectance of Plaster material
            self._surface_reflectance = loadtxt(
                Kt.REFLECTANCE_PATH+'Interp_ReflecFloor.txt'
                )
        else:
            raise ValueError("Reflectance name is not valid.")

        self._size = np.array(size)
        if self._size.size != 2:
            raise ValueError(
                "Size of the rectangular surface must be an 1d-numpy array [x y]")


    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, position):
        self._position = np.array(position,  dtype=np.float32)
        if self._position.size != 3:
            raise ValueError("Position must be a 3d-numpy array.")

    @property
    def normal(self) -> np.ndarray:
        return self._normal

    @normal.setter
    def normal(self, normal):
        self._normal = np.array(normal,  dtype=np.float32)
        if self._normal.size != 3:
            raise ValueError("Normal must be a 3d-numpy array.")

    @property
    def reflectance(self) -> str:
        return self._reflectance

    @reflectance.setter
    def reflectance(self, reflectance):
        self._reflectance = reflectance        
        if self._reflectance == 'plaster':
            # load the spectral reflectance of Plaster material
            self._surface_reflectance = loadtxt(
                    Kt.REFLECTANCE_PATH+'Interp_ReflecPlaster.txt'
                )            
        elif self._reflectance == 'floor':
            # load the spectral reflectance of Plaster material
            self._surface_reflectance = loadtxt(
                Kt.REFLECTANCE_PATH+'Interp_ReflecFloor.txt'
                )
        else:
            raise ValueError("Reflectance name is not valid.")
    
    @property
    def size(self) -> np.ndarray:
        """The size property"""
        return self._size

    @size.setter
    def size(self, size):
        self._size = np.array(size)
        if self._size.size != 2:
            raise ValueError(
                "Size of the rectangular surface must be an 1d-numpy array [x y]")

    def __str__(self) -> str:
        return (
            f'\n List of parameters for LED transmitter: \n'
            f'Name: {self._name}\n'
            f'Position [x y z]: {self._position} \n'
            f'Normal Vector [x y z]: {self._normal} \n'
            f'Lambert Number: {self._mlambert} \n'            
            f'Central Wavelengths [nm]: {self._wavelengths} \n'
            f'FWHM [nm]: {self._fwhm}\n'
            f'Luminous Flux [lm]: {self._luminous_flux}\n'
            f'ILER [W/lm]: \n {self._iler_matrix} \n'
            f'Average Power per Channel Color: \n {self._luminous_flux*self._avg_power} \n'
            f'Total Power emmited by the Transmitter [W]: \n {self._total_power} \n'
            
        )
    
    def plot_reflectance(self) -> None:
        plt.plot(
            self._surface_reflectance[:, 0],
            self._surface_reflectance[:, 1],
            color='black',
            linestyle='solid',
            label='Walls-Reflectance'
            )        

        #plt.title("Spectral Response of LEDs and Detectors", fontsize=20)
        plt.legend(
            loc='upper right',
            fontsize=14,
            ncol=1,
            # bbox_to_anchor=[0, 1],
            # shadow=True, 
            # fancybox=True
            )
        plt.xticks(
            # rotation=90,
            fontsize=18
            )
        plt.yticks(
            # rotation=90,
            fontsize=18
            )        
        plt.xlabel("Wavelength [nm]", fontsize=20)
        plt.ylabel("Relative Response",  fontsize=20)
        plt.grid()
        plt.xlim([400, 700])
        plt.ylim([0, 1.15])
        plt.show()