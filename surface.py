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
        vertex1: np.ndarray,
        vertex2: np.ndarray,
        vertex3: np.ndarray,
        vertex4: np.ndarray
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

        
        self._vertex1 = np.array(vertex1, dtype=np.float32)
        if self._vertex1.size != 3:
            raise ValueError("Vertex1 must be an 1d-numpy array [x y z].")
        
        self._vertex2 = np.array(vertex2, dtype=np.float32)
        if self._vertex2.size != 3:
            raise ValueError("Vertex2 must be an 1d-numpy array [x y z].")
        
        self._vertex3 = np.array(vertex3, dtype=np.float32)
        if self._vertex3.size != 3:
            raise ValueError("Vertex3 must be an 1d-numpy array [x y z].")
        
        self._vertex4 = np.array(vertex4, dtype=np.float32)
        if self._vertex4.size != 3:
            raise ValueError("Vertex4 must be an 1d-numpy array [x y z].")

    #create an array with the four vertices
    self._group_vertices()

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
            f'Reflectance Material: {self._reflectance} \n'
            f'Size [x y]: {self._size} \n'                     
            )
    
    def plot_reflectance(self) -> None:
        plt.plot(
            self._surface_reflectance[:, 0],
            self._surface_reflectance[:, 1],
            color='black',
            linestyle='solid',
            label='Surface-Reflectance'
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
            fontsize=10
            )
        plt.yticks(
            # rotation=90,
            fontsize=10
            )
        plt.title("Reflectance of the surface", fontsize=20)        
        plt.xlabel("Wavelength [nm]", fontsize=15)
        plt.ylabel("Relative Response",  fontsize=15)
        plt.grid()
        plt.xlim([400, 700])
        plt.ylim([0, 1.15])
        plt.show()

    def _group_vertices(self) -> None:
        self._vertices =  np.concatenate((
                [self._vertex1],
                [self._vertex2],
                [self._vertex3],
                [self._vertex4]
                ), 
                axis=0    
            )

        # print(self._vertices)
        
        

