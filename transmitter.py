import sys
sys.path.insert(1, '/home/juanpc/python_phd/cruft_sample/python-vlc-rm/src')

from vlc_rm.constants import Constants as Kt

# Numeric Numpy library
import numpy as np

# Library to plot the LED patter, SPD and responsivity
import matplotlib.pyplot as plt

from scipy import stats

import luxpy as lx


class Transmitter:
    """
    This class defines the transmitter features
    """

    def __init__(
        self,
        name: str,
        position: np.ndarray,
        normal: np.ndarray,
        wavelengths: np.ndarray,
        fwhm: np.ndarray,
        mlambert: float = 1,        
        modulation: str = 'ieee16',
        luminous_flux: float = 1
            ) -> None:

        self._name = name

        self._position = np.array(position, dtype=np.float32)
        if self._position.size != 3:
            raise ValueError("Position must be an 1d-numpy array [x y z].")

        self._normal = np.array([normal],  dtype=np.float32)
        if not (isinstance(self._normal, np.ndarray)) or self._normal.size != 3:
            raise ValueError("Normal must be an 1d-numpy array [x y z] dtype= float or int.")        

        self._mlambert = np.float32(mlambert)
        if self._mlambert.size > 1:
            raise ValueError("Lambert number must be scalar float.")
        elif mlambert <= 0:
            raise ValueError("Lambert number must be greater than zero.")        

        self._wavelengths = np.array(wavelengths, dtype=np.float32)
        if self._wavelengths.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of wavelengths array must be equal to the number of LEDs.")
        elif (np.any(self._wavelengths > 780) or np.any(self._wavelengths < 380)):
            raise ValueError(
                "Wavelengths must be between 380nm and 780 nm.")

        self._fwhm = np.array(fwhm, dtype=np.float32)
        if self._fwhm.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of FWHM array must be equal to the number of LEDs.")
        elif np.any(self._fwhm <= 0):
            raise ValueError(
                "FWDM must be non-negative.")


        self._modulation = modulation
        # define the modulation
        if self._modulation == 'ieee16':
            self._constellation = Kt.IEEE_16CSK
            self._order_csk = 16
        elif self._modulation == 'ieee8':
            self._constellation = Kt.IEEE_8CSK
            self._order_csk = 8
        elif self._modulation == 'ieee4':
            self._constellation = Kt.IEEE_4CSK
            self._order_csk = 4
        else:
            raise ValueError("Modulation is not valid.")

        self._luminous_flux = np.float32(luminous_flux)        
        if self._luminous_flux <= 0:
            raise ValueError("The luminous flux must be non-negative.")

        # Initial function
        self._create_led_spd()        
        self._compute_iler(self._led_spd)
        self._avg_power_color()

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
    def mlambert(self) -> float:
        return self._mlambert

    @mlambert.setter
    def mlambert(self, mlabert):
        if mlabert <= 0:
            raise ValueError("Lambert number must be greater than zero.")
        self._mlambert = mlabert

    @property
    def wavelengths(self) -> np.ndarray:
        return self._wavelengths

    @wavelengths.setter
    def wavelengths(self, wavelengths):
        self._wavelengths = np.array(wavelengths,  dtype=np.float32)
        if self._wavelengths.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of wavelengths array must be equal to the number of LEDs.")

    @property
    def fwhm(self) -> np.ndarray:
        return self._fwhm

    @fwhm.setter
    def fwhm(self, fwhm):
        self._fwhm = np.array(fwhm,  dtype=np.float32)
        if self._fwhm.size != Kt.NO_LEDS:
            raise ValueError(
                "Dimension of FWHM array must be equal to the number of LEDs.") 

    @property
    def modulation(self) -> str:
        return self._modulation

    @modulation.setter
    def modulation(self, modulation):
        self._modulation = modulation
        # define the modulation
        if self._modulation == 'ieee16':
            self._constellation = Kt.IEEE_16CSK
            self._order_csk = 16
        elif self._modulation == 'ieee8':
            self._constellation = Kt.IEEE_8CSK
            self._order_csk = 8
        elif self._modulation == 'ieee4':
            self._constellation
            self._order_csk = 4
        else:
            print("Modulation name is not valid")

    @property
    def luminous_flux(self) -> float:
        return self._luminous_flux

    @luminous_flux.setter
    def luminous_flux(self, luminous_flux):
        if luminous_flux < 0:
            raise ValueError("The luminous flux must be non-negative.")
        self._luminous_flux = luminous_flux

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

    def led_pattern(self) -> None:
        """Function to create a 3d radiation pattern of the LED source.

        The LED for recurse channel model is assumed as lambertian radiator.
        The number of lambert defines the directivity of the light source.

        Parameters:
            m: Lambert number

        Returns: None.

        """

        theta, phi = np.linspace(0, 2 * np.pi, 40), np.linspace(0, np.pi/2, 40)
        THETA, PHI = np.meshgrid(theta, phi)
        R = (self._mlambert + 1)/(2*np.pi)*np.cos(PHI)**self._mlambert
        X = R * np.sin(PHI) * np.cos(THETA)
        Y = R * np.sin(PHI) * np.sin(THETA)
        Z = R * np.cos(PHI)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'),
            linewidth=0, antialiased=False, alpha=0.5)

        plt.show()

    def _create_led_spd(self):
        """
        This function creates the normilized spectrum of the LEDs 
        from central wavelengths and FWHM.
        """
        # Array for wavelenght points from 380nm to (782-2)nm with 1nm steps
        self._array_wavelenghts = np.arange(380, 781, 1)

        # Numpy Array to save the spectral power distribution of each color channel
        self._led_spd = np.zeros((self._array_wavelenghts.size, Kt.NO_LEDS))
        self._spd_normalized = np.zeros((self._array_wavelenghts.size, Kt.NO_LEDS))

        for i in range(Kt.NO_LEDS):
            # Arrays to estimates the normalized spectrum of LEDs
            self._led_spd[:, i] = stats.norm.pdf(
                self._array_wavelenghts, self._wavelengths[i], self._fwhm[i]/2)
            
            self._spd_normalized[:, i] = self._led_spd[:, i]/np.max(self._led_spd[:, i])
        
    def plot_spd_at_1w(self):
        # plot red spd data
        for i in range(Kt.NO_LEDS):
            plt.plot(self._array_wavelenghts, self._avg_power[i]*self._led_spd[:, i])
        
        plt.title("Spectral Power Distribution at 1W/channel")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Power [W]")
        plt.grid()
        plt.show()
    
    def plot_spd_normalized(self):
        # plot red spd data
        for i in range(Kt.NO_LEDS):
            plt.plot(
                self._array_wavelenghts,
                self._spd_normalized[:, i]
                )
        
        plt.title("Normalized Spectral Power Distribution")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Normalied Power [W]")
        plt.grid()
        plt.show()
    
    
    def _compute_iler(self, spd_data) -> None:        
        """
        This function computes the inverse luminous efficacy radiation (LER) matrix.
        This matrix has a size of NO_LEDS x NO_LEDS
        """
        self._photometric = lx.spd_to_power(
            np.vstack(
                    [
                        self._array_wavelenghts,
                        spd_data[:, 0]
                    ]),
            'pu'
        )
        self._radiometric = lx.spd_to_power(
            np.vstack(
                    [
                        self._array_wavelenghts,
                        spd_data[:, 0]
                    ]),
            'ru'
        )
        self._iler_matrix = np.zeros((Kt.NO_LEDS, Kt.NO_LEDS))

        for i in range(Kt.NO_LEDS):
            self._iler_matrix[i, i] = 1/lx.spd_to_ler(
                np.vstack(
                    [
                        self._array_wavelenghts,
                        spd_data[:, i]
                    ])
                )

    def _avg_power_color(self):
        """
        This function computes the average radiometric power emmitted by 
        each color channel in the defined constellation.
        """
        
        self._avg_lm = np.mean(
                    self._constellation,
                    axis=1
                    )
        self._avg_power = np.transpose(
            np.matmul(
                self._iler_matrix,
                self._avg_lm
                )
            )

        self._total_power = self._luminous_flux*np.sum(self._avg_power)
        # Manual setted of avg_power by each color channels
        #self._avg_power = np.array([1, 1, 1])
