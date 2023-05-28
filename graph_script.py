# Import Transmitter module
from transmitter import Transmitter as Transmitter
# Import Surface module 
from surface import Surface as Surface
# Import Camera module
from camera import Camera as Camera
from rollingshutter import RollingShutter as RS
# Import NUmpy library
import numpy as np

# Library to plot 
import matplotlib.pyplot as plt


led1 = Transmitter(
        "Led1",
        position=[0, 0, 10],
        normal=[0, 0, -1],
        mlambert=1,
        wavelengths=[620, 530, 475],
        fwhm=[20, 30, 20],
        modulation='ieee16',
        frequency=4000,
        no_symbols=100,
        luminous_flux=5000
    )

surface = Surface(
    name="surfacePlaster",
    position=[0, 4, 2],
    reflectance='plaster',
    vertex1=[-2, 0, -2],
    vertex2=[2, 0, -2],
    vertex3=[2, 0, 2],
    vertex4=[-2, 0, 2]
    )

PIXEL_SIZE = 1.12e-6
MX = 1/PIXEL_SIZE  # number of pixels per unit distance in image coordinates in x direction
MY = 1/PIXEL_SIZE  # number of pixels per unit distance in image coordinates in y direction
FOCAL_LENGTH = 1e-3  # focal length
THETA_X = np.pi / 2.0  # roll angle
THETA_Y = np.pi  # pitch angle
THETA_Z = np.pi  # yaw angle
C = np.array([0, 0, 2])  # camera centre
IMAGE_WIDTH = 1200
IMAGE_HEIGTH = 800
PX= IMAGE_WIDTH/(2*MX)  # principal point x-coordinate
PY= IMAGE_HEIGTH/(2*MY)  # principal point y-coordinate

camera = Camera(
    name="camera1",
    focal_length=FOCAL_LENGTH,
    pixel_size=1/MX,
    px=PX,
    py=PY,
    mx=MX,
    my=MY,
    theta_x=THETA_X,
    theta_y=THETA_Y,
    theta_z=THETA_Z,
    centre=C,
    image_height=IMAGE_HEIGTH,
    image_width=IMAGE_WIDTH,    
    surface=surface,
    transmitter=led1,
    sensor='SonyStarvisBSI'
)
    
plt.plot(
        led1._array_wavelenghts,
        led1._spd_normalized[:, 0],
        color='r',
        linestyle='solid',
        label='Red-LED'
    )
plt.plot(
        led1._array_wavelenghts,
        led1._spd_normalized[:, 1],
        color='g',
        linestyle='solid',
        label='Green-LED'
    )
plt.plot(
        led1._array_wavelenghts,
        led1._spd_normalized[:, 2],
        color='b',
        linestyle='solid',
        label='Blue-LED'
    )

plt.plot(
        1e9 * camera._rgb_responsivity[:, 0],
        camera._rgb_responsivity[:, 1]/np.max(camera._rgb_responsivity[:, 1:3]),
        color='r',
        linestyle='dashed',
        label='Red-Detector'
    )
plt.plot(
        1e9 * camera._rgb_responsivity[:, 0],
        camera._rgb_responsivity[:, 2]/np.max(camera._rgb_responsivity[:, 1:3]),
        color='g',
        linestyle='dashed',
        label='Green-Detector'
    )
plt.plot(
        1e9 * camera._rgb_responsivity[:, 0],
        camera._rgb_responsivity[:, 3]/np.max(camera._rgb_responsivity[:, 1:3]),
        color='b',
        linestyle='dashed',
        label='Blue-Detector'
    )
plt.plot(
        surface._surface_reflectance[:, 0],
        surface._surface_reflectance[:, 1],
        color='black',
        linestyle='solid',
        label='Walls-Reflectance'
    )
#plt.title("Spectral Response of LEDs and Detectors", fontsize=20)
plt.legend(
    loc='upper right',
    fontsize=18,
    ncol=1,
    # bbox_to_anchor=[0, 1],
    # shadow=True, 
    # fancybox=True
    )
plt.xticks(
    # rotation=90,
    fontsize=28
    )
plt.yticks(
    # rotation=90,
    fontsize=28
    )        
plt.xlabel("Wavelength [nm]", fontsize=30)
plt.ylabel("Relative Spectrum and Response",  fontsize=30)
plt.grid()
plt.xlim([400, 700])
plt.ylim([0, 1.2])
plt.show()
