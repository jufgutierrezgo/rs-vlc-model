# Import Transmitter module
from transmitter import Transmitter as Transmitter
# Import Surface module 
from surface import Surface as Surface
# Import Camera module
from camera import Camera as Camera
from rollingshutter import RollingShutter as RS
# Import NUmpy library
import numpy as np

transmitter = Transmitter(
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

# transmitter.plot_spd_normalized()
# transmitter.plot_spd_at_1lm()
# transmitter.plot_led_pattern()
print(transmitter)

surface = Surface(
    name="surfacePlaster",
    position=[0, 4, 2],
    reflectance='plaster',
    vertex1=[-2, 0, -2],
    vertex2=[2, 0, -2],
    vertex3=[2, 0, 2],
    vertex4=[-2, 0, 2]
    )
# surface.plot_reflectance()
surface._group_vertices()
print(surface)



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
    transmitter=transmitter,
    sensor='SonyStarvisBSI',
    idark=1e-14
)
camera.plot_responsivity()
camera.plot_image_intensity()

