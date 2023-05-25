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
        position=[3, 0, 10],
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
    position=[3, 4, 2],
    reflectance='plaster',
    vertex1=[-2, 0, -2],
    vertex2=[2, 0, -2],
    vertex3=[2, 0, 2],
    vertex4=[-2, 0, 2]
    )
# surface.plot_reflectance()
surface._group_vertices()
print(surface)



MX = 1/1e0  # number of pixels per unit distance in image coordinates in x direction
MY = 1/1e0  # number of pixels per unit distance in image coordinates in y direction
FOCAL_LENGTH = 1/MX  # focal length
PX= 3/MX  # principal point x-coordinate
PY= 2/MY  # principal point y-coordinate
THETA_X = np.pi / 2.0  # roll angle
THETA_Y = np.pi  # pitch angle
THETA_Z = np.pi  # yaw angle
C = np.array([3, 2, 2])  # camera centre
IMAGE_HEIGTH = 4
IMAGE_WIDTH = 6

RESOLUTION_X = 600
RESOLUTION_Y = 400

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
    resolution_x=RESOLUTION_X,
    resolution_y=RESOLUTION_Y,  
    surface=surface,
    transmitter=transmitter,
    sensor='SonyStarvisBSI',
    idark=1e-14
)

camera.plot_image_intensity()

