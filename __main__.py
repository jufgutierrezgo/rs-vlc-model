#Import module Transmitter
from transmitter import Transmitter as Transmitter

from surface import Surface as Surface

from camera import Camera as Camera

import numpy as np



transmitter = Transmitter(
        "Led1",
        position=[3, 4, 10],
        normal=[0, 0, -1],
        mlambert=1,
        wavelengths=[620, 530, 475],
        fwhm=[20, 45, 20],
        modulation='ieee16',
        luminous_flux=5000
    )
transmitter.plot_spd_normalized()


surface1 = Surface(
    name="surfacePlaster",
    position=[0, 0, 1],
    reflectance='plaster',
    vertex1=[-1.0, 5.0, 4.0],
    vertex2=[1.0, 3.0, 5.0],
    vertex3=[1.0, 2.0, 2.0],
    vertex4=[-1.0, 4.0, 1.0]
    )
surface2 = Surface(
    name="surfacePlaster",
    position=[0, 0, 1],
    reflectance='plaster',
    vertex1=[0.0, 1.0, 0.0],
    vertex2=[2.0, 1.0, 0.0],
    vertex3=[2.0, 1.0, 2.0],
    vertex4=[0.0, 1.0, 2.0]
    )
surface3 = Surface(
    name="surfacePlaster",
    position=[0, 0, 1],
    reflectance='plaster',
    vertex1=[1.0, 5.0, 0.0],
    vertex2=[5.0, 5.0, 0.0],
    vertex3=[5.0, 9.0, 4.0],
    vertex4=[1.0, 9.0, 4.0]
    )
surface3.plot_reflectance()
surface3._group_vertices()

FOCAL_LENGTH = 3  # focal length
PX= 3  # principal point x-coordinate
PY= 2  # principal point y-coordinate
MX = 1.0  # number of pixels per unit distance in image coordinates in x direction
MY = 1.0  # number of pixels per unit distance in image coordinates in y direction
#THETA_X = 0  # roll angle
#THETA_Y = 0.0  # pitch angle
#THETA_Z = 0  # yaw angle
THETA_X = np.pi / 2.0  # roll angle
THETA_Y = np.pi  # pitch angle
THETA_Z = np.pi  # yaw angle
C = np.array([3, -5, 2])  # camera centre
IMAGE_HEIGTH = 4
IMAGE_WIDTH = 6
RESOLUTION_HEIGTH = 80
RESOLUTION_WIDTH = 120


camera = Camera(
    name="camera1",
    focal_length=FOCAL_LENGTH,
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
    resolution_h=RESOLUTION_HEIGTH,
    resolution_w=RESOLUTION_WIDTH,
    surface=surface3,
    transmitter=transmitter
)



