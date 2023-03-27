from constants import Constants as Kt

# numeric numpy library
import numpy as np

# Library to plot the LED patter, SPD and responsivity
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/juanpc/python_phd/camera-models')

#import surface module
from surface import Surface as Surface

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
        image_width: float,
        surface: Surface
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

        self._surface = surface             
        if not type(surface) is Surface:
            raise ValueError(
                "Surface attribute must be an object type Surface.")
        

        self._projected_points = self._project_surface()
        print(self._projected_points)

    def _project_surface(self) -> np.ndarray:
        
        calibration_kwargs = {"f": self._focal_length, "px": self._px, "py": self._py, "mx": self._mx, "my": self._my}
        rotation_kwargs = {"theta_x": self._theta_x, "theta_y": self._theta_y, "theta_z": self._theta_z}
        projection_kwargs = {**calibration_kwargs, **rotation_kwargs, "C": self._centre}

        K = get_calibration_matrix(**calibration_kwargs)
        print("Calibration matrix (K):\n", K.round(self._DECIMALS))
        R = get_rotation_matrix(**rotation_kwargs)
        print("\nRotation matrix (R):\n", R.round(self._DECIMALS))
        P = get_projection_matrix(**projection_kwargs)
        print("\nProjection matrix (P):\n", P.round(self._DECIMALS))
        
        dx, dy, dz = np.eye(3)
        world_frame = ReferenceFrame(
            origin=np.zeros(3), 
            dx=dx, 
            dy=dy,
            dz=dz,
            name="World",
        )
        camera_frame = ReferenceFrame(
            origin=self._centre, 
            dx=R @ dx, 
            dy=R @ dy,
            dz=R @ dz,
            name="Camera",
        )
        Z = PrincipalAxis(
            camera_center=self._centre,
            camera_dz=camera_frame.dz,
            f=self._focal_length,
        )

        image_frame = ReferenceFrame(
            origin=Z.p - camera_frame.dx * self._px - camera_frame.dy * self._py, 
            dx=R @ dx, 
            dy=R @ dy,
            dz=R @ dz,
            name="Image",
        )

        image_plane = ImagePlane(
            origin=image_frame.origin, 
            dx=image_frame.dx, 
            dy=image_frame.dy, 
            heigth=self._image_heigth,
            width=self._image_width,
            mx=self._mx,
            my=self._my,
        )
        image = Image(heigth=self._image_heigth, width=self._image_width)
        polygon_surface = Polygon(self._surface._vertices)
        # square1 = Polygon(np.array([
        #    [-1.0, 5.0, 4.0],
        #    [1.0, 3.0, 5.0],
        #    [1.0, 2.0, 2.0],
        #    [-1.0, 4.0, 1.0],
        # ]))
        # square2 = Polygon(np.array([
        #    [-2.0, 4.0, 5.0],
        #    [2.0, 4.0, 5.0],
        #    [2.0, 4.0, 1.0],
        #    [-2.0, 4.0, 1.0],
        # ]))


        fig = plt.figure(figsize=(self._image_width, self._image_heigth))
        ax = fig.gca()
        image.draw()
        polygon_surface.draw(**projection_kwargs)
        
        #square1.draw(**projection_kwargs)
        #square2.draw(**projection_kwargs, color="tab:purple")
        ax.set_title("Projection of Squares in the Image")
        plt.tight_layout()
        plt.show()

        return np.array(polygon_surface.x_list)

