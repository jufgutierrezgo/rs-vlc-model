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
        resolution_h: int,
        resolution_w: int,
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
        
        # resolution height
        self._resolution_h = resolution_h
        if self._resolution_h <= 0:
            raise ValueError("The RESOLUTION H must be an positive integer.")

        # resolution width
        self._resolution_w = resolution_w
        if self._resolution_w <= 0:
            raise ValueError("The RESOLUTION H must be an positive integer.")

        self._surface = surface             
        if not type(surface) is Surface:
            raise ValueError(
                "Surface attribute must be an object type Surface.")
        

        self._projected_points = self._project_surface()
        print("\n Projected Point onto Image Plane:")
        print(self._projected_points)
        self._pixels_inside =  self._points_inside()
        print("\n Pixels inside of the polygon:")
        print(self._pixels_inside)
        self.plot_binary_image(self._pixels_inside, self._resolution_h, self._resolution_w)

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

        print("Bases of the Image frame")
        print(image_frame.dx, image_frame.dy, image_frame.dz)
        grid3d_image = self._grid3d_image_plane(
            image_frame.dx, 
            image_frame.dy,
            image_frame.origin 
            )
        print(image_frame.origin)
        print(grid3d_image)
        
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

        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection="3d")
        world_frame.draw3d()
        camera_frame.draw3d()
        image_frame.draw3d()
        Z.draw3d()
        image_plane.draw3d()
        polygon_surface.draw3d(pi=image_plane.pi, C=self._centre)        
        ax.view_init(elev=45.0, azim=45.0)
        ax.set_title("CCD Camera Geometry")
        plt.tight_layout()
        plt.show()

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

    def _points_inside(self):
        
        # Compute the size of each cell
        pixel_size_x = self._image_width / self._resolution_w
        pixel_size_y = self._image_heigth / self._resolution_h

        # Create a grid of cell centers
        pixel_centers_x = np.linspace(
            pixel_size_x/2, 
            self._image_width-pixel_size_x/2, 
            self._resolution_w
            )
        pixel_centers_y = np.linspace(
            pixel_size_y/2, 
            self._image_heigth-pixel_size_y/2, 
            self._resolution_h
            )
        
        pixel_centers_xx, pixel_centers_yy = np.meshgrid(pixel_centers_x, pixel_centers_y)
        
        # Display the cell centers
        # print("Pixel's center X (meshgrid):")
        # print(pixel_centers_xx)
        # print("Pixel's center Y  (meshgrid):")
        # print(pixel_centers_yy)

        # poly_vertices = np.array([[1, 1], [3, 1], [3, 3], [1, 3]])
        # print("Pixel's center X:")
        # print(range(len(pixel_centers_x)))
        # print("Pixel's center Y:")
        # print(range(len(pixel_centers_y)))
        
        # Get points inside polygon
        points_inside = []
        pixels_inside = []
        for i in range(len(pixel_centers_x)):
            for j in range(len(pixel_centers_y)):                
                if self._point_inside_polygon(
                        pixel_centers_xx[j, i],
                        pixel_centers_yy[j, i],
                        self._projected_points):
                    points_inside.append([pixel_centers_xx[j,i], pixel_centers_yy[j,i]])
                    pixels_inside.append(np.array((j, i)))

        points_inside = np.transpose(np.array(points_inside))
        pixels_inside = np.transpose(np.array(pixels_inside))

        #PLot area inside of the polygon
        plt.scatter(pixel_centers_xx, pixel_centers_yy, s=1)
        plt.scatter(points_inside[0, :], points_inside[1, :], s=1)
        plt.show()
        print("\n Points inside polygon:")
        print(points_inside)
        # print("Pixels inside polygon:")
        # print(pixels_inside)

        return pixels_inside

    def _point_inside_polygon(self, x, y, poly):
        """
        Return True if the point x, y is inside the polygon defined 
        by the list of vertices poly.
        """

        n = len(poly)
        inside = False

        p1x, p1y = poly[0]
        for i in range(n+1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= x_inters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
    
    def plot_binary_image(self, pixels, heigth, width):
        
    
        binary_image = np.zeros((heigth, width))
        binary_image[pixels[0, :], pixels[1, :]] = 1

        # Plot binary matrix
        plt.imshow(binary_image, cmap='gray', interpolation='nearest')
        plt.title("BInary image of the area projected")
        # plt.scatter(pixels[1,:],pixels[0,:])
        # plt.xlim([0, self._resolution_w])
        # plt.ylim([0, self._resolution_h])
        plt.show()
    
    def _grid3d_image_plane(self, dx, dy, origin) -> np.ndarray:
        """
        Return an array with the XYZ coordinates of the pixels in
        the image plane.
        """

        grid3d_image_plane = np.zeros((self._image_width, self._image_heigth, 3))
        
        for i in range(self._image_width):
            for j in range(self._image_heigth):
                grid3d_image_plane[i, j, :] = origin + i*dx + j*dy + dx/2 + dy/2
        
        return grid3d_image_plane