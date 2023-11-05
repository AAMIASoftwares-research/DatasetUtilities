"""Here are defined the base classes that will hold the images and their
metadata.
Standardized methods are defined here to allow for easy access to the
images and their metadata.
"""

import os
from collections import UserDict
import numpy

class BoundingBoxDict(UserDict):
    """A dictionary that holds the bounding box of an image.

    Keys
    ----
    lower : numpy.ndarray
        The lower corner of the bounding box (x,y,z).
    upper : numpy.ndarray
        The upper corner of the bounding box (x,y,z).
    """
    def __init__(self, lower:numpy.ndarray, upper:numpy.ndarray):
        self.data = {"lower": lower, "upper": upper}

    def contains(self, location: numpy.ndarray):
        """Check if a location is inside the bounding box.

        Parameters
        ----------
        location : numpy.ndarray
            The location to check.
            location.shape = (3,) or (3,N) where N is the number of locations.
        """
        # Input rejection
        if location.ndim > 2:
            raise ValueError(f"Data must be a 2D array or a (x,y,z) array. Got {location.ndim}D array.")
        if len(location.shape) == 1:
            location = location.reshape(location.shape[0],1)
        if location.shape[0] != 3:
            raise ValueError(f"Data must have 3 rows. Got {location.shape[0]} rows.")
        # Check
        lower = self.data["lower"]
        upper = self.data["upper"]
        inside = numpy.logical_and.reduce((lower[0] <= location[:,0], location[:,0] <= upper[0],
                                            lower[1] <= location[:,1], location[:,1] <= upper[1],
                                            lower[2] <= location[:,2], location[:,2] <= upper[2]))
        return inside



    def __repr__(self):
        return f"BoundingBoxDict({self.data['lower']}, {self.data['upper']})"

    def __str__(self):
        return f"BoundingBoxDict({self.data['lower']}, {self.data['upper']})"

    def __getitem__(self, key):
        if key == "lower":
            return self.data["lower"]
        elif key == "upper":
            return self.data["upper"]
        else:
            raise KeyError(f"Key {key} not found in BoundingBoxDict.\nAvailable keys: lower, upper.")

    def __setitem__(self, key, value):
        if key == "lower":
            self.data["lower"] = value
        elif key == "upper":
            self.data["upper"] = value
        else:
            raise KeyError(f"Key {key} not found in BoundingBoxDict.\nAvailable keys: lower, upper.")

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __contains__(self, key):
        return key in self.data

#####################
# CT/CCTA BASIC IMAGE
#####################

class ImageCT(object):
    """Base class for CT/CCTA single images.

    Attributes
    ----------
    name : str
        Name of the image.
    data : numpy.ndarray
        The image data, a (X x Y x Z) array in int16 format.
        Hounsfield units are stored in the image data.
    origin : numpy.ndarray
        The origin of the image in RAS coordinates (x,y,z).
        millimeters coordinates are stored in the origin.
    spacing : numpy.ndarray
        The spacing of the image in RAS coordinates (x,y,z).
        millimeters coordinates are stored in the spacing.
    bounding_box : BoundingBoxDict
        The bounding box of the image in RAS coordinates.
        millimeters coordinates are stored in the bounding box.
    affine_ijk2ras : numpy.ndarray
        The affine transformation matrix from IJK of the numpy.ndarray data to RAS coordinates.
    affine_ras2ijk : numpy.ndarray
        The affine transformation matrix from RAS coordinates to IJK of the numpy.ndarray data.
    """
    name: str # Name of the image
    data: numpy.ndarray(dtype=numpy.int16) # dimensions: (x, y, z) in RAS coordinate system
    origin: numpy.ndarray # dimensions: (x, y, z) in RAS coordinate system
    spacing: numpy.ndarray # dimensions: (x, y, z) in RAS coordinate system
    bounding_box: BoundingBoxDict # {"lower": numpy.ndarray, "upper": numpy.ndarray} in RAS coordinate system
    # Affine Transformations
    affine_ijk2ras: numpy.ndarray # 4x4 affine transformation matrix from IJK to RAS
    affine_ras2ijk: numpy.ndarray # 4x4 affine transformation matrix from RAS to IJK


    def __init__(self, name:str, data:numpy.ndarray, origin:numpy.ndarray, spacing:numpy.ndarray):
        """Initialize the ImageCT object.
        
        Parameters
        ----------
        name : str
            Name of the image.
        data : numpy.ndarray
            The image data, a (X x Y x Z) array in int16 format, Hounsfield units.
        origin : numpy.ndarray
            The origin of the image in RAS coordinates (x,y,z), millimeters.
        spacing : numpy.ndarray
            The spacing of the image in RAS coordinates (x,y,z), millimeters.
        """
        self.name = name
        self.data = data.astype(numpy.int16) if data.dtype != numpy.int16 else data
        self.origin = origin.flatten()
        self.spacing = spacing.flatten()
        self.bounding_box = self._set_bounding_box()
        # Affine Transformations
        self.affine_ijk2ras = self._set_affine_ijk2ras()
        self.affine_ras2ijk = self._set_affine_ras2ijk()

    def _set_bounding_box(self):
        """Set the bounding box of the image in RAS coordinates.
        """
        lower = self.origin
        upper = self.origin + self.spacing * self.data.shape
        return BoundingBoxDict(lower, upper)
    
    def _set_affine_ijk2ras(self):
        """Set the affine transformation matrix from IJK to RAS coordinates.
        """
        affine = numpy.eye(4)
        affine[0:3, 0:3] = numpy.diag(self.spacing)
        affine[0:3, 3] = self.origin
        return affine
    
    def _set_affine_ras2ijk(self):
        """Set the affine transformation matrix from RAS to IJK coordinates.
        """
        affine = numpy.eye(4)
        affine[0:3, 0:3] = numpy.diag(1/self.spacing)
        affine[0:3, 3] = -self.origin/self.spacing
        return affine
    
    def transform_affine(self, points: numpy.ndarray, affine: numpy.ndarray):
        # NOTE TO MYSELF:
        # THIS METHOD SHOULD NOT STAY HERE. wOULD BE BETTER TO CREATE A
        # MODULE, DETACHED FROM ANYTHNG ELSE, TO DO THE WORK
        # hOWEVER, IT MAY BE OVERKILL, SO I'LL KEEP IT HERE FOR NOW. 
        """Transform the input points with the specified affine belonging to the object instance.

        Parameters
        ----------
        points : numpy.ndarray
            The points to transform.
            points.shape = (3,) or (3, N) where N is the number of locations.

        affine : numpy.ndarray
            The affine transformation matrix to use.
        """
        # Input rejection
        if points.ndim > 2:
            raise ValueError(f"Data must be a 2D array or a (x,y,z) array. Got {points.ndim}D array.")
        if len(points.shape) == 1:
            points = points.reshape(points.shape[0],1)
        if points.shape[0] != 3:
            raise ValueError(f"Data must have 3 rows. Got {points.shape[0]} rows.")
        if affine.shape != (4,4):
            raise ValueError(f"Affine must be a 4x4 matrix. Got {affine.shape} matrix.")
        # Transform
        points = numpy.vstack((points, numpy.ones(points.shape[1])))
        points = numpy.matmul(affine, points)
        return points[0:3, :]

    def sample(self, location: numpy.ndarray, interpolation: str = "nearest"):
        """Sample the image data at specific 3D location(s).

        Interpolation is applied to the image data to get the value at the
        specified location(s), if specified. "nearest" is the default and
        applies no interpolation.

        Parameters
        ----------
        location : numpy.ndarray
            The 3D location(s) to sample the image data.
            The location(s) must be in RAS coordinates (x,y,z), millimeters.
            location.shape = (3,) or (3, N) where N is the number of locations.
        interpolation : str, optional
            The interpolation method to use. Default is "nearest".
            Available methods are:
            - "nearest" : Nearest neighbor interpolation. Fastest.
            - "linear" : Linear interpolation. Slower.
        """
        # Input rejection
        if location.ndim > 2:
            raise ValueError(f"Data must be a 2D array or a (x,y,z) array. Got {location.ndim}D array.")
        if len(location.shape) == 1:
            location = location.reshape(location.shape[0],1)
        if location.shape[0] != 3:
            raise ValueError(f"Data must have 3 rows. Got {location.shape[0]} rows.")
        if not interpolation in ["nearest", "linear"]:
            raise ValueError(f"Interpolation method {interpolation} not available. Available methods are: nearest, linear.")
        # Transform location to IJK, keeping the decimals
        location_ras = location
        location = self.transform_affine(location, self.affine_ras2ijk)
        # Allocate output memory - locations outside the image will be set to the minimum of the image
        output = numpy.ones(location.shape[1]) * numpy.min(self.data)
        # check wgich input locations are inside the bbox
        input_inside = self.bounding_box.contains(location_ras)
        # Sample
        if interpolation == "nearest":
            location = numpy.round(location).astype(numpy.int)
            # we have to cycle to keep in consideration the data that are outside the image
            for i in range(location.shape[1]):
                if not input_inside[i]:
                    continue
                output[i] = self.data[location[0,i], location[1,i], location[2,i]]
        elif interpolation == "linear":
            location_floor = numpy.floor(location).astype(numpy.int)
            location_ceil = numpy.ceil(location).astype(numpy.int)
            location_floor_percent = (location-location_floor)/(location_ceil-location_floor)
            for i in range(location.shape[1]):
                if not input_inside[i]:
                    continue
                v_000 = self.data[location_floor[0,i], location_floor[1,i], location_floor[2,i]] # 0,0,0
                w_000 = location_floor_percent[0,i] * location_floor_percent[1,i] * location_floor_percent[2,i]
                v_100 = self.data[location_ceil[0,i], location_floor[1,i], location_floor[2,i]] # 1,0,0
                w_100 = (1-location_floor_percent[0,i]) * location_floor_percent[1,i] * location_floor_percent[2,i]
                v_010 = self.data[location_floor[0,i], location_ceil[1,i], location_floor[2,i]] # 0,1,0
                w_010 = location_floor_percent[0,i] * (1-location_floor_percent[1,i]) * location_floor_percent[2,i]
                v_110 = self.data[location_ceil[0,i], location_ceil[1,i], location_floor[2,i]] # 1,1,0
                w_110 = (1-location_floor_percent[0,i]) * (1-location_floor_percent[1,i]) * location_floor_percent[2,i]
                v_001 = self.data[location_floor[0,i], location_floor[1,i], location_ceil[2,i]] # 0,0,1
                w_001 = location_floor_percent[0,i] * location_floor_percent[1,i] * (1-location_floor_percent[2,i])
                v_101 = self.data[location_ceil[0,i], location_floor[1,i], location_ceil[2,i]] # 1,0,1
                w_101 = (1-location_floor_percent[0,i]) * location_floor_percent[1,i] * (1-location_floor_percent[2,i])
                v_011 = self.data[location_floor[0,i], location_ceil[1,i], location_ceil[2,i]] # 0,1,1
                w_011 = location_floor_percent[0,i] * (1-location_floor_percent[1,i]) * (1-location_floor_percent[2,i])
                v_111 = self.data[location_ceil[0,i], location_ceil[1,i], location_ceil[2,i]] # 1,1,1
                w_111 = (1-location_floor_percent[0,i]) * (1-location_floor_percent[1,i]) * (1-location_floor_percent[2,i])
                output[i] = v_000*w_000 + v_100*w_100 + v_010*w_010 + v_110*w_110 + v_001*w_001 + v_101*w_101 + v_011*w_011 + v_111*w_111
        # Out
        if output.shape[0] == 1:
            return output[0]


    def __repr__(self):
        return f"ImageCT({self.name}, data of shape {self.data.shape}, origin (mm) {self.origin}, spacing (mm) {self.spacing}, bounding box {self.bounding_box})"
    
    def __str__(self):
        return f"ImageCT({self.name}, data of shape {self.data.shape}, origin (mm) {self.origin}, spacing (mm) {self.spacing}, bounding box {self.bounding_box})"
    


class ImageSequenceCT(object):
    pass
