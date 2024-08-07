"""Here are defined the base classes that will hold the images and their
metadata.
Standardized methods are defined here to allow for easy access to the
images and their metadata.
"""

import os, copy
from collections import UserDict
import numpy
import scipy.ndimage

from mpl_toolkits.mplot3d.art3d import Line3DCollection

from HearticDatasetManager.affine import compose_affines, apply_affine_3d

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
        # Allocate output
        inside = numpy.ones(location.shape[1], dtype=bool)
        # Check
        lower = self.data["lower"]
        upper = self.data["upper"]
        inside[:] = numpy.logical_and(inside, lower[0] < location[0,:])
        inside[:] = numpy.logical_and(inside, location[0,:] < upper[0])
        inside[:] = numpy.logical_and(inside, lower[1] < location[1,:])
        inside[:] = numpy.logical_and(inside, location[1,:] < upper[1])
        inside[:] = numpy.logical_and(inside, lower[2] < location[2,:])
        inside[:] = numpy.logical_and(inside, location[2,:] < upper[2])
        return inside

    def get_artist(self) -> Line3DCollection:
        """Get a matplotlib.collections.LineCollection artist to plot the bounding box.
        """
        lx, ly, lz = self.data["lower"]
        ux, uy, uz = self.data["upper"]
        lines = [
            # bottom square
            [(lx, ly, lz),(ux, ly, lz)],
            [(lx, ly, lz),(lx, uy, lz)],
            [(lx, uy, lz),(ux, uy, lz)],
            [(ux, ly, lz),(ux, uy, lz)],
            # bottom to top
            [(lx, ly, lz),(lx, ly, uz)],
            [(ux, ly, lz),(ux, ly, uz)],
            [(lx, uy, lz),(lx, uy, uz)],
            [(ux, uy, lz),(ux, uy, uz)],
            # top square
            [(lx, ly, uz),(ux, ly, uz)],
            [(lx, ly, uz),(lx, uy, uz)],
            [(lx, uy, uz),(ux, uy, uz)],
            [(ux, ly, uz),(ux, uy, uz)]
        ]
        collection = Line3DCollection(
            lines,
            color="fuchsia",
            linestyle="--",
            linewidth=0.7,
            alpha=0.8
        )
        return collection
    
    def get_xlim(self) -> tuple[float, float]:
        """Get the limits of the bounding box in the x dimension.
        """
        lx = self.data["lower"][0]
        ux = self.data["upper"][0]
        return (lx, ux)
    
    def get_ylim(self) -> tuple[float, float]:
        """Get the limits of the bounding box in the y dimension.
        """
        ly = self.data["lower"][1]
        uy = self.data["upper"][1]
        return (ly, uy)

    def get_zlim(self) -> tuple[float, float]:
        """Get the limits of the bounding box in the z dimension.
        """
        lz = self.data["lower"][2]
        uz = self.data["upper"][2]
        return (lz, uz)

    def __repr__(self):
        return f"BoundingBoxDict({self.data['lower']}, {self.data['upper']})"

    def __str__(self):
        return f"BoundingBoxDict(lower: {self.data['lower']}, upper: {self.data['upper']})"

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
        A unique identifier for the image.
        This could be a filename, a URI, a DICOM SOP Instance UID,
        or some other unique identifier.
    data : numpy.ndarray
        The image data, a (X x Y x Z) array in int16 format.
        Hounsfield units are stored in the image data.
    shape : tuple
        The shape of the image data.
    origin : numpy.ndarray
        The origin of the image in RAS coordinates (x,y,z).
        The physical world coordinates of the first voxel in the
        image data in millimeters units.
    spacing : numpy.ndarray
        The spacing of the image in RAS coordinates (x,y,z).
        The physical distance between voxels in each dimension
        in millimeters units.
    bounding_box : BoundingBoxDict
        The bounding box of the image in RAS coordinates.
        Lower and upper coordinates are stored in the bounding box in mm units.
        Does not take into account the orientation of the image.
    affine_ijk2ras_direction : numpy.ndarray
        The diagonal matrix for IJK <-> RAS direction.
    affine_ijk2ras : numpy.ndarray
        The affine transformation matrix from IJK of the image data to RAS coordinates.
        This matrix encapsulates the origin, spacing, and orientation.
    affine_ras2ijk : numpy.ndarray
        The affine transformation matrix from RAS coordinates to IJK of the numpy.ndarray data.
    """
    name: str
    data: numpy.ndarray
    shape: tuple
    origin: numpy.ndarray
    spacing: numpy.ndarray
    bounding_box: BoundingBoxDict
    # Affine Transformations
    affine_ijk2ras_direction: numpy.ndarray
    affine_ijk2ras: numpy.ndarray
    affine_ras2ijk: numpy.ndarray

    def __init__(self, name:str, data:numpy.ndarray, origin:numpy.ndarray, spacing:numpy.ndarray, affine_ijk2ras_direction:numpy.ndarray=numpy.eye(4)):
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
        affine_ijk2ras_direction : numpy.ndarray, optional
            The diagonal matrix for IJK <-> RAS direction. Default is numpy.eye(4).
        """
        self.name = name
        self.data = data.astype(numpy.int16) if data.dtype != numpy.int16 else data
        self.shape = data.shape
        self.origin = origin.flatten()
        self.spacing = spacing.flatten()
        # Affine Transformations
        self.affine_ijk2ras_direction = affine_ijk2ras_direction
        self.affine_ijk2ras = self._set_affine_ijk2ras()
        self.affine_ras2ijk = self._set_affine_ras2ijk()
        # Bounding boxes
        self.bounding_box = self._set_bounding_box()
        self._conservative_b_box = self._set_bounding_box()
        self._conservative_b_box["lower"] += 0.001
        self._conservative_b_box["upper"] -= 0.001

    def _set_bounding_box(self):
        """Set the bounding box of the image in RAS coordinates.
        """
        low_ijk = numpy.zeros(shape=(3,))
        high_ijk = numpy.array(self.shape)-1
        # Transform to RAS
        low_ras = apply_affine_3d(self.affine_ijk2ras, low_ijk.T).T
        high_ras = apply_affine_3d(self.affine_ijk2ras, high_ijk.T).T
        # See which is lower and which is upper
        vs = numpy.vstack((low_ras, high_ras))
        lower_new = numpy.min(vs, axis=0)
        upper_new = numpy.max(vs, axis=0)
        return BoundingBoxDict(lower_new, upper_new)
    
    def _set_affine_ijk2ras(self):
        """Set the affine transformation matrix from IJK to RAS coordinates.
        """
        affine = numpy.eye(4)
        affine[0:3, 0:3] = numpy.diag(self.spacing)
        affine[0:3, 3] = apply_affine_3d(self.affine_ijk2ras_direction, self.origin.T).T # no idea why here it works like this
        affine = compose_affines([affine, self.affine_ijk2ras_direction]) # here order is not important
        return affine
    
    def _set_affine_ras2ijk(self):
        """Set the affine transformation matrix from RAS to IJK coordinates.
        """
        affine = numpy.eye(4)
        affine[0:3, 0:3] = numpy.diag(1/self.spacing)
        affine[0:3, 3] = -self.origin/self.spacing # no idea why now i do not have to transform the origin while before in _set_affine_ijk2ras i had to
        affine = compose_affines([affine, self.affine_ijk2ras_direction]) # here order is not important
        return affine

    def get_data_coordinates_grid(self) -> numpy.ndarray:
        """Returns a numpy.ndarray of the same shape of data+1, where
        the last dimension is either 0, 1 or 2 and corresponds to the
        physical x, y, and z of the voxel, and 3 to the value in the image data.

        Returns
        -------
        numpy.ndarray
            The coordinates grid in a (shape_x, shape_y, shape_z, 4) array. 
        """
        # i,j,k vector to be transformed int RAS
        num_voxels = numpy.prod(self.shape)
        ijk_vector = numpy.zeros(
            shape=(3, num_voxels),
        )
        c_ = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    ijk_vector[:, c_] = numpy.array([i, j, k])
                    c_ += 1
        # Transform to RAS
        ras_coords = numpy.zeros(
            shape=(self.shape[0], self.shape[1], self.shape[2], 4),
            dtype=numpy.float32
        )
        ras_vector = apply_affine_3d(self.affine_ijk2ras, ijk_vector)
        c_ = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    ras_coords[i, j, k, :3] = ras_vector[:, c_]
                    ras_coords[i, j, k,  3] = float(self.data[i, j, k])
                    c_ += 1
        return ras_coords


    def _sample_legacy(self, location: numpy.ndarray, interpolation: str = "nearest") -> numpy.ndarray:
        """ OLD FUNCTION - I'm keeping it just because it is a good example of how to do it without scipy.
        
        Sample the image data at specific 3D location(s) expressed in the RAS space.

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
        location = apply_affine_3d(self.affine_ras2ijk, location)
        # Allocate output memory - locations outside the image will be set to the minimum of the image
        output = numpy.zeros(location.shape[1]) + numpy.min(self.data)
        # check wgich input locations are inside the bbox
        input_inside = self._conservative_b_box.contains(location_ras)
        # Sample
        if interpolation == "nearest":
            location = numpy.round(location).astype("int")
            # we have to cycle to keep in consideration the data that are outside 
            # of the image bounding box
            # - save the indices of the locations inside the image
            inside_idxs = []
            for i in range(location.shape[1]):
                if not input_inside[i]:
                    continue
                if location[0,i] < 0 or location[0,i] >= self.shape[0]-1:
                    continue
                if location[1,i] < 0 or location[1,i] >= self.shape[1]-1:
                    continue
                if location[2,i] < 0 or location[2,i] >= self.shape[2]-1:
                    continue
                inside_idxs.append(i)
            # - sample
            if len(inside_idxs) > 0:
                output[inside_idxs] = self.data[location[0,inside_idxs], location[1,inside_idxs], location[2,inside_idxs]]
        elif interpolation == "linear":
            # trilinear interpolation
            # https://en.wikipedia.org/wiki/Trilinear_interpolation
            inside_idxs = []
            for i in range(location.shape[1]):
                if not input_inside[i]:
                    continue
                l = location[:,i]
                lf = numpy.floor(l).astype("int")
                lc = numpy.ceil(l).astype("int")
                if lf[0] < 0 or lc[0] > self.shape[0]-1:
                    continue
                if lf[1] < 0 or lc[1] > self.shape[1]-1:
                    continue
                if lf[2] < 0 or lc[2] > self.shape[2]-1:
                    continue
                inside_idxs.append(i)
                # POINT BY POINT PROCEDURE
                # This is what is happening. It is slow. Using the whole vectors is much faster.
                # x0, x1 = int(numpy.floor(location[0,i])), int(numpy.ceil(location[0,i]))
                # y0, y1 = int(numpy.floor(location[1,i])), int(numpy.ceil(location[1,i]))
                # z0, z1 = int(numpy.floor(location[2,i])), int(numpy.ceil(location[2,i]))
                # xd = (location[0,i] - x0) / (x1 - x0)
                # yd = (location[1,i] - y0) / (y1 - y0)
                # zd = (location[2,i] - z0) / (z1 - z0)
                # c00 = self.data[x0,y0,z0] * (1 - xd) + self.data[x1,y0,z0] * xd
                # c01 = self.data[x0,y0,z1] * (1 - xd) + self.data[x1,y0,z1] * xd
                # c10 = self.data[x0,y1,z0] * (1 - xd) + self.data[x1,y1,z0] * xd
                # c11 = self.data[x0,y1,z1] * (1 - xd) + self.data[x1,y1,z1] * xd
                # c0 = c00 * (1 - yd) + c10 * yd
                # c1 = c01 * (1 - yd) + c11 * yd
                # output[i] = c0 * (1 - zd) + c1 * zd
            if len(inside_idxs) > 0:
                # - clean
                x0 = numpy.floor(location[:,inside_idxs]).astype("int")
                x1 = numpy.ceil(location[:,inside_idxs]).astype("int")
                # -- in the following line, the complete mathematical formula should be:
                #    xd = (location[:,inside_idxs]-x0)/(x1-x0)
                #    but since x1-x0 = 1, we can simplify.
                #    Also, if x1-x0 == 0, we have a division by zero, so a bug.
                #    In htis way, we cannot have that kind of bug
                xd = (location[:,inside_idxs]-x0)
                # - sample
                c00 = self.data[x0[0,:],x0[1,:],x0[2,:]] * (1 - xd[0,:]) + self.data[x1[0,:],x0[1,:],x0[2,:]] * xd[0,:]
                c01 = self.data[x0[0,:],x0[1,:],x1[2,:]] * (1 - xd[0,:]) + self.data[x1[0,:],x0[1,:],x1[2,:]] * xd[0,:]
                c10 = self.data[x0[0,:],x1[1,:],x0[2,:]] * (1 - xd[0,:]) + self.data[x1[0,:],x1[1,:],x0[2,:]] * xd[0,:]
                c11 = self.data[x0[0,:],x1[1,:],x1[2,:]] * (1 - xd[0,:]) + self.data[x1[0,:],x1[1,:],x1[2,:]] * xd[0,:]
                c0 = c00 * (1 - xd[1,:]) + c10 * xd[1,:]
                c1 = c01 * (1 - xd[1,:]) + c11 * xd[1,:]
                output[inside_idxs] = c0 * (1 - xd[2,:]) + c1 * xd[2,:]
        # Clean output fron nans
        output[numpy.isnan(output)] = numpy.min(self.data)
        # Out
        if output.shape[0] == 1:
            return output[0]
        return output
    
    def sample(self, location: numpy.ndarray, interpolation: str = "nearest") -> numpy.ndarray:
        """Sample the image data at specific 3D location(s) expressed in the RAS space.

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
            The interpolation method to use, the order of the splice interpolation.
            Default is "nearest".
            Available methods are:
            - "nearest" : Nearest neighbor interpolation. (degree 0, fast)
            - "linear" : Linear interpolation. (degree 1, fast) : recommended
            - 2 : Degree 2 spline. (degree 2, slow)
            - 3 : Degree 3 spline. (degree 3, slow)
            - 4 : Degree 4 spline. (degree 4, slow)
            - 5 : Degree 5 spline. (degree 5, slow)
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
        location = apply_affine_3d(self.affine_ras2ijk, location)
        if interpolation == "nearest":
            order = 0
        elif interpolation == "linear":
            order = 1
        else:
            if isinstance(interpolation, int):
                order = interpolation
        output = scipy.ndimage.map_coordinates(
            self.data,
            location,
            order=order,
            mode="nearest",
            cval=numpy.min(self.data),
            prefilter=True
        )
        # Out
        if output.shape[0] == 1:
            return output[0]
        return output


    def __repr__(self):
        return f"ImageCT({self.name}, data of shape {self.data.shape}, origin (mm) {self.origin}, spacing (mm) {self.spacing}, bounding box {self.bounding_box})"
    
    def __str__(self):
        return f"ImageCT({self.name}, data of shape {self.data.shape}, origin (mm) {self.origin}, spacing (mm) {self.spacing}, bounding box {self.bounding_box})"
    


class ImageSequenceCT(object):
    pass




if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    # Test image sampling
    FOLDER = "E:\\MatteoLeccardi\\HearticData\\ASOCA\\"
    from ..asoca import DATASET_ASOCA_IMAGES_DICT
    from ..asoca import AsocaImageCT

    # interpolation test
    f = os.path.join(
        FOLDER,
        DATASET_ASOCA_IMAGES_DICT["Normal"][1]
    )
    image = AsocaImageCT(f)
    xl, xh = image.bounding_box.get_xlim()
    yl, yh = image.bounding_box.get_ylim()
    zl, zh = image.bounding_box.get_zlim()
    x = numpy.linspace(xl, xh, 600)
    y = numpy.linspace(yl, yh, 600)
    z = numpy.mean([zl, zh])
    # make a square grid
    x, y = numpy.meshgrid(x, y)
    points = numpy.vstack((x.flatten(), y.flatten(), z*numpy.ones(x.size))).T
    figure = plt.figure()
    # old interpolation
    ax1 = figure.add_subplot(121, projection="3d")
    ax1.add_artist(image.bounding_box.get_artist())
    t0 = time.time()
    c = image._sample_legacy(points.T, interpolation="linear")
    t1 = time.time()
    ax1.scatter(points[:,0], points[:,1], points[:,2], s=1, c=c, cmap="gray")
    ax1.set(
        title=f"Old interpolation ({t1-t0:.3f}s)", 
        xlim=(xl, xh), ylim=(yl, yh), zlim=(zl, zh))

    ax2 = figure.add_subplot(122, projection="3d")
    ax2.add_artist(image.bounding_box.get_artist())
    t0 = time.time()
    c = image.sample(points.T)
    t1 = time.time()
    ax2.scatter(points[:,0], points[:,1], points[:,2], s=1, c=c, cmap="gray")
    ax2.set(
        title=f"New interpolation ({t1-t0:.3f}s)",
        xlim=(xl, xh), ylim=(yl, yh), zlim=(zl, zh)
    )
    

    plt.show()
    quit()

    # iterative test
    for t_ in ["Normal", "Diseased"]:
        for n_im_ in range(1, 21):
            print(t_, n_im_, "\n")
            f = os.path.join(
                FOLDER,
                DATASET_ASOCA_IMAGES_DICT[t_][n_im_]
            )
            image = AsocaImageCT(f)
            

            # Test sampling
            xl, xh = image.bounding_box.get_xlim()
            yl, yh = image.bounding_box.get_ylim()
            zl, zh = image.bounding_box.get_zlim()

            points = 2*numpy.random.rand(5000, 3)-1.0 + numpy.array([xl, yl, zh])

            """ """
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.add_artist(image.bounding_box.get_artist())
            ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
            ax.set_xlim(points.min(axis=0)[0], points.max(axis=0)[0])
            ax.set_ylim(points.min(axis=0)[1], points.max(axis=0)[1])
            ax.set_zlim(points.min(axis=0)[2], points.max(axis=0)[2])
            plt.show()

            image.data = numpy.random.rand(*image.data.shape)
            image.data[10,10,10] = -50.0
            samples = image.sample(points.T, interpolation="linear")
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.add_artist(image.bounding_box.get_artist())
            ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c=samples)
            ax.set_xlim(points.min(axis=0)[0], points.max(axis=0)[0])
            ax.set_ylim(points.min(axis=0)[1], points.max(axis=0)[1])
            ax.set_zlim(points.min(axis=0)[2], points.max(axis=0)[2])
            plt.show()
            """ """

            ### intensive test
            i = 0
            #plt.ion()
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.add_artist(image.bounding_box.get_artist())
            scat = ax.scatter([0], [0], [0], s=1)
            ax.set_xlim(xl-10, xh+10)
            ax.set_ylim(yl-10, yh+10)
            ax.set_zlim(zl-10, zh+10)
            fig.canvas.draw_idle()
            visited_x, visited_y, visited_z = [], [], []
            while i < 5000:
                x, y, z = numpy.random.rand(3)
                x = x * (xh-xl)+10 + xl - 5
                x = numpy.random.choice([x, x, x, x, xl, xh])
                y = y * (yh-yl)+10 + yl - 5
                y = numpy.random.choice([y, y, y, y, yl, yh])
                z = z * (zh-zl)+10 + zl - 5
                z = numpy.random.choice([z, z, z, z, zl, zh])
                points = 4*numpy.random.rand(1000, 3) -2.0 + numpy.array([x, y, z])
                samples = image.sample(points.T, interpolation="linear")
                visited_x.append(x)
                visited_y.append(y)
                visited_z.append(z)
                if len(visited_x) > 3:
                    p_ = numpy.vstack((visited_x, visited_y, visited_z)).T
                    scat._offsets3d = (visited_x, visited_y, visited_z)
                    fig.canvas.draw_idle()
                    plt.pause(0.05)
                """ """
                print(f"{i:11d} : {x:4.6f}, {y:4.6f}, {z:4.6f}", end="\r")
                i += 1
            plt.close(fig)
              
