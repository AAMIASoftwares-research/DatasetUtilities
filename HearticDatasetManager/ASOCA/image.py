"""Utilities to import a cat08 dataset image into the base image class.
"""
import os
import numpy
import SimpleITK

from HearticDatasetManager.affine import apply_affine_3d, get_affine_3d_translation, get_affine_3d_scale
from HearticDatasetManager.image.image import ImageCT


class AsocaImageCT(ImageCT):
    """Class to load an ASOCA dataset image into the base image class.

    This class contains also the affine to transform ASOCA centelrines
    into the RAS image space.
    """
    affine_centerlines2ras: numpy.ndarray
    affine_ras2centerlines: numpy.ndarray

    def __init__(self, path: str):
        """Initialize the class.

        Parameters
        ----------
            path (str): Path to the .nrrd image file (.nrrd has to be specified).
        """
        # path
        path = self._clean_image_path(path) 
        # name
        name = self._get_image_name(path)
        # ImageCT data
        data, origin, spacing, affine_ijk2ras_direction = self._load_image(path)
        super().__init__(name, data, origin, spacing, affine_ijk2ras_direction)
        # Centerline data coordinates affine
        self.affine_centerlines2ras = self._get_affine_centerlines2ras()
        self.affine_ras2centerlines = self._get_affine_ras2centerlines()

    def _clean_image_path(self, path: str) -> str:
        """Clean the image path.
        """
        path = os.path.normpath(path)
        if os.path.isfile(path) and path.endswith(".nrrd"):
            return path
        else:
            raise ValueError("The path is not a recognized .nrrd file.")
        

    def _get_image_name(self, path: str) -> str:
        im_name = os.path.splitext(os.path.basename(path))[0]
        name = "ASOCA/" + im_name
        return name

    def _load_image(self, path) -> None:
        """Load the image into the class.
        """
        # Load the image
        image = SimpleITK.ReadImage(path)
        # Get the image array
        image_array = SimpleITK.GetArrayFromImage(image).astype(numpy.int16)
        # - now, data i,j,k correspond to z or S, y or A, x or R
        # - transpose data so that you have data[i, j, k] where i->x or R, j->y or A, k->z or S
        image_array = numpy.transpose(image_array, axes=(2, 1, 0))
        # Get the image spacing
        image_spacing = numpy.array(image.GetSpacing()).astype(numpy.float32).reshape((3,))
        # Get the image origin
        image_origin = numpy.array(image.GetOrigin()).astype(numpy.float32).reshape((3,))
        # Get the image direction
        # in ASOCA, the SimpleITK image direction is always the identity matrix
        # so it is basically useless. Keep the code for later reference.
        # >> image_direction = numpy.array(image.GetDirection())
        affine_ijk2ras_direction = numpy.eye(4)
        affine_ijk2ras_direction[0,0] = -1.0
        affine_ijk2ras_direction[1,1] = -1.0
        # transform origin according to ras orientation
        image_origin = apply_affine_3d(affine_ijk2ras_direction, image_origin)
        # out
        return (image_array, image_origin, image_spacing, affine_ijk2ras_direction)
    
    def _get_affine_centerlines2ras(self) -> numpy.ndarray:
        """Builds the affine to transform ASOCA centelrines into the RAS image space.
        This has been found out empirically and it works fine in 3D Slicer.
        """
        out_affine = numpy.array([
            [-1.0,  0.0, 0.0, -2*self.origin[0]],
            [ 0.0, -1.0, 0.0, -2*self.origin[1]],
            [ 0.0,  0.0, 1.0,               0.0],
            [ 0.0,  0.0, 0.0,               1.0]
        ])
        return out_affine
    
    def _get_affine_ras2centerlines(self) -> numpy.ndarray:
        """Builds the affine to transform ASOCA centelrines from ras back to the original space.
        This has been found out empirically and it works fine in 3D Slicer.
        """
        out_affine = numpy.array([
            [-1.0,  0.0, 0.0, 2*self.origin[0]],
            [ 0.0, -1.0, 0.0, 2*self.origin[1]],
            [ 0.0,  0.0, 1.0,              0.0],
            [ 0.0,  0.0, 0.0,              1.0]
        ])
        return out_affine

        
if __name__ == "__main__":
    # Example usage
    image_path = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\Diseased\\CTCA\\Diseased_1.nrrd"
    image_path = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\Normal\\CTCA\\Normal_1.nrrd"
    image = AsocaImageCT(image_path)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.set_facecolor("#010238")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("#010238")
    ax.grid(False)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Origin and BB
    ax.scatter(image.origin[0], image.origin[1], image.origin[2], c="r", s=40)
    ax.add_collection(image.bounding_box.get_artist())
    ax.set_xlim(image.bounding_box["lower"][0]-100, image.bounding_box["upper"][0]+100)
    ax.set_ylim(image.bounding_box["lower"][1]-100, image.bounding_box["upper"][1]+100)
    ax.set_zlim(image.bounding_box["lower"][2]-100, image.bounding_box["upper"][2]+100)
    
    # Slice
    z_ras = image.origin[2]+5
    xs = numpy.linspace(image.bounding_box["lower"][0]-2, image.bounding_box["upper"][0]+2, 200)
    ys = numpy.linspace(image.bounding_box["lower"][1]-2, image.bounding_box["upper"][1]+2, 200)
    points_to_sample = []
    for x in xs:
        for y in ys:
            points_to_sample.append([x, y, z_ras])
    points_to_sample = numpy.array(points_to_sample)
    samples = image.sample(points_to_sample.T, interpolation="linear")
    ax.scatter(
        points_to_sample[:,0],
        points_to_sample[:,1],
        points_to_sample[:,2],
        c=samples,
        cmap="gray",
        s=10,
        linewidths=0.0,
        antialiased=False
    )
    
    plt.show()

    # Histogram
    fig = plt.figure()
    plt.hist(image.data.flatten(), bins=500)
    plt.show()
    
    
