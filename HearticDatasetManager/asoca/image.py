"""Utilities to import a cat08 dataset image into the base image class.
"""
import os
import numpy
import SimpleITK
import nrrd

from HearticDatasetManager.affine import apply_affine_3d, compose_affines
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
        # Just for 3D Slicer visualization
        self.affine_centerlines2ras_slicer = self._get_affine_centerlines2ras_slicer()

    def _clean_image_path(self, path: str) -> str:
        """Clean the image path.
        """
        path = os.path.normpath(path).replace("\\", os.sep).replace("/", os.sep)
        if os.path.isfile(path) and path.endswith(".nrrd"):
            return path
        else:
            raise ValueError(f"The path is not a recognized .nrrd file or it cannot be found:\n{path}")
        

    def _get_image_name(self, path: str) -> str:
        im_name = os.path.splitext(os.path.basename(path))[0]
        name = "ASOCA/" + im_name
        return name

    def _load_image(self, path) -> None:
        """Load the image into the class.
        """
        try:
            # Load the image - first try with SimpleITK
            image = SimpleITK.ReadImage(
                path,
                SimpleITK.sitkInt16    
            )
            # Get the image array
            image_array = SimpleITK.GetArrayFromImage(image)
            # - While the Hounsfield units are fine for ASOCA images,
            #   values can go down even to -3000. We could leave it like so,
            #   but to give continuity with the other datasets for which the minimum is -1024,
            #   (approx the HU of air), we clip all values below -1024 to -1024.
            image_array[image_array < -1024] = -1024
            # - now, data i,j,k correspond to z or S, y or A, x or R
            # - transpose data so that you have data[i, j, k] where i->x or R, j->y or A, k->z or S
            image_array = numpy.transpose(image_array, axes=(2, 1, 0))
            # Get the image spacing
            image_spacing = numpy.array(image.GetSpacing()).astype(numpy.float32).reshape((3,))
            # Get the image origin
            image_origin = numpy.array(image.GetOrigin()).astype(numpy.float32).reshape((3,))
            # Get the image direction
            # in ASOCA, the image direction provided by SimpleITK is always the identity matrix
            # so it is basically useless. Keep the code for later reference.
            # >> image_direction = numpy.array(image.GetDirection())
            # In Slicer, to view the image in RAS coordinates, the following direction transform
            # is applied to the image:
            affine_ijk2ras_direction = numpy.eye(4)
            affine_ijk2ras_direction[0,0] = -1.0
            affine_ijk2ras_direction[1,1] = -1.0
            # transform origin according to ras orientation
            image_origin = apply_affine_3d(affine_ijk2ras_direction, image_origin)
        except Exception as e:
            # RECENTLY (JULY 2025) SIMPLE-ITK STARTED FAILING TO OPEN NRRD IMAGES -> FALL TO PYNRRD 
            #
            # Use pynrrd as fallback
            image_array, header = nrrd.read(path)
            # NRRD in this dataset is y, x, z; We do not transpose this, as all matrices works fine without transposition
            # they break if the image is transposed.
            # image_array = numpy.transpose(image_array, axes=(1, 0, 2))
            # After transposition, data is actuually i -> L, j -> P, k -> S
            # If we wanted the image to be with ijk compatible with ras, we could do
            # the following, but then we would have to check that all matrices for conversions 
            # (ijk2ras, etc...) work fine
            #image_array = image_array[::-1, ::-1,:]
            #
            # Get spacing from header
            if 'space directions' in header:
                spacing = []
                for v in header['space directions']:
                    spacing.append(numpy.linalg.norm(v))
                image_spacing = numpy.array(spacing, dtype=numpy.float32)
            elif 'spacings' in header:
                image_spacing = numpy.array(header['spacings'], dtype=numpy.float32)
            else:
                image_spacing = numpy.array([1.0, 1.0, 1.0], dtype=numpy.float32)
            # Get origin from header
            if 'space origin' in header:
                image_origin = numpy.array(header['space origin'], dtype=numpy.float32)
            else:
                image_origin = numpy.zeros(3, dtype=numpy.float32)
            # Get the image direction
            # in ASOCA, the image direction provided by SimpleITK is always the identity matrix
            # so it is basically useless. Keep the code for later reference.
            # >> image_direction = numpy.array(image.GetDirection())
            # In Slicer, to view the image in RAS coordinates, the following direction transform
            # is applied to the image:
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
            [-1.0,  0.0, 0.0, 2*self.origin[0]],
            [ 0.0, -1.0, 0.0, 2*self.origin[1]],
            [ 0.0,  0.0, 1.0,               0.0],
            [ 0.0,  0.0, 0.0,               1.0]
        ])
        return out_affine
    
    def _get_affine_ras2centerlines(self) -> numpy.ndarray:
        """Builds the affine to transform ASOCA centelrines from ras back to the original space.
        This has been found out empirically and it works fine in 3D Slicer.
        """
        out_affine = numpy.array([
            [-1.0,  0.0, 0.0, -2*self.origin[0]],
            [ 0.0, -1.0, 0.0, -2*self.origin[1]],
            [ 0.0,  0.0, 1.0,              0.0],
            [ 0.0,  0.0, 0.0,              1.0]
        ])
        return out_affine

    def _get_affine_centerlines2ras_slicer(self) -> numpy.ndarray:
        """Builds the affine to transform ASOCA centelrines into the RAS image space of 3D Slicer.
        This has been found out empirically and it works fine in 3D Slicer.
        """
        out_affine = numpy.array([
            [-1.0,  0.0, 0.0, -2*self.origin[0]],
            [ 0.0, -1.0, 0.0, -2*self.origin[1]],
            [ 0.0,  0.0, 1.0,               0.0],
            [ 0.0,  0.0, 0.0,               1.0]
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

    
    

if __name__ == "__main__" and 0:
    # check if all datasets are scaled the same
    import matplotlib.pyplot as plt
    base_path = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA"
    from .dataset import DATASET_ASOCA_IMAGES

    for im_path in DATASET_ASOCA_IMAGES:
        im_path = os.path.join(base_path, im_path)
        if not os.path.exists(im_path):
            continue
        image = AsocaImageCT(im_path)
        if 0:
            plt.hist(image.data.flatten(), bins=300)
            plt.title(os.path.basename(im_path))
            plt.show()
        if 1:
            # Image is shown in the contrary because
            # the ijk mapping is different from the ras mapping
            # >> plt.imshow(image.data[:,:,59], cmap="gray")
            # to view it correctly on video,
            # but wrong with respect to the matplotlib axes
            plt.imshow(image.data[::-1,:,100].T, cmap="gray")
            plt.show()
    quit()