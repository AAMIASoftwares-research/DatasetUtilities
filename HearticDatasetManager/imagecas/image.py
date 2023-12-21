"""Utilities to import a cat08 dataset image into the base image class.
"""
import os
import numpy
import nibabel

from HearticDatasetManager.affine import apply_affine_3d, compose_affines
from HearticDatasetManager.image.image import ImageCT


class ImagecasImageCT(ImageCT):
    """Class to load an ASOCA dataset image into the base image class.

    This dataset does not originally have centerlines associated with it.
    """

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
        

    def _clean_image_path(self, path: str) -> str:
        """Clean the image path.
        """
        path = os.path.normpath(path).replace("\\", os.sep).replace("/", os.sep)
        if os.path.isfile(path) and path.endswith(".nii.gz"):
            return path
        else:
            raise ValueError(f"The path is not a recognized .nii.gz file or it cannot be found:\n{path}")
        

    def _get_image_name(self, path: str) -> str:
        im_name = os.path.splitext(os.path.basename(path))[0]
        im_name = os.path.splitext(im_name)[0]
        name = "ImageCAS/" + im_name
        return name

    def _load_image(self, path) -> None:
        """Load the image into the class.
        """
        # Load the image
        image = nibabel.load(path)
        # Get the image array
        image_array: numpy.ndarray = image.get_fdata().astype(numpy.int16)
        # - Values can go down even to -3000. We could leave it like so,
        #   but to give continuity with the other datasets for which the minimum is -1024,
        #   (approx the HU of air), we clip all values below -1024 to -1024.
        #  ###############
        image_array.clip(min=-1024, out=image_array)
        # - Data i,j,k correspond to y or A, x or R, z or S. 
        #   -> Transpose data so that you have data[i, j, k] where i->x or R, j->y or A, k->z or S
        image_array = numpy.transpose(image_array, axes=(1, 0, 2))
        # Load the image header
        image_header = image.header
        # Get the image spacing
        image_spacing = numpy.array(
            [image_header["pixdim"][2], image_header["pixdim"][1], image_header["pixdim"][3]]
            ).astype(numpy.float32).reshape((3,))
        # Get the image origin
        image_origin = numpy.array(
            [image_header["qoffset_y"], image_header["qoffset_x"], image_header["qoffset_z"]]
            ).astype(numpy.float32).reshape((3,))
        # Get the image direction
        affine_ijk2ras_direction = numpy.eye(4)
        s_rows = [image_header["srow_x"], image_header["srow_y"], image_header["srow_z"]]
        for i in range(3):
            affine_ijk2ras_direction[i,i] *= numpy.sign(s_rows[i][i])
        # transform origin according to ras orientation
        image_origin = apply_affine_3d(affine_ijk2ras_direction, image_origin)
        # out
        return (image_array, image_origin, image_spacing, affine_ijk2ras_direction)





        
if __name__ == "__main__":
    # Example usage
    image_path = "C:\\Users\\lecca\\Desktop\\ImageCAS\\Data\\1.img.nii.gz"
    image = ImagecasImageCT(image_path)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.set_facecolor("#010238")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("#010238")
    ax.grid(False)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    ax.set_xlabel("R (x)", color="white")
    ax.set_ylabel("A (y)", color="white")
    ax.set_zlabel("S (z)", color="white")

    # Origin and BB
    ax.scatter(image.origin[0], image.origin[1], image.origin[2], c="r", s=40)
    ax.add_collection(image.bounding_box.get_artist())
    ax.set_xlim(image.bounding_box["lower"][0]-100, image.bounding_box["upper"][0]+100)
    ax.set_ylim(image.bounding_box["lower"][1]-100, image.bounding_box["upper"][1]+100)
    ax.set_zlim(image.bounding_box["lower"][2]-100, image.bounding_box["upper"][2]+100)
    
    # Slice
    z_ras = image.origin[2]+40
    xs = numpy.linspace(image.bounding_box["lower"][0]-2, image.bounding_box["upper"][0]+2, 200)
    ys = numpy.linspace(image.bounding_box["lower"][1]-2, image.bounding_box["upper"][1]+2, 200)
    points_to_sample = []
    for x in xs:
        for y in ys:
            points_to_sample.append([x, y, z_ras])
    points_to_sample = numpy.array(points_to_sample)
    samples = image.sample(points_to_sample.T, interpolation="linear")
    print(points_to_sample.shape, samples.shape)
    print(set(samples.tolist()))
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

    plt.hist(samples, bins=300)
    plt.show()

    
    

if __name__ == "__main__" and 0:
    # check if all datasets are scaled the same
    import matplotlib.pyplot as plt
    base_path = "C:\\Users\\lecca\\Desktop\\ImageCAS\\Data\\"
    ######################

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