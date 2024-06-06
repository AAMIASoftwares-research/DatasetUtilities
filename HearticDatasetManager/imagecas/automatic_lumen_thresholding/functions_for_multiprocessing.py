import numpy
from ..image import ImagecasImageCT, ImagecasLabelCT
from scipy.ndimage import binary_erosion
import nibabel

def get_histogram(image_file_path, label_file_path) -> list[float]:
        image = ImagecasImageCT(image_file_path)
        label = ImagecasLabelCT(label_file_path)
        pixel_intensities_histogram = [0]*3001 # from -1000 to 2000 included
        where = numpy.argwhere(label.data > 0)
        for x, y, z in where:
            if (image.data[x, y, z] >= -1000) and (image.data[x, y, z] <= 2000):
                pixel_intensities_histogram[image.data[x, y, z]+1000] += 1
        return pixel_intensities_histogram



def make_wall_lumen_label(image_file_path, label_file_path, save_path, lumen_thresh=150, lumen_label=2, wall_label=1, null_label=0):
    image = ImagecasImageCT(image_file_path)
    label = ImagecasLabelCT(label_file_path)
    # create the new label file, that has 0 and 1 as labels for Null and Wall
    new_label = ImagecasLabelCT(label_file_path)
    # outer layer erosion
    new_label.data = binary_erosion(label.data.astype(bool), iterations=1)
    new_label.data = new_label.data.astype(int)
    new_label.data += label.data
    # now, erode the lumen label so that a pixel does not have neighbours of Null label
    # in the slice
    where = numpy.argwhere(new_label.data == lumen_label)
    for x, y, z in where:
        if null_label in new_label.data[x-1:x+2, y-1:y+2, z]:
            new_label.data[x, y, z] = wall_label
    # intensity-based erosion of the wall label
    # only in the locations where the previously found lumen is
    # after finding the pixels we have to search,
    # the lumen label is reset to be all just wall label
    where = numpy.argwhere(new_label.data == lumen_label)
    new_label.data = new_label.data.astype(bool).astype(int)
    for x, y, z in where:
        if image.data[x, y, z] >= lumen_thresh:
            new_label.data[x, y, z] = lumen_label
    # if a wessel wall pixel is surrounded, on the slice, 
    # by lumen pixels, it is a lumen pixel
    # this is to prevent holes in the lumen label
    where = numpy.argwhere(new_label.data == wall_label)
    for x, y, z in where:
        square_ = new_label.data[x-1:x+2, y-1:y+2, z].copy()
        square_[1, 1] = 100
        if wall_label not in square_:
            new_label.data[x, y, z] = lumen_label
    # return or save with nibabel to nii.gz
    if save_path == "":
        return new_label
    nib_label = nibabel.load(label_file_path)
    nib_label_new = nibabel.Nifti1Image(
        numpy.flip(new_label.data, axis=0).astype(numpy.uint8),
        nib_label.affine
    )
    nibabel.save(nib_label_new, save_path)