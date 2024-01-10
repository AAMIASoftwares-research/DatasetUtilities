# This module is composed of just this file.
# Should have a one-time use case only.
# This is to create the dataset for the segmentation task
# performed by SegMENTOR, created by Luca Marsilio and Pietro Cerveri.

import os, time
import numpy
import nibabel
import matplotlib.pyplot as plt

from ...affine import apply_affine_3d
from ..image import ImagecasImageCT, ImagecasLabelCT
from ..dataset import DATASET_IMAGECAS_IMAGES, DATASET_IMAGECAS_LABELS


_IMAGECAS_DATA_FOLDER = "E:\\MatteoLeccardi\\HearticData\\ImageCAS\\Data\\"

_SAVE_FOLDER = f"E:\\MatteoLeccardi\\HearticData\\ImageCAS_segmentor\\Data\\"



def preprocess_image_couple(number: int):
    # Load the image and the mask
    #############################
    image_path = os.path.join(
        _IMAGECAS_DATA_FOLDER,
        f"{int(number)}.img.nii.gz"
    )
    image = ImagecasImageCT(image_path)
    mask_path = os.path.join(
        _IMAGECAS_DATA_FOLDER,
        f"{int(number)}.label.nii.gz"
    )
    mask = ImagecasLabelCT(mask_path)
    # - check shapes
    if image.data.shape != mask.data.shape:
        raise ValueError(f"Image ({number}) and mask have different shapes!", image.data.shape, mask.data.shape)
    # Preprocess
    #############################
    # - get voxels to create a bounding box
    mask_indices = numpy.argwhere(mask.data == 1)
    x_voxels_lims = [numpy.min(mask_indices[:, 0]), numpy.max(mask_indices[:, 0])]
    y_voxels_lims = [numpy.min(mask_indices[:, 1]), numpy.max(mask_indices[:, 1])]
    z_voxels_lims = [numpy.min(mask_indices[:, 2]), numpy.max(mask_indices[:, 2])]
    # - - expand bounding box by 5 voxels
    x_voxels_lims[0] = max(x_voxels_lims[0]-5, 0)
    x_voxels_lims[1] = min(x_voxels_lims[1]+5, image.data.shape[0]-1)
    y_voxels_lims[0] = max(y_voxels_lims[0]-5, 0)
    y_voxels_lims[1] = min(y_voxels_lims[1]+5, image.data.shape[1]-1)
    z_voxels_lims[0] = max(z_voxels_lims[0]-5, 0)
    z_voxels_lims[1] = min(z_voxels_lims[1]+5, image.data.shape[2]-1)
    # - create cropped image
    cropped_image_samples = image.data[
        x_voxels_lims[0]:x_voxels_lims[1]+1,
        y_voxels_lims[0]:y_voxels_lims[1]+1,
        z_voxels_lims[0]:z_voxels_lims[1]+1
    ]
    cropped_image_samples = cropped_image_samples.astype("int16")
    # - create cropped and resampled mask
    cropped_mask_samples = mask.data[
        x_voxels_lims[0]:x_voxels_lims[1]+1,
        y_voxels_lims[0]:y_voxels_lims[1]+1,
        z_voxels_lims[0]:z_voxels_lims[1]+1
    ]
    cropped_mask_samples = cropped_mask_samples.astype("bool").astype("uint8")
    # Set limits in ras coordinates
    ###############################
    # - transform these voxels into mm in RAS coord system
    x_mm_lims = [apply_affine_3d(image.affine_ijk2ras, numpy.array([x_, 0, 0]).T)[0,0] for x_ in x_voxels_lims]
    x_mm_lims.sort()
    y_mm_lims = [apply_affine_3d(image.affine_ijk2ras, numpy.array([0, y_, 0]).T)[1,0] for y_ in y_voxels_lims]
    y_mm_lims.sort()
    z_mm_lims = [apply_affine_3d(image.affine_ijk2ras, numpy.array([0, 0, z_]).T)[2,0] for z_ in z_voxels_lims]
    z_mm_lims.sort()
    # Save
    #############################
    # - create folder
    save_folder = _SAVE_FOLDER
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # - save image as nifti
    nifti_affine = numpy.eye(4)
    nifti_affine[0, 3] = x_mm_lims[0]
    nifti_affine[1, 3] = y_mm_lims[0]
    nifti_affine[2, 3] = z_mm_lims[0]
    nifti_spacing = numpy.array([1., image.spacing[0], image.spacing[1], image.spacing[2], 1., 1., 1., 1.])
    nifti_img = nibabel.Nifti1Image(
        cropped_image_samples,
        affine=nifti_affine
    )
    nifti_img.header["pixdim"] = nifti_spacing
    nifti_img.header["scl_slope"] = 1.0
    nifti_img.header["scl_inter"] = 0.0
    nifti_img.header["qoffset_x"] = x_mm_lims[0]
    nifti_img.header["qoffset_y"] = y_mm_lims[0]
    nifti_img.header["qoffset_z"] = z_mm_lims[0]
    nibabel.save(
        nifti_img,
        os.path.join(_SAVE_FOLDER, f"{number}.img.nii.gz")
    )
    # - save mask as nifti
    nifti_mask = nibabel.Nifti1Image(
        cropped_mask_samples,
        affine=nifti_affine
    )
    nifti_mask.header["pixdim"] = nifti_spacing
    nifti_mask.header["scl_slope"] = 1.0
    nifti_mask.header["scl_inter"] = 0.0
    nifti_mask.header["qoffset_x"] = x_mm_lims[0]
    nifti_mask.header["qoffset_y"] = y_mm_lims[0]
    nifti_mask.header["qoffset_z"] = z_mm_lims[0]
    nibabel.save(
        nifti_mask,
        os.path.join(_SAVE_FOLDER, f"{number}.label.nii.gz")
    )

if __name__ == "__main__":
    t0 = time.time()
    for i, file in enumerate(DATASET_IMAGECAS_IMAGES):
        num = int(file.split(".")[0])
        print(f"Working on {num} ({100*i/len(DATASET_IMAGECAS_IMAGES):.2f}%)")
        preprocess_image_couple(num)
    print("Done! Process took ", time.time()-t0, " seconds (~ ", int((time.time()-t0)/60), " minutes)")
