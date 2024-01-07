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


_IMAGECAS_DATA_FOLDER = "C:\\Users\\lecca\\Desktop\\ImageCAS\\Data\\"

_RESPACING_MM = 0.5

_SAVE_FOLDER = f"C:\\Users\\lecca\\Desktop\\ImageCAS_segmentor_{_RESPACING_MM:.2f}mm\\Data\\"



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
    # - transform these voxels into mm in RAS coord system
    x_mm_lims = [apply_affine_3d(image.affine_ijk2ras, numpy.array([x_, 0, 0]).T)[0,0] for x_ in x_voxels_lims]
    x_mm_lims.sort()
    y_mm_lims = [apply_affine_3d(image.affine_ijk2ras, numpy.array([0, y_, 0]).T)[1,0] for y_ in y_voxels_lims]
    y_mm_lims.sort()
    z_mm_lims = [apply_affine_3d(image.affine_ijk2ras, numpy.array([0, 0, z_]).T)[2,0] for z_ in z_voxels_lims]
    z_mm_lims.sort()
    # - decrease the minimums by 4*new spacing
    x_mm_lims[0] -= 4*_RESPACING_MM
    y_mm_lims[0] -= 4*_RESPACING_MM
    z_mm_lims[0] -= 4*_RESPACING_MM
    # - increase the maximums by 4*new spacing
    x_mm_lims[1] += 4*_RESPACING_MM
    y_mm_lims[1] += 4*_RESPACING_MM
    z_mm_lims[1] += 4*_RESPACING_MM
    # - get the number of samples along the three axis
    x_num_samples = int((x_mm_lims[1] - x_mm_lims[0]) / _RESPACING_MM)
    y_num_samples = int((y_mm_lims[1] - y_mm_lims[0]) / _RESPACING_MM)
    z_num_samples = int((z_mm_lims[1] - z_mm_lims[0]) / _RESPACING_MM)
    # - samples along axis
    x_samples = numpy.linspace(x_mm_lims[0], x_mm_lims[0]+x_num_samples*_RESPACING_MM, x_num_samples)
    y_samples = numpy.linspace(y_mm_lims[0], y_mm_lims[0]+y_num_samples*_RESPACING_MM, y_num_samples)
    z_samples = numpy.linspace(z_mm_lims[0], z_mm_lims[0]+z_num_samples*_RESPACING_MM, z_num_samples)
    # - create sampling grid
    sampling_grid = []
    for x_ in x_samples:
        for y_ in y_samples:
            for z_ in z_samples:
                sampling_grid.append([x_, y_, z_])
    sampling_grid = numpy.array(sampling_grid)
    # - create cropped and resampled image
    cropped_image_samples = image.sample(sampling_grid.T, interpolation="linear").T
    cropped_image_samples = cropped_image_samples.reshape(x_num_samples, y_num_samples, z_num_samples)
    cropped_image_samples = cropped_image_samples.astype("int16")
    # - create cropped and resampled mask
    cropped_mask_samples = mask.sample(sampling_grid.T, interpolation="nearest").T
    cropped_mask_samples = cropped_mask_samples.reshape(x_num_samples, y_num_samples, z_num_samples)
    cropped_mask_samples = cropped_mask_samples.astype("bool").astype("uint8")
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
    nifti_spacing = numpy.array([1., _RESPACING_MM, _RESPACING_MM, _RESPACING_MM, 1., 1., 1., 1.])
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
    for i, file in enumerate(DATASET_IMAGECAS_IMAGES[:3]):
        num = int(file.split(".")[0])
        print(f"Working on {num} ({100*i/len(DATASET_IMAGECAS_IMAGES):.2f}%)")
        preprocess_image_couple(num)
    print("Done! Process took ", time.time()-t0, " seconds (~ ", int((time.time()-t0)/60), " minutes)")
