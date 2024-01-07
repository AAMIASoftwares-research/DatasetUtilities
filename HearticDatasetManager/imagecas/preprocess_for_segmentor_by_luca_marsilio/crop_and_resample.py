# This module is composed of just this file.
# Should have a one-time use case only.
# This is to create the dataset for the segmentation task
# performed by SegMENTOR, created by Luca Marsilio and Pietro Cerveri.

import os, time
import numpy
import nibabel
import matplotlib.pyplot as plt

from ...affine import apply_affine_3d
from ..image import AsocaImageCT


_ASOCA_DATA_FOLDER = "C:\\Users\\lecca\\OneDrive - Politecnico di Milano\\zzzz_datasets_neuroengineering_2023\\ASOCA\\"

_SUBS = ["Normal", "Diseased"]

_RESPACING_MM = 0.5

_SAVE_FOLDER = f"C:\\Users\\lecca\\OneDrive - Politecnico di Milano\\zzzz_datasets_neuroengineering_2023\\ASOCA_segmentor_{_RESPACING_MM:.2f}mm\\"



def preprocess_image_couple(subset: str, number: int):
    if subset not in _SUBS:
        raise ValueError("Invalid subset name")
    if number < 1 or number > 20:
        raise ValueError("Invalid image number")
    # Load the image and the mask
    #############################
    image_path = os.path.join(
        _ASOCA_DATA_FOLDER,
        subset,
        "CTCA",
        subset+f"_{number}.nrrd"
    )
    image = AsocaImageCT(image_path)
    mask_path = os.path.join(
        _ASOCA_DATA_FOLDER,
        subset,
        "Annotations",
        subset+f"_{number}.nrrd"
    )
    mask = AsocaImageCT(mask_path)
    # - fix rescaling
    _min = numpy.min(mask.data) # step necessary because of how AsocaImageCT works
    mask.data[mask.data > _min] = 1
    mask.data[mask.data == _min] = 0
    # - fix affines
    mask.affine_centerlines2ras = image.affine_centerlines2ras
    mask.affine_ras2centerlines = image.affine_ras2centerlines
    mask.affine_centerlines2ras_slicer = image.affine_centerlines2ras_slicer
    mask.affine_ijk2ras = image.affine_ijk2ras
    mask.affine_ras2ijk = image.affine_ras2ijk
    mask.affine_ijk2ras_direction = image.affine_ijk2ras_direction
    # - check shapes
    if image.data.shape != mask.data.shape:
        raise ValueError("Image and mask have different shapes!", image.data.shape, mask.data.shape)
    # Preprocess
    #############################
    # - get voxels to create a bounding box
    mask_indices = numpy.argwhere(mask.data == 1)
    x_voxels_lims = [numpy.min(mask_indices[:, 0]), numpy.max(mask_indices[:, 0])]
    y_voxels_lims = [numpy.min(mask_indices[:, 1]), numpy.max(mask_indices[:, 1])]
    z_voxels_lims = [numpy.min(mask_indices[:, 2]), numpy.max(mask_indices[:, 2])]
    # - transform these voxels into mm in RAS coord system
    x_mm_lims = [apply_affine_3d(image.affine_ijk2ras, numpy.array([x_, 0, 0]).T)[0] for x_ in x_voxels_lims]
    x_mm_lims.sort()
    y_mm_lims = [apply_affine_3d(image.affine_ijk2ras, numpy.array([0, y_, 0]).T)[1] for y_ in y_voxels_lims]
    y_mm_lims.sort()
    z_mm_lims = [apply_affine_3d(image.affine_ijk2ras, numpy.array([0, 0, z_]).T)[2] for z_ in z_voxels_lims]
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
    sampling_grid = numpy.array(sampling_grid).squeeze(-1)
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
    save_folder = os.path.join(_SAVE_FOLDER, subset)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    ctca_folder = os.path.join(save_folder, "CTCA")
    if not os.path.exists(ctca_folder):
        os.makedirs(ctca_folder)
    annotations_folder = os.path.join(save_folder, "Annotations")
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)
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
        os.path.join(ctca_folder, f"{subset}_{number}.nii.gz")
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
        os.path.join(annotations_folder, f"{subset}_{number}.nii.gz")
    )

t0 = time.time()
for sub in _SUBS:
    for num in range(1, 21):
        print(f"Working on {sub} {num}")
        preprocess_image_couple(sub, num)
print("Done! Process took ", time.time()-t0, " seconds (", (time.time()-t0)/60, " minutes)")
