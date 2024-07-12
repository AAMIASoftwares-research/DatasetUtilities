import os
import numpy
from scipy.ndimage import binary_erosion, gaussian_filter
import matplotlib.pyplot as plt
import multiprocessing
import nibabel

from ..image import ImagecasImageCT, ImagecasLabelCT
from ..dataset import DATASET_IMAGECAS_IMAGES, DATASET_IMAGECAS_LABELS

BASE_FOLDER = "E:\\MatteoLeccardi\\HearticData\\ImageCAS\\"
DATA_FOLDER = "Data\\"
TARGET_FOLDER = "LabelsAugmentedLumenWallAdaptive\\"

NULL_LABEL = 0
WALL_LABEL = 1
LUMEN_LABEL = 2


from .functions_for_multiprocessing import get_histogram
if __name__ == "__main__" and 0:
    # load all images, save intensities of pixels that are part of the label
    # segmentations into a vector, at the end show the histogram
    # do it with multiprocessing
    N_CPU = multiprocessing.cpu_count()
    with multiprocessing.Pool(N_CPU-1) as pool:
        results = pool.starmap(
            get_histogram, 
            [
                (BASE_FOLDER + DATA_FOLDER + im_, BASE_FOLDER + DATA_FOLDER + la_)
                for i, (im_, la_) in enumerate(
                    zip(DATASET_IMAGECAS_IMAGES, DATASET_IMAGECAS_LABELS)
                )
                #if i < 300
            ]
        )
    results = numpy.array(results)
    results = numpy.sum(results, axis=0)
    pixel_intensities_histogram = results / numpy.sum(results)
    plt.plot(range(-1000, 2001), pixel_intensities_histogram, "-", linewidth=0.6)
    plt.show()


# Check wehter the HU intenisties follow some sort of distribution along the z axis of the images
# (which almost always corresponds to the superior-inferior axis of the patient)
if __name__ == "__main__" and 0:
    results_list_dict: dict[int: list] = {} # key: z (0 the highest z in the image where label>0), value: list of intensities
    for i, (im_, la_) in enumerate(
        zip(DATASET_IMAGECAS_IMAGES, DATASET_IMAGECAS_LABELS)
    ):
        print(f"Processing image {i+1}/{len(DATASET_IMAGECAS_IMAGES)}")
        image = ImagecasImageCT(BASE_FOLDER + DATA_FOLDER + im_)
        image.data = gaussian_filter(image.data, sigma=1)
        label = ImagecasLabelCT(BASE_FOLDER + DATA_FOLDER + la_)
        # erode the label just to be sure of taking the center of the lumen
        label.data = binary_erosion(label.data, iterations=2)
        # get the intensities of the pixels that are part of the label
        pos_label = numpy.where(label.data == 1)
        z_max, z_min = numpy.max(pos_label[2]), numpy.min(pos_label[2])
        intensities = image.data[pos_label]
        for z in range(z_min, z_max+1):
            z_norm = z_max - z
            if z_norm not in results_list_dict:
                results_list_dict[z_norm] = []
            results_list_dict[z_norm].extend(intensities[pos_label[2] == z])
    # plot the boxplot depending on the z (on x axis)
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.boxplot([results_list_dict[z] for z in sorted(results_list_dict.keys())])
    ax.set_xticklabels([str(z) for z in sorted(results_list_dict.keys())])
    plt.show()
    #
    print("{")
    for z in sorted(results_list_dict.keys()):
        print(f"    {z}: {numpy.quantile(results_list_dict[z], [0.25, 0.5, 0.75])},")
    print("}")
    #
    quit()

# now create the labels for the lumen and wall
# we start from the wall, as the label we already have is quite large and also includes arterial wall
# then, we find the lumen by thresholding the image in the locations inside the wall label
# threshold:

from .functions_for_multiprocessing import make_wall_lumen_label

LUMEN_THRESH = 170

# test
if __name__ == "__main__" and 0:
    image_path = BASE_FOLDER + DATA_FOLDER + DATASET_IMAGECAS_IMAGES[0]
    label_path = BASE_FOLDER + DATA_FOLDER + DATASET_IMAGECAS_LABELS[0]
    new_label = make_wall_lumen_label(image_path, label_path, "", LUMEN_THRESH, LUMEN_LABEL, WALL_LABEL, NULL_LABEL)
    
    figure = plt.figure()
    ax = figure.add_subplot(111)
    for i in range(20, 180):
        ax.clear()
        ax.imshow(new_label.data[:, :, i], vmin=0, vmax=2, cmap="coolwarm")
        ax.set_xlabel(f"Slice {i}")
        plt.pause(0.05)
    plt.show()
    figure = plt.figure()
    ax = figure.add_subplot(111)
    image = ImagecasImageCT(image_path)
    ax.imshow(image.data[:, :, 100], vmin=-1000, vmax=800, cmap="gray")
    im_wall = numpy.zeros((image.data[:, :, 100].shape[0], image.data[:, :, 100].shape[1], 3), dtype=int)
    im_wall[:,:,1] = (new_label.data[:, :, 100] == WALL_LABEL).astype(int)*255
    ax.imshow(im_wall, alpha=0.1)
    im_lumen = numpy.zeros((image.data[:, :, 100].shape[0], image.data[:, :, 100].shape[1], 3), dtype=int)
    im_lumen[:,:,0] = (new_label.data[:, :, 100] == LUMEN_LABEL).astype(int)*255
    ax.imshow(im_lumen, alpha=0.1)
    plt.show()
    
    # test I/O
    if 0:
        if not os.path.exists(BASE_FOLDER + TARGET_FOLDER):
            os.makedirs(BASE_FOLDER + TARGET_FOLDER)
        new_label_path = BASE_FOLDER + TARGET_FOLDER + DATASET_IMAGECAS_LABELS[0]
        make_wall_lumen_label(image_path, label_path, new_label_path)



# MAIN

if __name__ == "__main__" and 1:
    import time
    t0_ = time.time()
    N_CPU = max(multiprocessing.cpu_count() - 1, 1)
    N_CPU = 4
    from .functions_for_multiprocessing import INTENSITIES_DISTR_DICT
    input_list = [
        (
            BASE_FOLDER + DATA_FOLDER + im_, 
            BASE_FOLDER + DATA_FOLDER + la_, 
            BASE_FOLDER + TARGET_FOLDER + la_, 
            LUMEN_THRESH,
            LUMEN_LABEL,
            WALL_LABEL,
            NULL_LABEL,
            INTENSITIES_DISTR_DICT
        )
        for i, (im_, la_) in enumerate(
            zip(DATASET_IMAGECAS_IMAGES, DATASET_IMAGECAS_LABELS)
        )
    ]
    if not os.path.exists(BASE_FOLDER + TARGET_FOLDER):
        os.makedirs(BASE_FOLDER + TARGET_FOLDER)
    with multiprocessing.Pool(N_CPU-1) as pool:
        pool.starmap(make_wall_lumen_label, input_list)
    te = time.time()
    print(f"Done in {te-t0_} seconds! (approx {(te-t0_)/60} minutes)")
   