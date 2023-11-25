import os, sys
import numpy
import matplotlib.pyplot as plt

import hcatnetwork

from .affine import apply_affine_3d
from .image import ImageCT
from .cat08 import Cat08ImageCT, DATASET_CAT08_IMAGES_TRAINING, DATASET_CAT08_GRAPHS_RESAMPLED_05MM
from .asoca import AsocaImageCT, DATASET_ASOCA_IMAGES_DICT, DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT


DATASETS_FOLDER = "C:\\Users\\lecca\\OneDrive - Politecnico di Milano\\zzzz_datasets_neuroengineering_2023".replace("\\", "/")


def get_image_on_xy_plane(image: AsocaImageCT|Cat08ImageCT, center_ras: numpy.ndarray, samples_per_side: int, mm_per_side: float):
    xs = numpy.linspace(center_ras[0] - mm_per_side/2, center_ras[0] + mm_per_side/2, samples_per_side)
    ys = numpy.linspace(center_ras[1] - mm_per_side/2, center_ras[1] + mm_per_side/2, samples_per_side)
    z = center_ras[2]
    points_to_sample = []
    for x in xs:
        for y in ys:
            points_to_sample.append([x, y, z])
    points_to_sample = numpy.array(points_to_sample).squeeze(axis=-1)
    samples = image.sample(points_to_sample.T, "linear")
    # plot
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.set_title(image.name)
    ax.add_artist(image.bounding_box.get_artist())
    ax.scatter(points_to_sample[:,0], points_to_sample[:,1], points_to_sample[:,2], c=samples, marker='.', s=5, cmap='gray')
    ax.scatter(center_ras[0], center_ras[1], center_ras[2]+2, c='r', marker='o', s=20)
    plt.show()

if __name__ == "__main__":
    
    im_file_list = [os.path.join(DATASETS_FOLDER,"CAT08",DATASET_CAT08_IMAGES_TRAINING[i]) for i in range(7)]
    im_file_list.extend(
        [os.path.join(DATASETS_FOLDER,"ASOCA",DATASET_ASOCA_IMAGES_DICT["Normal"][i]) for i in range(20)]
    )
    im_file_list.extend(
        [os.path.join(DATASETS_FOLDER,"ASOCA",DATASET_ASOCA_IMAGES_DICT["Diseased"][i]) for i in range(20)]
    )

    g_file_list = [os.path.join(DATASETS_FOLDER,"CAT08",DATASET_CAT08_GRAPHS_RESAMPLED_05MM[i]) for i in range(7)]
    g_file_list.extend(
        [os.path.join(DATASETS_FOLDER,"ASOCA",DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT["Normal"][i]) for i in range(20)]
    )
    g_file_list.extend(
        [os.path.join(DATASETS_FOLDER,"ASOCA",DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT["Diseased"][i]) for i in range(20)]
    )

    image_class_list = [Cat08ImageCT]*7 + [AsocaImageCT]*40
    
    for imf, gf, im_class in zip(im_file_list, g_file_list, image_class_list):
        im = im_class(imf)
        g = hcatnetwork.io.load_graph(gf, output_type=hcatnetwork.graph.SimpleCenterlineGraph)
        l_id, r_id = g.get_coronary_ostia_node_id()
        node = g.nodes[l_id]
        center = numpy.array([node['x'], node['y'], node['z']])
        center_ras = apply_affine_3d(im.affine_centerlines2ras, center)
        get_image_on_xy_plane(im, center_ras, 200, 60)
    
    
    
