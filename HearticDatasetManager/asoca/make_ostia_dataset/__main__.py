import os
import numpy
import json
import matplotlib.pyplot as plt

import hcatnetwork

from HearticDatasetManager.affine import apply_affine_3d
from HearticDatasetManager.asoca.image import AsocaImageCT
from HearticDatasetManager.asoca.dataset import DATASET_ASOCA_TRAINING, DATASET_ASOCA_GRAPHS_RESAMPLED_05MM

def asoca_get_ostia_single_patient(path_to_asoca_patient_graph: str, affine_centerlines2ras: numpy.ndarray, affine_ras2ijk: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Output tuple:
    - 0: ostia points in ras space: numpy.ndarray of shape (1, 2, 3)
    - 1: ostia points in ijk space: numpy.ndarray of shape (1, 2, 3)
    - 2: left/right labels: numpy.ndarray of shape (1, 2)
    """
    g = hcatnetwork.io.load_graph(
        path_to_asoca_patient_graph,
        output_type=hcatnetwork.graph.SimpleCenterlineGraph
    )
    left_ostium, right_ostium = None, None
    for n in g.nodes:
        if g.nodes[n]["side"] == hcatnetwork.node.ArteryNodeSide.LEFT and g.nodes[n]["topology"] == hcatnetwork.node.ArteryNodeTopology.OSTIUM:
            left_ostium = [g.nodes[n]["x"], g.nodes[n]["y"], g.nodes[n]["z"]]
        if g.nodes[n]["side"] == hcatnetwork.node.ArteryNodeSide.RIGHT and g.nodes[n]["topology"] == hcatnetwork.node.ArteryNodeTopology.OSTIUM:
            right_ostium = [g.nodes[n]["x"], g.nodes[n]["y"], g.nodes[n]["z"]]
        if left_ostium is not None and right_ostium is not None:
            break
    ostia = numpy.array([left_ostium, right_ostium], dtype=numpy.float32)
    ostia_ras = apply_affine_3d(affine_centerlines2ras, ostia.T).T
    ostia_ijk = numpy.round(
        apply_affine_3d(affine_ras2ijk, ostia_ras.T).T
    ).astype(numpy.int16)
    # 0: left, 1: right
    return ostia_ras[None, ...], ostia_ijk[None, ...]

def asoca_get_ostia_all_patients(path_to_asoca_folder: str, affine_centerlines2ras_list: list[numpy.ndarray], affine_ras2ijk_list: list[numpy.ndarray]) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Output tuple:
    - 0: ostia points in ras space: numpy.ndarray of shape (N, 2, 3)
    - 1: ostia points in ijk space: numpy.ndarray of shape (N, 2, 3)
    - 2: left/right labels: numpy.ndarray of shape (N, 2)
    """
    ostia_ras_list = []
    ostia_ijk_list = []
    for i, patient_graph in enumerate(DATASET_ASOCA_GRAPHS_RESAMPLED_05MM):
        ostia_ras, ostia_ijk = asoca_get_ostia_single_patient(
            os.path.join(path_to_asoca_folder, patient_graph),
            affine_centerlines2ras_list[i],
            affine_ras2ijk_list[i]
        )
        ostia_ras_list.append(ostia_ras)
        ostia_ijk_list.append(ostia_ijk)
    return (
        numpy.concatenate(ostia_ras_list, axis=0), 
        numpy.concatenate(ostia_ijk_list, axis=0), 
    )

if __name__ == '__main__':
    print("Testing ASOCA dataset ostia extraction")
    asoca_folder = r"/scratch/mleccardi/Data/ASOCA"
    asoca_image_files = [os.path.join(asoca_folder, f)for f in DATASET_ASOCA_TRAINING]
    images = [
        AsocaImageCT(f)
        for f in asoca_image_files
    ]
    quit(asoca_image_files)

    
    ostia_ras, ostia_ijk, ostia_left_right = asoca_get_ostia_all_patients(
        asoca_folder,
        [img.affine_centerlines2ras for img in images],
        [img.affine_ras2ijk for img in images]
    )

    # show the image and the ostia point overlapped
    for i in range(len(images)):
        img_numpy = images[i].data
        ostia_i = ostia_ijk[i].astype(int)
        plt.figure(figsize=(10, 10))
        plt.suptitle(DATASET_ASOCA_TRAINING[i])
        for v in range(len(ostia_i)):
            plt.subplot(1, len(ostia_i), v+1)
            plt.imshow(img_numpy[:,:,ostia_i[v,2]], cmap='gray')
            overlay = numpy.zeros_like(img_numpy[:,:,ostia_i[v,2]])
            overlay[ostia_i[v,0]-2:ostia_i[v,0]+3, ostia_i[v,1]-2:ostia_i[v,1]+3] = 1
            plt.imshow(overlay, alpha=0.5, 
                    cmap='Reds' if ostia_left_right[i][v] == 1 else 'Blues'
            )
            plt.xlabel("Right" if ostia_left_right[i][v] == 1 else "Left")
            plt.axis('off')
        plt.show()


'''
import os
import numpy
import json
import matplotlib.pyplot as plt

import hcatnetwork

from HearticDatasetManager.affine import apply_affine_3d
from HearticDatasetManager.asoca.image import AsocaImageCT
from HearticDatasetManager.asoca.dataset import DATASET_ASOCA_TRAINING



# get ASOCA data


def asoca_get_ostia_single_patient(path_to_asoca_patient_graph: str, affine_centerlines2ras: list[numpy.ndarray], affine_ras2ijk: list[numpy.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Output tuple:
    - 0: ostia points in ras space: torch.Tensor of shape (1, 2, 3)
    - 1: ostia points in ijk space: torch.Tensor of shape (1, 2, 3)
    - 2: left/right labels: torch.Tensor of shape (1, 2)
    """
    # Here, we do not have the ostium points, rather we just have the centerlines.
    # Centerlines are in vtp format (which is a mess to open),
    # or converted int an hcatnetwork graph.
    g = hcatnetwork.io.load_graph(
        path_to_asoca_patient_graph,
        output_type=hcatnetwork.graph.SimpleCenterlineGraph
    )
    left_ostium, right_ostium = None, None
    for n in g.nodes:
        if g.nodes[n]["side"] == hcatnetwork.node.ArteryNodeSide.LEFT and g.nodes[n]["topology"] == hcatnetwork.node.ArteryNodeTopology.OSTIUM:
            left_ostium = [g.nodes[n]["x"], g.nodes[n]["y"], g.nodes[n]["z"]]
        if g.nodes[n]["side"] == hcatnetwork.node.ArteryNodeSide.RIGHT and g.nodes[n]["topology"] == hcatnetwork.node.ArteryNodeTopology.OSTIUM:
            right_ostium = [g.nodes[n]["x"], g.nodes[n]["y"], g.nodes[n]["z"]]
        if left_ostium is not None and right_ostium is not None:
            break
    ostia = torch.tensor(
        [left_ostium, right_ostium],
        dtype=torch.float32
    )
    ostia_ras = torch.tensor(
        apply_affine_3d(affine_centerlines2ras, ostia.numpy().T).T,
        dtype=torch.float32
    )
    ostia_ijk = torch.tensor(
        apply_affine_3d(affine_ras2ijk, ostia_ras.numpy().T).T,
        dtype=torch.int16
    )
    # get the right and left ostia labels
    # 0: left, 1: right
    # in asoca, first ostium is always left, second is always right
    ostia_left_right = torch.zeros((2,), dtype=torch.uint8)
    ostia_left_right[0] = 0
    ostia_left_right[1] = 1
    # return
    # 0: ostia points in ras space
    # 1: ostia points in ijk space
    # 2: left/right labels
    return ostia_ras[None, ...], ostia_ijk[None, ...], ostia_left_right[None, ...]

def asoca_get_ostia_all_patients(path_to_asoca_folder: str, affine_centerlines2ras_list: list[numpy.ndarray], affine_ras2ijk_list: list[numpy.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Output tuple:
    - 0: ostia points in ras space: torch.Tensor of shape (N, 2, 3)
    - 1: ostia points in ijk space: torch.Tensor of shape (N, 2, 3)
    - 2: left/right labels: torch.Tensor of shape (N, 2)
    """
    ostia_ras_list = []
    ostia_ijk_list = []
    ostia_left_right_list = []
    for i, patient_graph in enumerate(DATASET_ASOCA_GRAPHS_RESAMPLED_05MM):
        ostia_ras, ostia_ijk, ostia_left_right = asoca_get_ostia_single_patient(
            os.path.join(path_to_asoca_folder, patient_graph),
            affine_centerlines2ras_list[i],
            affine_ras2ijk_list[i]
        )
        ostia_ras_list.append(ostia_ras)
        ostia_ijk_list.append(ostia_ijk)
        ostia_left_right_list.append(ostia_left_right)
    # return
    # 0: ostia points in ras space
    # 1: ostia points in ijk space
    # 2: left/right labels
    return (
        torch.cat(ostia_ras_list, dim=0), 
        torch.cat(ostia_ijk_list, dim=0), 
        torch.cat(ostia_left_right_list, dim=0)
    )


if __name__ == '__main__' and 0:
    print("Testing ASOCA dataset ostia extraction")
    asoca_folder = r"E:\MatteoLeccardi\HearticData\ASOCA"
    images = [
        AsocaImageCT(os.path.join(asoca_folder, f))
        for f in DATASET_ASOCA_TRAINING
    ]
    ostia_ras, ostia_ijk, ostia_left_right = asoca_get_ostia_all_patients(
        asoca_folder,
        [img.affine_centerlines2ras for img in images],
        [img.affine_ras2ijk for img in images]
    )
    print(f"Function asoca_get_ostia_all_patients gets:\n- ostia_ras: {ostia_ras.shape}\n- ostia_ijk: {ostia_ijk.shape}\n- ostia_left_right: {ostia_left_right.shape}")

    # show the image and the ostia point overlapped
    for i in range(len(images)):
        img_numpy = images[i].data
        ostia_i = ostia_ijk[i].numpy().astype(int)
        plt.figure(figsize=(10, 10))
        plt.suptitle(DATASET_ASOCA_TRAINING[i])
        for v in range(len(ostia_i)):
            plt.subplot(1, len(ostia_i), v+1)
            plt.imshow(img_numpy[:,:,ostia_i[v,2]], cmap='gray')
            overlay = numpy.zeros_like(img_numpy[:,:,ostia_i[v,2]])
            overlay[ostia_i[v,0]-2:ostia_i[v,0]+3, ostia_i[v,1]-2:ostia_i[v,1]+3] = 1
            plt.imshow(overlay, alpha=0.5, 
                    cmap='Reds' if ostia_left_right[i][v] == 1 else 'Blues'
            )
            plt.xlabel("Right" if ostia_left_right[i][v] == 1 else "Left")
            plt.axis('off')
        plt.show()
'''