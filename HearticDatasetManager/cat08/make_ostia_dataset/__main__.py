import os
import numpy
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from HearticDatasetManager.affine import apply_affine_3d
from HearticDatasetManager.cat08.image import Cat08ImageCT
from HearticDatasetManager.cat08.dataset import DATASET_CAT08_IMAGES


# get CAT08 ostia label data (from files named 'pointS.txt')

def cat08_get_all_patient_folders(path_to_cat08_folder: str) -> list[str]:
    """
    Get all the patient folders in the CAT08 dataset.
    """
    return [
        os.path.dirname(
            os.path.join(path_to_cat08_folder, img_subpath)
        )
        for img_subpath in DATASET_CAT08_IMAGES
    ]

def cat08_get_all_ostia_files_single_patient(path_to_patient_folder: str) -> list[str]:
    """
    Get all the ostia files in a single patient folder in the CAT08 dataset.
    """
    return [
        os.path.join(path_to_patient_folder, vessel, 'pointS.txt')
        for vessel in os.listdir(path_to_patient_folder)
        if os.path.isdir(os.path.join(path_to_patient_folder, vessel)) and "vessel" in vessel
    ]

def cat08_get_ostia_single_patient(path_to_cat08_patient_folder: str, affine_centerlines2ras: numpy.ndarray, affine_ras2ijk: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Output tuple:
    - 0: ostia points in ras space: numpy.ndarray of shape (1, 2, 3)
    - 1: ostia points in ijk space: numpy.ndarray of shape (1, 2, 3)

    It is always left first, right second:
    - (1,0,:) -> left 
    - (1,1,:) -> right
    """
    
    ostia = numpy.array(
        [
                numpy.loadtxt(
                    f,
                    delimiter=' '
                ).tolist()
                for f in cat08_get_all_ostia_files_single_patient(path_to_cat08_patient_folder)
        ],
        dtype=numpy.float32,
    )
    ostia_ras = apply_affine_3d(affine_centerlines2ras, ostia.T).T
    
    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=10, copy_x=False, n_init=5)
    # In CAT08, each ostia subset has 4 points, where
    # three of them are close together, one is far away.
    # We want to take the median point of the three close together.
    # We use KMeans to cluster the points in two clusters and take the median point of the two clusters.
    if len(ostia_ras) < 2:
        raise ValueError('Not enough ostia points, there should be 4 for this dataset. Found instead: ', len(ostia_ras))
    
    # cluster the points in two clusters and take the median point of the two clusters
    kmeans.fit(ostia_ras)
    cluster1 = ostia_ras[kmeans.labels_ == 0]
    cluster2 = ostia_ras[kmeans.labels_ == 1]
    med_cluster1 = numpy.median(cluster1, axis=0)
    med_cluster2 = numpy.median(cluster2, axis=0)
    ostia_ras_two = numpy.array([med_cluster1, med_cluster2])
    
    # 0: left, 1: right
    # this method is simplicistic, but it works for CAT08
    ostia_1 = ostia_ras_two[0]
    ostia_2 = ostia_ras_two[1]
    if ostia_1[0] > ostia_2[0]:
        # right is before left -> correct
        ostia_ras_two = ostia_ras_two[::-1]
        
    # transform all points to IJK space
    ostia_ijk = numpy.array(
        apply_affine_3d(affine_ras2ijk, ostia_ras_two.T).T,
        dtype=numpy.int16
    )
    
    # return
    # 0: ostia points in ras space
    # 1: ostia points in ijk space
    out = (
        ostia_ras_two[None,...], 
        ostia_ijk[None,...]
    )
    return out

def cat08_get_ostia_all_patients(path_to_cat08_folder: str, affine_centerlines2ras_list: list[numpy.ndarray], affine_ras2ijk_list: list[numpy.ndarray]) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Output tuple:
    - 0: ostia points in ras space: numpy.ndarray of shape (N, 2, 3)
    - 1: ostia points in ijk space: numpy.ndarray of shape (N, 2, 3)
    - 2: left/right labels: numpy.ndarray of shape (N, 2)
    """
    ostia_ras_list = []
    ostia_ijk_list = []
    for i, patient_folder in enumerate(cat08_get_all_patient_folders(path_to_cat08_folder)):
        ostia_ras, ostia_ijk = cat08_get_ostia_single_patient(
            patient_folder,
            affine_centerlines2ras_list[i],
            affine_ras2ijk_list[i]
        )
        ostia_ras_list.append(ostia_ras)
        ostia_ijk_list.append(ostia_ijk)
    # return
    # 0: ostia points in ras space
    # 1: ostia points in ijk space
    return (
        numpy.concatenate(ostia_ras_list, axis=0), 
        numpy.concatenate(ostia_ijk_list, axis=0), 
    )






if __name__ == '__main__':
    print("CAT08 dataset ostia extraction")
    cat08_folder = r"/scratch/mleccardi/Data/CAT08/"
    cat08_image_files = [os.path.join(cat08_folder, f) for f in DATASET_CAT08_IMAGES]
    cat08_images = [
        Cat08ImageCT(f)
        for f in cat08_image_files
    ]
    
    ostia_ras, ostia_ijk = cat08_get_ostia_all_patients(
        cat08_folder,
        [img.affine_centerlines2ras for img in cat08_images],
        [img.affine_ras2ijk for img in cat08_images]
    )


    if 0:
        # show the image and the ostia point overlapped
        for i in range(len(cat08_images)):
            img_numpy = cat08_images[i].data
            ostia_i = ostia_ijk[i]

            plt.figure(figsize=(10, 10))
            plt.suptitle(DATASET_CAT08_IMAGES[i])
            for v in range(len(ostia_i)):
                plt.subplot(1, len(ostia_i), v+1)
                plt.imshow(img_numpy[:,:,ostia_i[v,2]], cmap='gray')
                overlay = numpy.zeros_like(img_numpy[:,:,ostia_i[v,2]])
                overlay[ostia_i[v,0]-2:ostia_i[v,0]+3, ostia_i[v,1]-2:ostia_i[v,1]+3] = 1
                plt.imshow(overlay, alpha=0.5, 
                        cmap='Reds' if v == 1 else 'Blues'
                )
                plt.xlabel("Right" if v == 1 else "Left")
                plt.axis('off')
            plt.show()

    # Save all ras and ijk ostia positions in the dataset
    for i, patient_folder in enumerate(cat08_get_all_patient_folders(cat08_folder)):
        # save ras
        patient_number = patient_folder[-2:]
        file_name = f'ostia{patient_number}.json'
        data = {
            'ijk': {
                'left': [int(x) for x in ostia_ijk[i,0,:]],
                'right': [int(x) for x in ostia_ijk[i,1,:]]
            },
            'ras': {
                'left': [round(float(x), 6) for x in ostia_ras[i,0,:]],
                'right': [round(float(x), 6) for x in ostia_ras[i,1,:]]
            }
        }
        out_path = os.path.join(patient_folder, file_name)
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
    print('Ostia converted and saved for CAT08 dataset as jsons inside each image folder.')




