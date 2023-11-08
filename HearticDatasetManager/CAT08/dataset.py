"""Here are stored the standard names of all the datasets.
"""
# All images in the cat08 dataset
DATASET_CAT08_IMAGES = [
    "dataset00/image00.mhd",
    "dataset01/image01.mhd",
    "dataset02/image02.mhd",
    "dataset03/image03.mhd",
    "dataset04/image04.mhd",
    "dataset05/image05.mhd",
    "dataset06/image06.mhd",
    "dataset07/image07.mhd",
    "dataset08/image08.mhd",
    "dataset09/image09.mhd",
    "dataset10/image10.mhd",
    "dataset11/image11.mhd",
    "dataset12/image12.mhd",
    "dataset13/image13.mhd",
    "dataset14/image14.mhd",
    "dataset15/image15.mhd",
    "dataset16/image16.mhd",
    "dataset17/image17.mhd",
    "dataset18/image18.mhd",
    "dataset19/image19.mhd",
    "dataset20/image20.mhd",
    "dataset21/image21.mhd",
    "dataset22/image22.mhd",
    "dataset23/image23.mhd",
    "dataset24/image24.mhd",
    "dataset25/image25.mhd",
    "dataset26/image26.mhd",
    "dataset27/image27.mhd",
    "dataset28/image28.mhd",
    "dataset29/image29.mhd",
    "dataset30/image30.mhd",
    "dataset31/image31.mhd"
]

# Images for which the cat08 dataset provides the ground truth centerlines
DATASET_CAT08_IMAGES_TRAINING = DATASET_CAT08_IMAGES[:8]

# Images for which the cat08 dataset provides no ground truth
DATASET_CAT08_IMAGES_TESTING = DATASET_CAT08_IMAGES[8:]

# HCATNetwork graphs
DATASET_CAT08_GRAPHS = [
    "centerlines_graphs/dataset00.GML",
    "centerlines_graphs/dataset01.GML",
    "centerlines_graphs/dataset02.GML",
    "centerlines_graphs/dataset03.GML",
    "centerlines_graphs/dataset04.GML",
    "centerlines_graphs/dataset05.GML",
    "centerlines_graphs/dataset06.GML",
    "centerlines_graphs/dataset07.GML"
]

DATASET_CAT08_GRAPHS_RESAMPLED_05MM = [
    n.replace(".GML", "_0.5mm.GML") if n.find("_0.5mm.GML") == -1 else n for n in DATASET_CAT08_GRAPHS
]
