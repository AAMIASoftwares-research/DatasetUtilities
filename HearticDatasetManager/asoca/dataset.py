"""Here are stored the standard names of all the datasets.
"""
import os



# All images in the cat08 dataset
DATASET_ASOCA_IMAGES = [
    "Normal/CTCA/Normal_1.nrrd",
    "Normal/CTCA/Normal_2.nrrd",
    "Normal/CTCA/Normal_3.nrrd",
    "Normal/CTCA/Normal_4.nrrd",
    "Normal/CTCA/Normal_5.nrrd",
    "Normal/CTCA/Normal_6.nrrd",
    "Normal/CTCA/Normal_7.nrrd",
    "Normal/CTCA/Normal_8.nrrd",
    "Normal/CTCA/Normal_9.nrrd",
    "Normal/CTCA/Normal_10.nrrd",
    "Normal/CTCA/Normal_11.nrrd",
    "Normal/CTCA/Normal_12.nrrd",
    "Normal/CTCA/Normal_13.nrrd",
    "Normal/CTCA/Normal_14.nrrd",
    "Normal/CTCA/Normal_15.nrrd",
    "Normal/CTCA/Normal_16.nrrd",
    "Normal/CTCA/Normal_17.nrrd",
    "Normal/CTCA/Normal_18.nrrd",
    "Normal/CTCA/Normal_19.nrrd",
    "Normal/CTCA/Normal_20.nrrd",
    "Normal/Testset_Normal/0.nrrd",
    "Normal/Testset_Normal/1.nrrd",
    "Normal/Testset_Normal/2.nrrd",
    "Normal/Testset_Normal/3.nrrd",
    "Normal/Testset_Normal/4.nrrd",
    "Normal/Testset_Normal/5.nrrd",
    "Normal/Testset_Normal/6.nrrd",
    "Normal/Testset_Normal/7.nrrd",
    "Normal/Testset_Normal/8.nrrd",
    "Normal/Testset_Normal/9.nrrd",
    "Diseased/CTCA/Diseased_1.nrrd",
    "Diseased/CTCA/Diseased_2.nrrd",
    "Diseased/CTCA/Diseased_3.nrrd",
    "Diseased/CTCA/Diseased_4.nrrd",
    "Diseased/CTCA/Diseased_5.nrrd",
    "Diseased/CTCA/Diseased_6.nrrd",
    "Diseased/CTCA/Diseased_7.nrrd",
    "Diseased/CTCA/Diseased_8.nrrd",
    "Diseased/CTCA/Diseased_9.nrrd",
    "Diseased/CTCA/Diseased_10.nrrd",
    "Diseased/CTCA/Diseased_11.nrrd",
    "Diseased/CTCA/Diseased_12.nrrd",
    "Diseased/CTCA/Diseased_13.nrrd",
    "Diseased/CTCA/Diseased_14.nrrd",
    "Diseased/CTCA/Diseased_15.nrrd",
    "Diseased/CTCA/Diseased_16.nrrd",
    "Diseased/CTCA/Diseased_17.nrrd",
    "Diseased/CTCA/Diseased_18.nrrd",
    "Diseased/CTCA/Diseased_19.nrrd",
    "Diseased/CTCA/Diseased_20.nrrd",
    "Diseased/Testset_Diseased/10.nrrd",
    "Diseased/Testset_Diseased/11.nrrd",
    "Diseased/Testset_Diseased/12.nrrd",
    "Diseased/Testset_Diseased/13.nrrd",
    "Diseased/Testset_Diseased/14.nrrd",
    "Diseased/Testset_Diseased/15.nrrd",
    "Diseased/Testset_Diseased/16.nrrd",
    "Diseased/Testset_Diseased/17.nrrd",
    "Diseased/Testset_Diseased/18.nrrd",
    "Diseased/Testset_Diseased/19.nrrd"
]

DATASET_ASOCA_IMAGES = [os.path.normpath(n) for n in DATASET_ASOCA_IMAGES]

DATASET_ASOCA_IMAGES_DICT = {
    "Normal": DATASET_ASOCA_IMAGES[:20],
    "Normal Test": DATASET_ASOCA_IMAGES[20:30],
    "Diseased": DATASET_ASOCA_IMAGES[30:50],
    "Diseased Test": DATASET_ASOCA_IMAGES[50:]
}

# Images for which the cat08 dataset provides the ground truth centerlines
DATASET_ASOCA_TRAINING = DATASET_ASOCA_IMAGES_DICT["Normal"]
DATASET_ASOCA_TRAINING += DATASET_ASOCA_IMAGES_DICT["Diseased"]

# Images for which the cat08 dataset provides no ground truth
DATASET_ASOCA_TESTING = DATASET_ASOCA_IMAGES_DICT["Normal Test"]
DATASET_ASOCA_TESTING += DATASET_ASOCA_IMAGES_DICT["Diseased Test"]

# HCATNetwork graphs
DATASET_ASOCA_GRAPHS = [
    "Normal/Centerlines_graphs/Normal_1.GML",
    "Normal/Centerlines_graphs/Normal_2.GML",
    "Normal/Centerlines_graphs/Normal_3.GML",
    "Normal/Centerlines_graphs/Normal_4.GML",
    "Normal/Centerlines_graphs/Normal_5.GML",
    "Normal/Centerlines_graphs/Normal_6.GML",
    "Normal/Centerlines_graphs/Normal_7.GML",
    "Normal/Centerlines_graphs/Normal_8.GML",
    "Normal/Centerlines_graphs/Normal_9.GML",
    "Normal/Centerlines_graphs/Normal_10.GML",
    "Normal/Centerlines_graphs/Normal_11.GML",
    "Normal/Centerlines_graphs/Normal_12.GML",
    "Normal/Centerlines_graphs/Normal_13.GML",
    "Normal/Centerlines_graphs/Normal_14.GML",
    "Normal/Centerlines_graphs/Normal_15.GML",
    "Normal/Centerlines_graphs/Normal_16.GML",
    "Normal/Centerlines_graphs/Normal_17.GML",
    "Normal/Centerlines_graphs/Normal_18.GML",
    "Normal/Centerlines_graphs/Normal_19.GML",
    "Normal/Centerlines_graphs/Normal_20.GML",
    "Diseased/Centerlines_graphs/Diseased_1.GML",
    "Diseased/Centerlines_graphs/Diseased_2.GML",
    "Diseased/Centerlines_graphs/Diseased_3.GML",
    "Diseased/Centerlines_graphs/Diseased_4.GML",
    "Diseased/Centerlines_graphs/Diseased_5.GML",
    "Diseased/Centerlines_graphs/Diseased_6.GML",
    "Diseased/Centerlines_graphs/Diseased_7.GML",
    "Diseased/Centerlines_graphs/Diseased_8.GML",
    "Diseased/Centerlines_graphs/Diseased_9.GML",
    "Diseased/Centerlines_graphs/Diseased_10.GML",
    "Diseased/Centerlines_graphs/Diseased_11.GML",
    "Diseased/Centerlines_graphs/Diseased_12.GML",
    "Diseased/Centerlines_graphs/Diseased_13.GML",
    "Diseased/Centerlines_graphs/Diseased_14.GML",
    "Diseased/Centerlines_graphs/Diseased_15.GML",
    "Diseased/Centerlines_graphs/Diseased_16.GML",
    "Diseased/Centerlines_graphs/Diseased_17.GML",
    "Diseased/Centerlines_graphs/Diseased_18.GML",
    "Diseased/Centerlines_graphs/Diseased_19.GML",
    "Diseased/Centerlines_graphs/Diseased_20.GML"
]
DATASET_ASOCA_GRAPHS = [os.path.normpath(n) for n in DATASET_ASOCA_GRAPHS]

DATASET_ASOCA_GRAPHS_DICT = {
    "Normal": DATASET_ASOCA_GRAPHS[:20],
    "Diseased": DATASET_ASOCA_GRAPHS[20:]
}

DATASET_ASOCA_GRAPHS_RESAMPLED_05MM = [
    n.replace(".GML", "_0.5mm.GML") if n.find("_0.5mm.GML") == -1 else n for n in DATASET_ASOCA_GRAPHS
]

DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT = {
    "Normal": DATASET_ASOCA_GRAPHS_RESAMPLED_05MM[:20],
    "Diseased": DATASET_ASOCA_GRAPHS_RESAMPLED_05MM[20:]
}

DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_30DEG = [
    n.replace(".GML", "_intersections_30deg.GML") if n.find("_intersections_30deg.GML") == -1 else n for n in DATASET_ASOCA_GRAPHS_RESAMPLED_05MM
]

DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_30DEG_DICT = {
    "Normal": DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_30DEG[:20],
    "Diseased": DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_30DEG[20:]
}

DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_35DEG = [
    n.replace(".GML", "_intersections_35deg.GML") if n.find("_intersections_35deg.GML") == -1 else n for n in DATASET_ASOCA_GRAPHS_RESAMPLED_05MM
]

DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_35DEG_DICT = {
    "Normal": DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_35DEG[:20],
    "Diseased": DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_35DEG[20:]
}

DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_40DEG = [
    n.replace(".GML", "_intersections_40deg.GML") if n.find("_intersections_40deg.GML") == -1 else n for n in DATASET_ASOCA_GRAPHS_RESAMPLED_05MM
]

DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_40DEG_DICT = {
    "Normal": DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_40DEG[:20],
    "Diseased": DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DIRECTIONS_40DEG[20:]
}


