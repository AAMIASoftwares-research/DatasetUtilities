import os, sys
import numpy


if __name__ == "__main__":
    print("Running 'HearticDatasetManager.playground' module")
    
    ## view a saved centerline graph
    import hcatnetwork, networkx
    
    from .cat08.dataset import DATASET_CAT08_GRAPHS_RESAMPLED_05MM
    folder = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\".replace("\\", "/")
    file = os.path.join(
        folder,
        DATASET_CAT08_GRAPHS_RESAMPLED_05MM[0]
    )
    g = hcatnetwork.io.load_graph(
        file,
        output_type=hcatnetwork.graph.SimpleCenterlineGraph
    )
    hcatnetwork.draw.draw_simple_centerlines_graph_2d(g)

    from .asoca.dataset import DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT
    folder = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\Data\\ASOCA\\".replace("\\", "/")
    file = os.path.join(
        folder,
        DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT["Normal"][4]
    )
    g = hcatnetwork.io.load_graph(
        file,
        output_type=hcatnetwork.graph.SimpleCenterlineGraph
    )
    hcatnetwork.draw.draw_simple_centerlines_graph_2d(g)

    # View in in slicer
    from .asoca.dataset import DATASET_ASOCA_IMAGES_DICT
    from .asoca.image import AsocaImageCT
    folder = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\Data\\ASOCA\\".replace("\\", "/")
    file = os.path.join(
        folder,
        DATASET_ASOCA_IMAGES_DICT["Normal"][4]
    )
    quit()
    image = AsocaImageCT(file)
    save_folder = "C:\\users\\lecca\\desktop\\slicer_examples_asoca_normal_3\\".replace("\\", "/")
    hcatnetwork.utils.slicer.convert_graph_to_3dslicer_fiducials(
        g,
        save_folder+"fiducials_asoca_normal_3",
        affine_transformation_matrix=image.affine_centerlines2ras
    )
    hcatnetwork.utils.slicer.convert_graph_to_3dslicer_opencurve(
        g,
        save_folder,
        affine_transformation_matrix=image.affine_centerlines2ras
    )

    # view image in 3d in python
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.set_facecolor("#010238")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor("#010238")
    ax.grid(False)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # Origin and BB
    ax.scatter(image.origin[0], image.origin[1], image.origin[2], c="r", s=40)
    ax.add_collection(image.bounding_box.get_artist())
    ax.set_xlim(image.bounding_box["lower"][0]-100, image.bounding_box["upper"][0]+100)
    ax.set_ylim(image.bounding_box["lower"][1]-100, image.bounding_box["upper"][1]+100)
    ax.set_zlim(image.bounding_box["lower"][2]-100, image.bounding_box["upper"][2]+100)
    
    # Slice
    z_ras = image.origin[2]+5
    xs = numpy.linspace(image.bounding_box["lower"][0]-2, image.bounding_box["upper"][0]+2, 200)
    ys = numpy.linspace(image.bounding_box["lower"][1]-2, image.bounding_box["upper"][1]+2, 200)
    points_to_sample = []
    for x in xs:
        for y in ys:
            points_to_sample.append([x, y, z_ras])
    points_to_sample = numpy.array(points_to_sample)
    from .affine import apply_affine_3d, get_affine_3d_rotation_around_vector
    if 0:
        A = get_affine_3d_rotation_around_vector(
            numpy.array([0, 0, 1]),
            numpy.pi/6
        )
        points_to_sample = apply_affine_3d(points_to_sample.T, A).T
    samples = image.sample(points_to_sample.T, interpolation="linear")
    ax.scatter(
        points_to_sample[:,0],
        points_to_sample[:,1],
        points_to_sample[:,2],
        c=samples,
        cmap="gray",
        s=10,
        linewidths=0.0,
        antialiased=False
    )
    
    plt.show()


