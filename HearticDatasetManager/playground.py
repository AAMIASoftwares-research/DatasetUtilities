import os, sys
import numpy
import networkx, hcatnetwork

import matplotlib.pyplot as plt


if __name__ == "__main__" and 0:
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

    # View in slicer
    from .asoca.dataset import DATASET_ASOCA_IMAGES_DICT
    from .asoca.image import AsocaImageCT
    folder = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\".replace("\\", "/")
    file = os.path.join(
        folder,
        DATASET_ASOCA_IMAGES_DICT["Normal"][0]
    )
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
    if 1:
        A = get_affine_3d_rotation_around_vector(
            numpy.array([0, 0, 1]),
            numpy.array([-50, -50, image.origin[2]]),
            numpy.pi/6
        )
        ax.plot(
            [-50, -50],
            [-50, -50],
            [image.origin[2], image.origin[2]+50],
            c="r",
            linewidth=1,
            zorder=100
        )
        points_to_sample = apply_affine_3d(A, points_to_sample.T).T
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










#####
# VIDEO
#####

# A video of the view of the artery from a centelrine perspective along a path



if __name__ == "__main__":
    # get the image
    folder = os.path.normpath(
        "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\Data\\ASOCA\\"
    )
    from .asoca.dataset import DATASET_ASOCA_IMAGES_DICT
    from .asoca.image import AsocaImageCT
    file = os.path.join(
        folder,
        DATASET_ASOCA_IMAGES_DICT["Normal"][0]
    )
    image = AsocaImageCT(file)

    # get the centerline graph path
    from .asoca.dataset import DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT
    file = os.path.join(
        folder,
        DATASET_ASOCA_GRAPHS_RESAMPLED_05MM_DICT["Normal"][0]
    )
    graph = hcatnetwork.io.load_graph(
        file,
        output_type=hcatnetwork.graph.SimpleCenterlineGraph
    )
    # choose the path to show
    # hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph)

    # get centelrine path - LCA
    from .affine import apply_affine_3d
    ostium_id, endpoint_id = "2924", "7164"
    path = networkx.shortest_path(graph, ostium_id, endpoint_id)
    path_lengths = networkx.shortest_path_length(graph, ostium_id, endpoint_id)
    # to RAS
    path_ras = numpy.array(
        [
            [graph.nodes[node_id]["x"], graph.nodes[node_id]["y"], graph.nodes[node_id]["z"]]
            for node_id in path
        ]
    )
    path_ras = apply_affine_3d(image.affine_centerlines2ras, path_ras.T).T
    # check alignment
    
    if 0:
        fig = plt.figure("prova")
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(image.origin[0], image.origin[1], image.origin[2], c="r", s=40)
        ax.add_artist(image.bounding_box.get_artist())
        ax.plot(path_ras[:,0], path_ras[:,1], path_ras[:,2], c="r", linewidth=1, zorder=100)
        # image scatter
        z = path_ras[0,2]
        xs = numpy.linspace(image.bounding_box["lower"][0]-2, image.bounding_box["upper"][0]+2, 100)
        ys = numpy.linspace(image.bounding_box["lower"][1]-2, image.bounding_box["upper"][1]+2, 100)
        points_to_sample = numpy.zeros((len(xs)*len(ys), 3))
        i = 0
        for x in xs:
            for y in ys:
                points_to_sample[i,:] = [x, y, z]
                i += 1
        samples = image.sample(points_to_sample.T)
        ax.scatter(points_to_sample[:,0], points_to_sample[:,1], points_to_sample[:,2], c=samples, cmap="gray", s=10, linewidths=0.0, antialiased=False)
        # plot
        ax.set_xlim(image.bounding_box["lower"][0]-10, image.bounding_box["upper"][0]+10)
        ax.set_ylim(image.bounding_box["lower"][1]-10, image.bounding_box["upper"][1]+10)
        ax.set_zlim(image.bounding_box["lower"][2]-10, image.bounding_box["upper"][2]+10)
        ax.autoscale_view()
        plt.show()
        quit()

    # make the figure with two axes
    
    fig = plt.figure("2")
    ax = fig.add_subplot(111)
    
    #im_hu = image.sample(path_ras.T, interpolation="linear")
    #im_hu = numpy.vstack([im_hu]*5)
    N_PER_SIDE = 30
    points_to_sample = numpy.zeros((N_PER_SIDE*N_PER_SIDE, 3))
    im3 = numpy.zeros((N_PER_SIDE, N_PER_SIDE, len(path)))
    for i, node_id in enumerate(path):
        print(i, len(path))
        x, y, z = path_ras[i, :]
        Dx = numpy.linspace(x-15, x+15, N_PER_SIDE)
        Dy = numpy.linspace(y-15, y+15, N_PER_SIDE)
        for j, dx in enumerate(Dx):
            for k, dy in enumerate(Dy):
                points_to_sample[j*N_PER_SIDE+k, :] = [dx, dy, z]
        samples = image.sample(points_to_sample.T, interpolation="nearest")
        samples = samples.reshape((N_PER_SIDE, N_PER_SIDE))
        im3[:,:,i] = samples
    
    im_a = ax.imshow(
        im3[:,:,0],
        cmap="gray",
        aspect="equal",
        vmin = -10,
        vmax = 1000
    )
    
    for i in range(len(path)):
        im_a.set_data(im3[:,:,i])
        ax.set_title(f"Frame {i} of {len(path)}")
        fig.canvas.draw_idle()
        plt.pause(0.1)
    plt.show()

    # cycle