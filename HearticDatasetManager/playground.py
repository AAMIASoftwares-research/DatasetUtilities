import os, sys
import numpy
import networkx, hcatnetwork

import matplotlib.pyplot as plt


if __name__ == "__main__" and 0:
    from .asoca.dataset import DATASET_ASOCA_IMAGES_DICT
    from .asoca.image import AsocaImageCT
    folder = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\".replace("\\", "/")
    file = os.path.join(
        folder,
        DATASET_ASOCA_IMAGES_DICT["Normal"][0]
    )
    image = AsocaImageCT(file)
    print(image.bounding_box)
    # get points to sample
    xs = numpy.linspace(-100, -80, int(20*(1/0.05)))
    ys = numpy.linspace(-30, -10, int(20*(1/0.05)))
    zs = numpy.zeros(len(xs))-140
    points = numpy.ones((3, len(xs)*len(ys)), dtype="float")
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            points[:, i*len(ys)+j] = [x, y, zs[i]]
    samples = image.sample(points, interpolation="linear")
    # plot 3d
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.add_artist(image.bounding_box.get_artist())
    ax.scatter(image.origin[0], image.origin[1], image.origin[2], c="r", s=40)
    ax.scatter(points[0,:], points[1,:], points[2,:], c=samples, s=40)
    ax.set_xlim(image.bounding_box["lower"][0]-10, image.bounding_box["upper"][0]+10)
    ax.set_ylim(image.bounding_box["lower"][1]-10, image.bounding_box["upper"][1]+10)
    ax.set_zlim(image.bounding_box["lower"][2]-10, image.bounding_box["upper"][2]+10)
    plt.show()

    # make points into 2d image
    plt.imshow(samples.reshape((len(xs), len(ys))), cmap="gray")
    plt.show()




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

if __name__ == "__main__" and 1:
    # get the image

    # fix centerlines2ras also for cat08
    # then, video of the centerline


    folder = os.path.normpath(
        "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\"
    )
    from .cat08.dataset import DATASET_CAT08_IMAGES
    from .asoca.dataset import DATASET_ASOCA_IMAGES_DICT
    from .cat08.image import Cat08ImageCT
    file = os.path.join(
        folder,
        DATASET_CAT08_IMAGES[0]
    )
    image = Cat08ImageCT(file)

    # get the centerline graph path
    from .cat08.dataset import DATASET_CAT08_GRAPHS
    file = os.path.join(
        folder,
        DATASET_CAT08_GRAPHS[0]
    )
    graph = hcatnetwork.io.load_graph(
        file,
        output_type=hcatnetwork.graph.SimpleCenterlineGraph
    )
    # choose the path to show
    # hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph); quit()

    # get centelrine path - LCA
    from .affine import apply_affine_3d
    ostium_id, endpoint_id = "2924", "7164" # asoca lad
    ostium_id, endpoint_id = "7353", "15465" # cat08 lad
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
    fig.set_facecolor("#000000")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#000000")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    N_PER_SIDE = 100
    points_to_sample = numpy.zeros((N_PER_SIDE*N_PER_SIDE*len(path), 3))
    im3 = numpy.zeros((N_PER_SIDE, N_PER_SIDE, len(path)))
    for i in range(len(path)-1)[:20]:
        print(i, len(path))
        # get direction unit vector
        v = path_ras[i+1, :] - path_ras[i, :]
        v = v / numpy.linalg.norm(v)
        # get the set of points to sample on the plane with normal v
        # -- get a vector perpendicular to v
        v1 = numpy.array([1, 0, 0]) if v[0] == 0 else numpy.array([-v[1]/v[0], 1, 0])
        v1 = v1 / numpy.linalg.norm(v1)
        # -- get a vector perpendicular to v and d
        v2 = numpy.cross(v, v1)
        v2 = v2 / numpy.linalg.norm(v2)
        # -- get the set of points
        Dx = numpy.linspace(-4.5, 4.5, N_PER_SIDE)
        Dy = numpy.linspace(-4.5, 4.5, N_PER_SIDE)
        for j, dx in enumerate(Dx):
            for k, dy in enumerate(Dy):
                points_to_sample[i*N_PER_SIDE*N_PER_SIDE+j*N_PER_SIDE+k, :] = path_ras[i, :] + v1*dx + v2*dy
    points_to_sample = points_to_sample[:20*N_PER_SIDE*N_PER_SIDE,:]
    samples = image.sample(points_to_sample.T, interpolation="linear")
    samples = samples.reshape((20, N_PER_SIDE, N_PER_SIDE)).transpose((1,2,0))
    
    plt.imshow(samples[int(N_PER_SIDE/2),:,:], cmap="gray", aspect="equal", vmin = -100, vmax = 1000)
    plt.show()
    
    im3[:,:,:20] = samples
    im_a = ax.imshow(
        im3[:,:,0],
        cmap="gray",
        aspect="equal",
        vmin = -100,
        vmax = 1000
    )
    
    for i in range(len(path)):
        im_a.set_data(im3[:,:,i])
        ax.set_title(f"Frame {i} of {len(path)}")
        fig.canvas.draw_idle()
        if graph.nodes[path[i]]["topology"] == hcatnetwork.node.ArteryNodeTopology.INTERSECTION:
            plt.pause(0.05)
        else:
            plt.pause(0.001)
    plt.show()

    fig3 = plt.figure("3")
    ax3 = fig3.add_subplot(111)
    ax3.imshow(im3[int(N_PER_SIDE/2),:,:], cmap="gray", aspect="equal", vmin = -100, vmax = 1000)
    pos_bif = []
    for i in range(len(path)):
        if graph.nodes[path[i]]["topology"] == hcatnetwork.node.ArteryNodeTopology.INTERSECTION:
            pos_bif.append(i)
    ax.scatter(pos_bif, numpy.zeros(len(pos_bif))+60, c="r", s=40)
    plt.show()

    # cycle