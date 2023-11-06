import os, sys
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import networkx
import hcatnetwork
import HearticDatasetManager.cat08.convert_to_hcatnetwork.utils as util

"""How does this script work:
This script works in two modes, decided by setting the flag "OPTION_EXPLORE_CENTERLINE".
OPTION_EXPLORE_CENTERLINE = True:
    The program is used to explore the centelrines point by point.
    This is necessary to create segments: a segment is a section of centerlines where the
    centerline either runs alone or it runs with other centerlines side by side, from start to intersection
    or from intersection to intersection.
OPTION_EXPLORE_CENTERLINE = False:
    Given the segments info obtained in the previous step, and saved in the format explained below,
    the centerline graph is created and saved using the HCATNetwork (https://github.com/AAMIASoftwares-research/HCATNetwork), based on NetworkX.
"""
OPTION_EXPLORE_CENTERLINE = 0

IM_NUMBER = 0 # accepted 0 to 7
IM_NUMBER = int(IM_NUMBER)

# Respacing of the graph - this is the original mean respacing described in the paper
POINTS_TARGET_SPACING = 0.03 # mm -> each point of the graph will be, more or less, 0.03 mm apart


CAT08_IM_folder = os.path.normpath(
    f"C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\dataset{IM_NUMBER:02d}\\"
)

# The next is a list of tuples (start_index, end_index, category)
# where end_index is the index up to which (excluded) the vessel points belong to the category,
# which is encoded as a simple integer.
# Later, points of different centerlines belonging to the same category will be processed
# and a mean will be taken.
# Also, a connection list is presented, which shows which category connects into which other.
# A self loop means no connections, es (0,0) means that segment 0 is not connected to anything.
# IM00
cat08_00_00_indices_cathegories = [
    (0, 7354, 0)
]
cat08_00_01_indices_cathegories = [
    (0, 651, 1),
    (651, 976, 2),
    (976, 3277, 5)
]
cat08_00_02_indices_cathegories = [
    (0, 655, 1),
    (655, 4114, 3)
]
cat08_00_03_indices_cathegories = [
    (0, 650, 1),
    (650, 981, 2),
    (981, 2352, 4)
]
cat08_00_indices_cathegories = [cat08_00_00_indices_cathegories, cat08_00_01_indices_cathegories, cat08_00_02_indices_cathegories, cat08_00_03_indices_cathegories]
cat08_00_connections = [(0,0), (1,2), (1,3), (2,4), (2,5)]
# IM01
cat08_01_00_indices_cathegories = [
    (0, 7450, 0)
]
cat08_01_01_indices_cathegories = [
    (0, 741, 1),
    (741, 2057, 2),
    (2057, 5821, 5)
]
cat08_01_02_indices_cathegories = [
    (0, 665, 1),
    (665, 3501, 3)
]
cat08_01_03_indices_cathegories = [
    (0, 620, 1),
    (620, 1951, 2),
    (1951, 3624, 4)
]
cat08_01_indices_cathegories = [cat08_01_00_indices_cathegories, cat08_01_01_indices_cathegories, cat08_01_02_indices_cathegories, cat08_01_03_indices_cathegories]
cat08_01_connections = [(0,0), (1,2), (1,3), (2,4), (2,5)]
# IM02
cat08_02_00_indices_cathegories = [
    (0, 5381, 0)
]
cat08_02_01_indices_cathegories = [
    (0, 511, 1),
    (511, 755, 2),
    (755, 4812, 5)
]
cat08_02_02_indices_cathegories = [
    (0, 501, 1),
    (501, 3331, 3)
]
cat08_02_03_indices_cathegories = [
    (0, 551, 1),
    (551, 795, 2),
    (795, 2336, 4)
]
cat08_02_indices_cathegories = [cat08_02_00_indices_cathegories, cat08_02_01_indices_cathegories, cat08_02_02_indices_cathegories, cat08_02_03_indices_cathegories]
cat08_02_connections = [(0,0), (1,2), (1,3), (2,4), (2,5)]
# IM03
cat08_03_00_indices_cathegories = [
    (0, 6791, 0)
]
cat08_03_01_indices_cathegories = [
    (0, 516, 1),
    (516, 1301, 2),
    (1301, 5290, 5)
]
cat08_03_02_indices_cathegories = [
    (0, 481, 1),
    (481, 4350, 3)
]
cat08_03_03_indices_cathegories = [
    (0, 507, 1),
    (507, 1285, 2),
    (1285, 3013, 4)
]
cat08_03_indices_cathegories = [cat08_03_00_indices_cathegories, cat08_03_01_indices_cathegories, cat08_03_02_indices_cathegories, cat08_03_03_indices_cathegories]
cat08_03_connections = [(0,0), (1,2), (1,3), (2,4), (2,5)]
# IM04
cat08_04_00_indices_cathegories = [
    (0, 6171, 0)
]
cat08_04_01_indices_cathegories = [
    (0, 325, 1),
    (325, 985, 2),
    (985, 4975, 5)
]
cat08_04_02_indices_cathegories = [
    (0, 335, 1),
    (335, 3723, 3)
]
cat08_04_03_indices_cathegories = [
    (0, 305, 1),
    (305, 965, 2),
    (965, 2956, 4)
]
cat08_04_indices_cathegories = [cat08_04_00_indices_cathegories, cat08_04_01_indices_cathegories, cat08_04_02_indices_cathegories, cat08_04_03_indices_cathegories]
cat08_04_connections = [(0,0), (1,2), (1,3), (2,4), (2,5)]
# IM05
cat08_05_00_indices_cathegories = [
    (0, 4260, 0)
]
cat08_05_01_indices_cathegories = [
    (0, 570, 1),
    (570, 1400, 2),
    (1400, 5207, 5)
]
cat08_05_02_indices_cathegories = [
    (0, 592, 1),
    (592, 2396, 3)
]
cat08_05_03_indices_cathegories = [
    (0, 545, 1),
    (545, 1380, 2),
    (1380, 2553, 4)
]
cat08_05_indices_cathegories = [cat08_05_00_indices_cathegories, cat08_05_01_indices_cathegories, cat08_05_02_indices_cathegories, cat08_05_03_indices_cathegories]
cat08_05_connections = [(0,0), (1,2), (1,3), (2,4), (2,5)]
# IM06
cat08_06_00_indices_cathegories = [
    (0, 4260, 0)
]
cat08_06_01_indices_cathegories = [
    (0, 475, 1),
    (475, 915, 2),
    (915, 5207, 5)
]
cat08_06_02_indices_cathegories = [
    (0, 485, 1),
    (485, 2396, 3)
]
cat08_06_03_indices_cathegories = [
    (0, 470, 1),
    (470, 900, 2),
    (900, 2553, 4)
]
cat08_06_indices_cathegories = [cat08_06_00_indices_cathegories, cat08_06_01_indices_cathegories, cat08_06_02_indices_cathegories, cat08_06_03_indices_cathegories]
cat08_06_connections = [(0,0), (1,2), (1,3), (2,4), (2,5)]
# IM07
cat08_07_00_indices_cathegories = [
    (0, 6051, 0)
]
cat08_07_01_indices_cathegories = [
    (0, 360, 1),
    (360, 5874, 5)
]
cat08_07_02_indices_cathegories = [
    (0, 330, 1),
    (330, 465, 2),
    (465, 2111, 3)
]
cat08_07_03_indices_cathegories = [
    (0, 350, 1),
    (350, 485, 2),
    (485, 2287, 4)
]
cat08_07_indices_cathegories = [cat08_07_00_indices_cathegories, cat08_07_01_indices_cathegories, cat08_07_02_indices_cathegories, cat08_07_03_indices_cathegories]
cat08_07_connections = [(0,0), (1,2), (1,5), (2,3), (2,4)]

# attach everything together to gather these info programmatically and automatically
cat08_ic_list = [
    cat08_00_indices_cathegories,
    cat08_01_indices_cathegories,
    cat08_02_indices_cathegories,
    cat08_03_indices_cathegories,
    cat08_04_indices_cathegories,
    cat08_05_indices_cathegories,
    cat08_06_indices_cathegories,
    cat08_07_indices_cathegories
]
cat08_conn_list = [
    cat08_00_connections,
    cat08_01_connections,
    cat08_02_connections,
    cat08_03_connections,
    cat08_04_connections,
    cat08_05_connections,
    cat08_06_connections,
    cat08_07_connections
]



if __name__ == "__main__":
    # load centelrines
    centerlines_list = util.importCenterlineList(CAT08_IM_folder)
    if 0:
        util.plotCenterlineList(centerlines_list)
    if not OPTION_EXPLORE_CENTERLINE:
        n_cat08_img = int(os.path.split(CAT08_IM_folder)[1][-2:])
        centerlines_ic_list = cat08_ic_list[n_cat08_img]
        centerlines_cat_conn = cat08_conn_list[n_cat08_img]
    else:
        # This visualisation tool is used to get the indices, for each centerline, of the
        # common segments manually, since doing it automatically is requiring too much development time
        # and is not worth it.
        # the results are saved, for each centerline, on top of the main section of this script
        util.plotCenterlineListWithIndexes(centerlines_list)
        sys.exit(0)

    # divide left and right arterial tree
    t1_i, t1_list, t2_i, t2_list = util.divideArterialTreesFromOriginalCenterlines(centerlines_list)
    t1_ic_list = [centerlines_ic_list[i] for i in t1_i]
    t2_ic_list = [centerlines_ic_list[i] for i in t2_i]
    if 0:
        util.plotCenterlineList(t1_list)
        util.plotCenterlineList(t2_list)
    
    # Now you have, for each tree, a series of indices and cathegories telling you
    # what segments in each centelrine should be united with other centerlines, and with which ones.
    print("Working on single segments...")
    t1_final_segments_tuples_list = util.routineMergeCenterlineCommonSegments(t1_list, t1_ic_list)
    t2_final_segments_tuples_list = util.routineMergeCenterlineCommonSegments(t2_list, t2_ic_list)

    if 1:
        # final plot of all disjointed segments
        ax = plt.subplot(111, projection="3d")
        for ccc in t1_final_segments_tuples_list:
            ax.plot(ccc[0][:,0], ccc[0][:,1], ccc[0][:,2],".-")
        plt.show()
        ax = plt.subplot(111, projection="3d")
        for ccc in t2_final_segments_tuples_list:
            ax.plot(ccc[0][:,0], ccc[0][:,1], ccc[0][:,2],".-")
        plt.show()

    # Now, just stitch the segments together
    # following the connection lists in a graph
    print("Building the graph...")
    g_dict = hcatnetwork.graph.SimpleCenterlineGraphAttributes()
    g_dict["image_id"] = f"CAT08/dataset{IM_NUMBER:02d}"
    g_dict["are_left_right_disjointed"] = True
    graph_bcg = hcatnetwork.graph.SimpleCenterlineGraph(g_dict)
    # t1 - RCA
    util.buildAndConnectGraph(
        graph_bcg,
        t1_final_segments_tuples_list,
        centerlines_cat_conn,
        tree_class=hcatnetwork.node.ArteryNodeSide.RIGHT,
        graph_nodes_target_spacing_mm=POINTS_TARGET_SPACING
    )
    print(" RCA done")
    # t2 - LCA
    util.buildAndConnectGraph(
        graph_bcg,
        t2_final_segments_tuples_list,
        centerlines_cat_conn,
        tree_class=hcatnetwork.node.ArteryNodeSide.LEFT,
        graph_nodes_target_spacing_mm=POINTS_TARGET_SPACING
    )
    print(" LCA done")
    


    #######
    # Viz
    #######

    if 1:
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph_bcg)
    


    ########################
    # Save the graph to file
    ########################

    # original one
    graph_save_path = f"C:\\Users\\lecca\\Desktop\\GraphsCAT08\\dataset{IM_NUMBER:02d}.GML"
    hcatnetwork.io.save_graph(graph_bcg, graph_save_path)
    print("saved " + graph_save_path)

    # Resampled to 0.5 mm
    graph_save_path = f"C:\\Users\\lecca\\Desktop\\GraphsCAT08\\dataset{IM_NUMBER:02d}_0.5mm.GML"
    graph_05mm = graph_bcg.resample(mm_between_nodes=0.5)
    hcatnetwork.io.save_graph(graph_05mm, graph_save_path)
    print("saved " + graph_save_path)

    

    ###########################
    # double-checking
    ###########################

    print("DONE\n\n")
    graph_reloaded_05mm = hcatnetwork.io.load_graph(graph_save_path, output_type=hcatnetwork.graph.SimpleCenterlineGraph)
    hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph_reloaded_05mm)
    
    sys.exit()





    


























    #######################
    #### LEGACY STUFF #####
    #######################

    # The following stuff all proved to be almost working, but not completely working
    # These are kept because, in any case, they might contain some useful stuff for the future

    # less old method
    # Worked almost fine, but messy at intersections and points were not ordered from ostium onward.


    # get connection matrix
    # the connection matrix N x N, with N the number of centerlines in the single arterial tree,
    # stores the index of the ith (row) centerline for which, after that index, the centelrine is
    # no more connected to the jth (column) centelrine.
    # Should be symmetric if the analysed centerline are all from the same tree (same ostium), 
    # but not symmetric if centerlines of different trees are used.
    # tree 1
    t1_connection_matrix = numpy.zeros((len(t1_list), len(t1_list)), dtype="int") - 1
    for i_v in range(len(t1_list)):
        for j_v in range(len(t1_list)):
            if i_v != j_v:
                # work in i_v
                t1_connection_matrix[i_v, j_v] = util.getCentelrinesFurthestConnectionIndex(
                    c_i=t1_list[i_v],
                    c_j=t1_list[j_v], 
                    thresh_radius_multiplier=0.3
                )
    # tree 2
    t2_connection_matrix = numpy.zeros((len(t2_list), len(t2_list)), dtype="int") - 1
    for i_v in range(len(t2_list)):
        for j_v in range(len(t2_list)):
            if i_v != j_v:
                # work in i_v
                t2_connection_matrix[i_v, j_v] = util.getCentelrinesFurthestConnectionIndex(
                    c_i=t2_list[i_v],
                    c_j=t2_list[j_v], 
                    thresh_radius_multiplier=0.3
                )
        
    if 0:
        print("debug plot")
        import matplotlib.pyplot as plt
        for i in range(len(t2_list)):
            plt.scatter(t2_list[i][:,0], t2_list[i][:,1], s=20)
        for i_v in range(len(t2_list)):
            for j_v in range(len(t2_list)):
                if t2_connection_matrix[i_v, j_v] != -1:
                    idx = t2_connection_matrix[i_v, j_v]
                    plt.scatter(
                        t2_list[i_v][idx,0],
                        t2_list[i_v][idx,1],
                        s=10, c="red")
                    plt.plot(
                        [t2_list[i_v][t2_connection_matrix[i_v, j_v],0], t2_list[j_v][t2_connection_matrix[j_v, i_v],0]],
                        [t2_list[i_v][t2_connection_matrix[i_v, j_v],1], t2_list[j_v][t2_connection_matrix[j_v, i_v],1]],
                        color="red"
                    )
        plt.axis("equal")
        plt.show()

    # mean for each centerline with the closest point on the connected centelrines
    print("Connection Matrix:\n", t2_connection_matrix)

    """ Usa le distanze e classifica usando un int

    """
    t2_class_list = []
    for i_v, v_i in enumerate(t2_list):
        point_class_list = [0 for i in range(v_i.shape[0])]
        for i_p, p in enumerate(v_i):
            if i_p < min(t2_connection_matrix[i_v,t2_connection_matrix[i_v,:]>-1]):
                # first junction not reached yet
                point_class_list[i_p] = sum([(c+1)+10**c for c in range(len(t2_list))])
            else:
                # the first junction ever has been reached -> infer the class with the connection matrix
                point_class_list[i_p] = sum([(c+1)*(10**c) for c in range(len(t2_list)) if (i_p < t2_connection_matrix[i_v,c] and i_p != c)])
                ## the line above is wrong, sistemala
        t2_class_list.append(point_class_list)

      
    if 1:
        print("debug plot")
        #print(t2_class_list)
        import matplotlib.pyplot as plt
        classes_extended = []
        points_extended = []
        for i in range(len(t2_list)):
            points_extended.extend(t2_list[i])
            classes_extended.extend(t2_class_list[i])
        points_extended = numpy.array(points_extended)
        classes_extended = numpy.array(classes_extended)
        plt.scatter(points_extended[:,0], points_extended[:,1], s=40, alpha=0.7, c=classes_extended)
        plt.axis("equal")
        plt.legend()
        plt.colorbar()
        plt.show()
        plt.plot(classes_extended, ".")
        cx = 0
        for v in t2_class_list:
            plt.vlines(len(v)+cx, ymin=-5, ymax=300, color="red")
            cx += len(v)
        plt.show()

    

    # tests on tree 2
    points_list = []
    for i_v in range(len(t2_list)):
        for i_p in range(len(t2_list[i_v])):
            if (t2_connection_matrix[i_v, :] != -1).any():
                p_to_mean = [t2_list[i_v][i_p].tolist()]
                for j_v in range(len(t2_list)):
                    if t2_connection_matrix[i_v, j_v] != -1 and i_p < t2_connection_matrix[i_v, j_v]:
                        idx_min, _ = util.getPointToCenterlinePointsMinDistance(
                            p=t2_list[i_v][i_p],
                            centerline=t2_list[j_v]
                        )
                        if idx_min < t2_connection_matrix[j_v, i_v]:
                            p_to_mean.append(t2_list[j_v][idx_min].tolist())
                p_mean = numpy.mean(p_to_mean, axis=0).tolist() if len(p_to_mean) > 1 else p_to_mean[0]
            else:
                p_mean = t2_list[i_v][i_p].tolist()
            if not p_mean in points_list:
                points_list.append(p_mean)
    points_list = numpy.array(points_list)

    if 1:
        print("debug plot")
        import matplotlib.pyplot as plt
        for i in range(len(t2_list)):
            plt.scatter(t2_list[i][:,0], t2_list[i][:,1], s=40, alpha=0.7)
        plt.scatter(points_list[:,0], points_list[:,1], s=10)
        plt.axis("equal")
        plt.show()
        






    sys.exit()
    # old method
    

    # Now find the almost-mean-shift
    spacing=0.29
    max_length = numpy.max([getCentelrineLength(c) for c in v_list])
    tuple_list = []
    r_thresh_modifier_count = 0
    len_dbscan_set_previous = 0 
    d_ostium_array = numpy.linspace(0,max_length, int(max_length/spacing))
    i_ = 0
    while i_ < d_ostium_array.shape[0]:
        d_ostium = d_ostium_array[i_]
        p_d = [getContinuousCenterline(c, d_ostium) for c in v_list]
        p_d_pruned = [x for x in p_d if x is not None]
        r = numpy.mean([p[3] for p in p_d_pruned])
        if r_thresh_modifier_count == 0:
            r_thresh = 0.75*numpy.exp(-0.8*d_ostium/10) + 0.25
        else:
            r_thresh = 0.12
            r_thresh_modifier_count -= 1
        db = DBSCAN(
            eps=r_thresh*r,
            min_samples=1
        ).fit(numpy.array(p_d_pruned)[:,:3])
        if i_ > 0 and r_thresh_modifier_count==0:
            if len(set(db.labels_)) > len_dbscan_set_previous:
                # This means that, since the last step, something branched off
                # To make sure that the point in which it branches off is very near to the previous segment
                # (no hard turns), we go back 3 < n < 5 steps and temporarily set the 
                # segments-branching threshold (eps in DBSCAN) to a much lower value, so that
                # the interested segments can branch off before they would do with the previous threshold.
                n_backwards = 4
                i_ = max(0, i_ - n_backwards - 1)
                r_thresh_modifier_count = n_backwards + 1
                for ii in range(
                    min(len(tuple_list),n_backwards+1)
                    ):
                    tuple_list.pop()
                len_dbscan_set_previous = len(set(db.labels_))
                continue
        if r_thresh_modifier_count == 0:
            len_dbscan_set_previous = len(set(db.labels_))
        p_out_list = []
        for index in set(db.labels_):
            idx_positions = numpy.argwhere(db.labels_ == index).flatten()                
            if len(idx_positions) < 1:
                raise RuntimeError("Should always be at least 1")
            elif len(idx_positions) == 1:
                p_out_list.append(p_d_pruned[idx_positions[0]])
            else:
                p_d_pruned_out = numpy.array(
                    [p_d_pruned[i] for i in idx_positions]
                )
                p_new = numpy.mean(
                    p_d_pruned_out,
                    axis=0
                )
                p_out_list.append(p_new)
        tuple_list.append((d_ostium, p_out_list))
        # Update iterator
        i_ += 1
        print(f"{100*i_/len(d_ostium_array):3.1f}", end="\r")
    print("")

    vl, cl = [], []
    for t in tuple_list:
        for v in t[1]:
            vl.append([v[0], v[1], v[2], v[3]])
            cl.append(t[0])
    vl = numpy.array(vl)
    cl = numpy.array(cl).flatten()
    # plots
    if 0:
        for i_ in range(4):
            plt.plot(v_list[i_][:,0],v_list[i_][:,1])
        plt.scatter(vl[:,0], vl[:,1], c=cl)
        plt.show()

    ## now, we do a final dbscan so we can separate the two arterial trees (L and R)
    # Note that this step is possible under the assumption that the arterial trees are disconnected
    # This happens in 95% of humans, but not in all of them
    # This assumption i smore than ok for CAT08 and any other dataset annotations probably,
    # but for an inferred arterial tree this should not be taken for granted.
    min_distance_between_two_trees = 2.5 #mm
    db = DBSCAN(eps=min_distance_between_two_trees, min_samples=10).fit(vl[:,:3])
    tree1 = vl[numpy.argwhere(db.labels_==0).flatten(),:]
    d_ostium1 = cl[numpy.argwhere(db.labels_==0).flatten()]
    tree2 = vl[numpy.argwhere(db.labels_==1).flatten(),:]
    d_ostium2 = cl[numpy.argwhere(db.labels_==1).flatten()]

    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.scatter(tree1[:,0], tree1[:,1], tree1[:,2], s=10*tree1[:,3], c=d_ostium1, cmap="plasma")
    ax.scatter(tree2[:,0], tree2[:,1], tree2[:,2], s=10*tree2[:,3], c=d_ostium2, cmap="rainbow")
    plt.show()

    # now, it only remains to create the final graph




   
        






