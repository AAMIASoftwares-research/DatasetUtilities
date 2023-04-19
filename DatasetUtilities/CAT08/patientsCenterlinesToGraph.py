import os, sys
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import networkx, HCATNetwork
import utils as util

"""How does this script work:
This script works in two modes, decided by setting the flag "OPTION_EXPLORE_CENTERLINE".
OPTION_EXPLORE_CENTERLINE = True:
    The program is used to explore the centelrines point by point.
    This is necessary to create segments: a segment is a section of centerlines where the
    centerline either runs alone or it runs with other centerlines side by side, from start to intersection
    or from intersection to intersection.
OPTION_EXPLORE_CENTERLINE = False:
    Given the segments info obtained in the previous step, and saved in the format explained below,
    the centerline graph is created and saved using the CATNetwork (https://github.com/AAMIASoftwares-research/HCATNetwork) + NetworkX packages.
"""
OPTION_EXPLORE_CENTERLINE = 0

IM_NUMBER = 1 # accepted 0 to 7
IM_NUMBER = int(IM_NUMBER)

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
    # t1
    if len(t1_ic_list) < 2:
        # centerline is alone
        t1_final_segments_tuples_list = [(t1_list[0],t1_ic_list[0][0][2])]
    else:
        # More than one centelrine: create mean segments
        # of which the points are ordered from closer to ostium (in terms of arc length)
        # to furthest.
        # Associate each point list (mean segment) to its segment cathegory index (an int)
        categories = []
        for a in t1_ic_list:
            for b in a:
                categories.append(b[2])
        t1_categories = set(categories)
        t1_final_segments_tuples_list = []
        for c in t1_categories:
            # get start and end indices
            list_of_centerline_segments = []
            for i_c, cent in enumerate(t1_list):
                for iiii, ic_tuple in enumerate(t1_ic_list[i_c]):
                    if ic_tuple[2] == c:
                        start_index = ic_tuple[0] if iiii == 0 else ic_tuple[0] + 10 # to avoid overlaps at junctions and smooth them out
                        segment = cent[start_index:ic_tuple[1]:]
                        list_of_centerline_segments.append(segment)
                        break
            # Now, the mean shift on paths algorithm should be applied
            # However, it is quite complicated and is not deemed necessary in this case
            # Solution:
            # The longest path (the segment with the most points) is selected and duplicated.
            # For each datapoint of this segment, its position and radius is averaged with the
            # closest point of each of the other segments.
            # Moreover, to have clean junctions, if the segments are not the ones containing the ostium,
            # all segments' first N points (10<N<20, ~ 3 mm) are discarded to allow a cleaner connection
            # The last step is to smooth it out a little and to sample the obtained curved once every 3 mm
            if len(list_of_centerline_segments) < 1:
                raise RuntimeError("list_of_centerline_segments should always have at least one member, instead it is empty.")
            elif len(list_of_centerline_segments) == 1:
                t1_final_segments_tuples_list.append((list_of_centerline_segments[0],c))
            else:
                # get longer segment, and use that segment indicisation to compute the mean path
                len_longest_segment = 0
                i_longest_segment = -1
                for i_seg, segment in enumerate(list_of_centerline_segments):
                    if segment.shape[0] > len_longest_segment:
                        len_longest_segment = segment.shape[0]
                        i_longest_segment = i_seg
                # now, apply the mean for each point of the longest segment with the closest points of the other segments
                final_segment = list_of_centerline_segments[i_longest_segment].copy()
                for i_p_fs in range(len_longest_segment):
                    points_to_mean_list = []
                    for j in range(len(list_of_centerline_segments)):
                        if j != i_longest_segment:
                            idxx, _ = util.getPointToCenterlinePointsMinDistance(
                                final_segment[i_p_fs],
                                list_of_centerline_segments[j]
                            )
                            points_to_mean_list.append(list_of_centerline_segments[j][idxx])
                        else:
                            points_to_mean_list.append(final_segment[i_p_fs])
                    final_segment[i_p_fs,:] = numpy.mean(points_to_mean_list, axis=0)
                # smooth it out
                for i_xyzr in range(final_segment.shape[1]):
                    fsa = numpy.insert(final_segment[:,i_xyzr], 0, final_segment[0,i_xyzr])
                    fsa = numpy.append(fsa, fsa[-1])
                    final_segment[:,i_xyzr] = numpy.convolve(fsa, [1/3, 1/3, 1/3])[2:-2]
                # now, the final segment is ready, just sample it once every 0.3 mm of arc length
                #          use getCenterlinePointFromArcLength()
                total_len = util.getCentelrineArcLength(final_segment)
                accumulated_len = 0.0
                arc_len_list = []
                while accumulated_len <= total_len:
                    arc_len_list.append(accumulated_len)
                    accumulated_len += 0.3
                resampled_final_segment = numpy.zeros((len(arc_len_list), final_segment.shape[1]))
                for i in range(len(arc_len_list)):
                    resampled_final_segment[i,:] = util.getCenterlinePointFromArcLength(final_segment, arc_len_list[i])
                # save it
                t1_final_segments_tuples_list.append((final_segment,c))
    # t2
    if len(t2_ic_list) < 2:
        # centerline is alone
        t2_final_segments_tuples_list = [(t2_list[0],t2_ic_list[0][0][2])]
    else:
        # More than one centelrine: create mean segments
        # of which the points are ordered from closer to ostium (in terms of arc length)
        # to furthest.
        # Associate each point list (mean segment) to its segment cathegory index (an int)
        categories = []
        for a in t2_ic_list:
            for b in a:
                categories.append(b[2])
        t2_categories = set(categories)
        t2_final_segments_tuples_list = []
        for c in t2_categories:
            # get start and end indices
            list_of_centerline_segments = []
            for i_c, cent in enumerate(t2_list):
                for iiii, ic_tuple in enumerate(t2_ic_list[i_c]):
                    if ic_tuple[2] == c:
                        start_index = ic_tuple[0] if iiii == 0 else ic_tuple[0] + 10 # to avoid overlaps at junctions and smooth them out
                        segment = cent[start_index:ic_tuple[1]:]
                        list_of_centerline_segments.append(segment)
                        break
            # Now, the mean shift on paths algorithm should be applied
            # However, it is quite complicated and is not deemed necessary in this case
            # Solution:
            # The longest path (the segment with the most points) is selected and duplicated.
            # For each datapoint of this segment, its position and radius is averaged with the
            # closest point of each of the other segments.
            # Moreover, to have clean junctions, if the segments are not the ones containing the ostium,
            # all segments' first N points (10<N<20, ~ 3 mm) are discarded to allow a cleaner connection
            # The last step is to smooth it out a little and to sample the obtained curved once every 3 mm
            if len(list_of_centerline_segments) < 1:
                raise RuntimeError("list_of_centerline_segments should always have at least one member, instead it is empty.")
            elif len(list_of_centerline_segments) == 1:
                t2_final_segments_tuples_list.append((list_of_centerline_segments[0],c))
            else:
                # get longer segment, and use that segment indicisation to compute the mean path
                len_longest_segment = 0
                i_longest_segment = -1
                for i_seg, segment in enumerate(list_of_centerline_segments):
                    if segment.shape[0] > len_longest_segment:
                        len_longest_segment = segment.shape[0]
                        i_longest_segment = i_seg
                # now, apply the mean for each point of the longest segment with the closest points of the other segments
                final_segment = list_of_centerline_segments[i_longest_segment].copy()
                for i_p_fs in range(len_longest_segment):
                    points_to_mean_list = []
                    for j in range(len(list_of_centerline_segments)):
                        if j != i_longest_segment:
                            idxx, _ = util.getPointToCenterlinePointsMinDistance(
                                final_segment[i_p_fs],
                                list_of_centerline_segments[j]
                            )
                            points_to_mean_list.append(list_of_centerline_segments[j][idxx])
                        else:
                            points_to_mean_list.append(final_segment[i_p_fs])
                    final_segment[i_p_fs,:] = numpy.mean(points_to_mean_list, axis=0)
                # smooth it out
                for i_xyzr in range(final_segment.shape[1]):
                    fsa = numpy.insert(final_segment[:,i_xyzr], 0, final_segment[0,i_xyzr])
                    fsa = numpy.append(fsa, fsa[-1])
                    final_segment[:,i_xyzr] = numpy.convolve(fsa, [1/3, 1/3, 1/3])[2:-2]
                # now, the final segment is ready, just sample it once every 0.3 mm of arc length
                total_len = util.getCentelrineArcLength(final_segment)
                accumulated_len = 0.0
                arc_len_list = []
                while accumulated_len <= total_len:
                    arc_len_list.append(accumulated_len)
                    accumulated_len += 0.3
                resampled_final_segment = numpy.zeros((len(arc_len_list), final_segment.shape[1]))
                for i in range(len(arc_len_list)):
                    resampled_final_segment[i,:] = util.getCenterlinePointFromArcLength(final_segment, arc_len_list[i])
                # save it
                t2_final_segments_tuples_list.append((final_segment,c))

    # Now, just stitch the segments together following the connection lists in a graph,
    # and there you go, you should be done
    print("Debug plot...")
    ax = plt.subplot(111, projection="3d")
    for ccc in t1_final_segments_tuples_list:
        ax.plot(ccc[0][:,0], ccc[0][:,1], ccc[0][:,2],".-")
    plt.show()
    ax = plt.subplot(111, projection="3d")
    for ccc in t2_final_segments_tuples_list:
        ax.plot(ccc[0][:,0], ccc[0][:,1], ccc[0][:,2],".-")
    plt.show()




    g_dict = HCATNetwork.graph.BasicCenterlineGraph()
    g_dict["image_id"] = f"CAT08_dataset{IM_NUMBER:02d}"
    g_dict["are_left_right_disjointed"] = True
    graph_bcg = networkx.Graph(**g_dict)
    graph_node_index_counter = 0
    # t1 - RCA for CAT08
    if len(t1_final_segments_tuples_list) == 1:
        # just one segment in the whole arterial tree
        print("alone, think about it later cause it is easier")
    else:
        # Multiple 
        print("t1 is never alone in the pure form of the cat08 dataset")


    # t2 - LCA for CAT08
    if len(t2_final_segments_tuples_list) == 1:
        # just one segment in the whole arterial tree
        print("alone, think about it later cause it is easier")
    else:
        ############
        ###########
        ###########
        # populate graph with intra-connected segments
        segment_start_end_nodes_dict = {}
        for t2_ in t2_final_segments_tuples_list:
            segment_start_end_nodes_dict.update(
                util.connectGraphIntersegment(
                    graph=graph_bcg,
                    sgm_tuple=t2_,
                    tree=HCATNetwork.node.ArteryPointTree.LEFT,
                    connections=centerlines_cat_conn
                )
            )
        # connect segments
        for conn_tuple in centerlines_cat_conn:
            if conn_tuple[0] == conn_tuple[1]:
                continue
            # take last node of first segment in conn_tuple and connect it to first node of last segment in conn_tuple
            edge_features = HCATNetwork.edge.BasicEdge()
            node1 = segment_start_end_nodes_dict[conn_tuple[0]]["e"]
            node2 = segment_start_end_nodes_dict[conn_tuple[1]]["s"]
            node1_v = HCATNetwork.node.SimpleCenterlineNode(**graph_bcg.nodes[node1]).getVertexNumpyArray()
            node2_v = HCATNetwork.node.SimpleCenterlineNode(**graph_bcg.nodes[node2]).getVertexNumpyArray()
            edge_features["euclidean_distance"] = float(numpy.linalg.norm(node1_v-node2_v))
            edge_features.updateWeightFromEuclideanDistance()
            graph_bcg.add_edge(node1, node2, **edge_features)

        # Plot  ###################### keep only for debugging - use at the end to plot graph, try in 3d ###############
        if 1:
            color_list__ = []
            pos_dict__ = {}
            for n in graph_bcg.nodes:
                n_scn = HCATNetwork.node.SimpleCenterlineNode(**(graph_bcg.nodes[n]))
                pos_dict__.update(**{n: n_scn.getVertexList()[:2]})
                color_list__.append(n_scn["topology_class"].value)
            networkx.draw(
                graph_bcg,
                **{"with_labels": False, 
                    "node_color": color_list__, 
                    "node_size": 50,
                    "pos": pos_dict__,
                    }
            )
            plt.show()

        quit()

        util.connectGraph(
            graph_bcg,
            t2_final_segments_tuples_list,
            centerlines_cat_conn,
            HCATNetwork.node.ArteryPointTree.LEFT
        )


        ###########################
        ###########################
        #### kinda works, but do not understand it anymore
        ############################
        ############################
        # Multiple 
        for points_list, category in t2_final_segments_tuples_list:
            # deal with segment points
            for i, p in enumerate(points_list):
                # node
                n_dict = HCATNetwork.node.getNodeDictFromKeyList(HCATNetwork.node.SimpleCenterlineNode_KeysList)
                if category == min(t2_categories) and i == 0:
                    n_dict["class"] = "o"
                elif i == len(points_list) - 1:
                    has_downstream_connecion = False
                    for conn in centerlines_cat_conn:
                        if categories == conn[0]:
                            has_downstream_connecion = True
                            break
                    n_dict["class"] = "i" if has_downstream_connecion else "e"
                else:
                    n_dict["class"] = "s"
                n_dict["x"], n_dict["y"], n_dict["z"], n_dict["r"] = p[0], p[1], p[2], p[3]
                n_dict["t"] = 0.0
                n_dict["tree"] = "l"
                prev_graph_node_index_counter = graph_node_index_counter
                g.add_node(str(category*1000 + graph_node_index_counter), **n_dict)
                #########################################################################################################################################
                #########################################################################################################################################
                #########################################################################################################################################
                # SPOSTA TUTTO IN UNA FUNZIONE CHE CONNETTE TUTTI I PUNTI DI UN SINGOLO SEGMENTO
                # UNA VOLTA FATTO QUESTO, CONNETTERE I SEGMENTI FRA LORO DEVE ESSERE PIù FACILE
                # DEVI FARLO IN MODO AUTOMATICO, NON AD HOC, PERCHE SE IN FUTURO AVRAI ALTRI DATI
                # SU CUI APPLICARE QUESTO CONCETTO, ALMENO SARà GIà PRONTO
                # prova qualcosa di ricorsivo....
                #########################################################################################################################################
                #########################################################################################################################################
                #########################################################################################################################################
                # upstream edge
                if n_dict["class"] != "o":
                    e_dict = HCATNetwork.edge.getEdgeDictFromKeyList(HCATNetwork.edge.BasicEdge_KeysList)
                    e_dict["signed_distance"] = - numpy.linalg.norm(
                        HCATNetwork.node.getNumpyVertexFromSimpleCenterlineNode(n_dict) - \
                        HCATNetwork.node.getNumpyVertexFromSimpleCenterlineNode(g.nodes[str(prev_graph_node_index_counter)]),
                        axis=0
                    )
                    e_dict["weight"] = e_dict["signed_distance"]
                # downstream edge cannot be done 
                # do not know if it can be done

            # deal with smooth connections
            
            


            
                
                
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
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




   
        






