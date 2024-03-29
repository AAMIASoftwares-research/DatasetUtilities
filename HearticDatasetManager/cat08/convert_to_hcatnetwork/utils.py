import os
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import networkx
import hcatnetwork


def importCenterlineList(cat08_im_folder_path:str):
    v_list = []
    for i_ in range(4):
        v_file_path = os.path.join(cat08_im_folder_path, f"vessel{i_}", "reference.txt")
        v_list.append(numpy.loadtxt(v_file_path, delimiter=" ", usecols=range(4)))
    return v_list


def getCurveArcLength(curve: numpy.ndarray, startIdx: int|None = None, endIdx: int|None = None) -> float:
        """ Function to get the length of a curve, defined as a 2D/3D point sequence
        starting from the ostium and ending at its endpoint.
        :param curve: NxM numpy.ndarray with M>=2
        :param startIdx: index of the starting point of the arc length measure. If None, 0
        :param endIdx: index of the end point of the arc length measure. If None: end
        """
        if curve.shape[1] < 2:
            raise RuntimeWarning(f"Shape of the passed numpy.ndarray is not acceptable, must be NxM with M>=3, instead it is: {curve.shape[0]}")
        n_cols = int(min(curve.shape[1], 3))
        if startIdx is None:
            startIdx = 0
        if endIdx is None:
            endIdx = curve.shape[0]
        if endIdx <= startIdx + 1:
            raise RuntimeError(f"Invalid startIdx={startIdx} and endIdx={endIdx} with curve of shape={curve.shape}: too close!")
        return float(numpy.sum(
            numpy.linalg.norm(
                curve[startIdx+1:endIdx,:3]-curve[startIdx:endIdx-1,:3],
                axis = 1)
            )
        )


def getCentelrineArcLength(centerline: numpy.ndarray, startIdx: int|None = None, endIdx: int|None = None) -> float:
        """ Function to get the length of a centerline, defined as a 3D point sequence
        starting from the ostium and ending at its endpoint.
        :param centerline: NxM numpy.ndarray with M>=3
        :param startIdx: index of the starting point of the arc length measure. If None, 0
        :param endIdx: index of the end point of the arc length measure. If None: end
        """
        if centerline.shape[1] < 3:
            raise RuntimeWarning(f"Shape of the passed numpy.ndarray is not acceptable, must be NxM with M>=3, instead it is: {centerline.shape[0]}")
        if startIdx is None:
            startIdx = 0
        if endIdx is None:
            endIdx = centerline.shape[0]
        if endIdx <= startIdx + 1:
            raise RuntimeError(f"Invalid startIdx={startIdx} and endIdx={endIdx} with centerline of shape={centerline.shape}: too close!")
        return float(numpy.sum(
            numpy.linalg.norm(
                centerline[startIdx+1:endIdx,:3]-centerline[startIdx:endIdx-1,:3],
                axis = 1)
            )
        )

def getCurvePointFromArcLength(reference_curve: numpy.ndarray, arc_length:float):
    """Linear interpolation"""
    total_length = getCurveArcLength(reference_curve) # controls on curve data shape happens inside there
    if arc_length == 0:
        return reference_curve[0]
    if arc_length == total_length:
        return reference_curve[-1]
    if arc_length > total_length or arc_length < 0:
        return None
    n_cols = int(min(reference_curve.shape[1], 3))
    idx_before = 0
    len_before = 0.0
    last_len_before = 0.0
    while len_before < arc_length:
        last_len_before = numpy.linalg.norm(reference_curve[idx_before+1,:n_cols]-reference_curve[idx_before,:n_cols])
        len_before += last_len_before
        idx_before += 1
    # Reached the point for which the s is between before and after
    exceeded_len = arc_length - (len_before-last_len_before)
    covered_percentage = exceeded_len/last_len_before if last_len_before != 0 else 0.0
    out_point =  covered_percentage*reference_curve[idx_before,:] + (1-covered_percentage)*reference_curve[idx_before-1,:]
    return out_point

def getCenterlinePointFromArcLength(reference_centerline: numpy.ndarray, arc_length:float):
    """Linear interpolation"""
    total_length = getCentelrineArcLength(reference_centerline)
    if arc_length == 0:
        return reference_centerline[0]
    if arc_length == total_length:
        return reference_centerline[-1]
    if arc_length > total_length or arc_length < 0:
        return None
    idx_before = 0
    len_before = 0.0
    last_len_before = 0.0
    while len_before < arc_length:
        last_len_before = numpy.linalg.norm(reference_centerline[idx_before+1,:3]-reference_centerline[idx_before,:3])
        len_before += last_len_before
        idx_before += 1
    # Reached the point for which the s is between before and after
    exceeded_len = arc_length - (len_before-last_len_before)
    covered_percentage = exceeded_len/last_len_before if last_len_before != 0 else 0.0
    out_point =  covered_percentage*reference_centerline[idx_before,:] + (1-covered_percentage)*reference_centerline[idx_before-1,:]
    return out_point

if __name__ == "__main__":
    np = int(1800)
    p = numpy.zeros((np,4))
    p[:,0] = numpy.linspace(0.,2.,np)
    arclen_ = getCentelrineArcLength(p); print(arclen_)
    plt.plot(p[:,0], p[:,1]+0.05, ".-")
    t_vec = numpy.arange(0.,2.0,0.03)
    p = [getCenterlinePointFromArcLength(p, t_) for t_ in t_vec]
    p = numpy.array(p)
    plt.plot(p[:,0], p[:,1], ".-")
    plt.show()

def divideArterialTreesFromOriginalCenterlines(centerlines_list: list) -> tuple:
    ## preprocess the centerlines: necessary because some of them have the ostium very
    # different from one another in the common segments
    # 1 - find the two arterial trees right away: we use just the first points of the centerlines
    ostia = numpy.array([l[0,:3] for l in centerlines_list])
    min_distance_between_two_trees = 20 #mm
    db = DBSCAN(eps=min_distance_between_two_trees, min_samples=1).fit(ostia)
    tree1_idxs = numpy.argwhere(db.labels_==0).flatten()
    tree1_list = [centerlines_list[i] for i in tree1_idxs]
    tree1_ostia = ostia[tree1_idxs,:]
    tree2_idxs = numpy.argwhere(db.labels_==1).flatten()
    tree2_list = [centerlines_list[i] for i in tree2_idxs]
    tree2_ostia = ostia[tree2_idxs,:]
    return (tree1_idxs, tree1_list, tree2_idxs, tree2_list)



def getPointToCenterlinePointsMinDistance(p:numpy.ndarray, centerline: numpy.ndarray) -> tuple:
    """Description later..."""
    d = numpy.linalg.norm(p[:3] - centerline[:,:3], axis= 1)
    idx_min = numpy.argmin(d).flatten()[0]
    dist_min = d[idx_min]
    return (idx_min, dist_min)

def getCentelrinesFurthestConnectionIndex(c_i: numpy.ndarray, c_j: numpy.ndarray, thresh_radius_multiplier=0.4):
    """
    """
    conn_index = int(-1)
    # work in c_i
    d_last = 1e9
    for i_ in range(len(c_i)):
        # get dist and compute logic
        _, dmin = getPointToCenterlinePointsMinDistance(
            p=c_i[i_],
            centerline=c_j
        )
        thresh = 0.5 * c_i[i_][3]
        if dmin < thresh:
            d_last = dmin
            conn_index = i_
    # go back a couple of indexes to have smoother intersections
    if conn_index > 30:
        for i_ in range(conn_index, conn_index - 30, -1):
            _, dmin = getPointToCenterlinePointsMinDistance(
                p=c_i[i_],
                centerline=c_j
            )
            if dmin < 0.1*c_i[i_][3]:
                conn_index = i_ - 3
                break
            if dmin < d_last:
                d_last = dmin
                conn_index = i_
    return conn_index


#############################
# MAIN ROUTINE TO FUSE
# COMMON SEGMENT CENTERLINES
# TOGETHER
#############################

def routineMergeCenterlineCommonSegments(t_list, t_ic_list) -> list[tuple[numpy.ndarray,int]]:
    if len(t_ic_list) < 2:
        # centerline is alone
        t_final_segments_tuples_list = [(t_list[0],t_ic_list[0][0][2])]
    else:
        # More than one centelrine: create mean segments
        # of which the points are ordered from closer to ostium (in terms of arc length)
        # to furthest.
        # Associate each point list (mean segment) to its segment cathegory index (an int)
        categories = []
        for a in t_ic_list:
            for b in a:
                categories.append(b[2])
        t2_categories = set(categories)
        t_final_segments_tuples_list = []
        for c in t2_categories:
            # get start and end indices
            list_of_centerline_segments = []
            for i_c, cent in enumerate(t_list):
                for iiii, ic_tuple in enumerate(t_ic_list[i_c]):
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
            # The last step is to smooth it out a little
            # Resampling of the curve is not performed, as it can be performed later
            if len(list_of_centerline_segments) < 1:
                raise RuntimeError("list_of_centerline_segments should always have at least one member, instead it is empty.")
            elif len(list_of_centerline_segments) == 1:
                # now, the final segment is ready
                # out
                t_final_segments_tuples_list.append((list_of_centerline_segments[0],c))
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
                            idxx, _ = getPointToCenterlinePointsMinDistance(
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
                # now, the final segment is ready. Resampling is postponed to later
                # save it
                t_final_segments_tuples_list.append((final_segment,c))
    return t_final_segments_tuples_list




#####################
# BUILDING THE GRAPHS
#####################

def cubiBezier(
        P0: numpy.ndarray, 
        P1: numpy.ndarray, 
        P2: numpy.ndarray, 
        P3: numpy.ndarray, 
        t: float) -> numpy.ndarray:
    res = P0 * (-t**3+3*t**2-3*t+1) + \
          P1 * (3*t**3-6*t**2+3*t) + \
          P2 * (-3*t**3+3*t**2)  + \
          P3 * (t**3)
    return res

def test_cubicBezierRespaced():
    p0 = numpy.array([0,0])
    p1 = numpy.array([1,0])
    p2 = numpy.array([4,1.7])
    p3 = numpy.array([7,1.5])
    for i_, p in enumerate([p0,p1,p2,p3]):
        plt.scatter(p[0], p[1], c="green" if i_<1.5 else "blue")
    t = numpy.array([0,1,2,3])
    plt.plot((p1[0]-p0[0])*t+p0[0], (p1[1]-p0[1])*t+p0[1],"--", color="grey")
    plt.plot((p2[0]-p3[0])*t+p3[0], (p2[1]-p3[1])*t+p3[1],"--", color="grey")
    d = numpy.linalg.norm(p2 - p1)
    v12 = (p2 - p1)/numpy.linalg.norm(p2 - p1)
    pc1 = v12*0.333*d + p1
    v32 = (p2 - p3)/numpy.linalg.norm(p2 - p3)
    pc2 = v32*0.333*d + p2
    for i_, p in enumerate([pc1, pc2]):
        plt.scatter(p[0], p[1], c="orange")
    pt = numpy.array([cubiBezier(p1,pc1,pc2,p2,t_) for t_ in numpy.linspace(0,1,300)])
    # Respacing is useful so do not delete this code, however it is not needed for the+
    # single connection segment, rather the connection segment will be pre-pended
    # to the segment, and only then everything will be respaced
    #pt_arclen = getCurveArcLength(pt)
    #spacing = 0.2 # mm
    #t_vec = numpy.arange(0,pt_arclen,spacing, dtype="float")
    #pt_respaced = numpy.array([getCurvePointFromArcLength(pt, t_) for t_ in t_vec])
    #last = None if abs(numpy.linalg.norm(pt_respaced[-1] - p2) - spacing) > 0.5*spacing else -1
    #plt.plot(pt_respaced[1:last,0], pt_respaced[1:last,1], "r.-")
    plt.plot(pt[:,0], pt[:,1], "r.-")
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    test_cubicBezierRespaced()


def connectGraphIntersegment(
        graph: hcatnetwork.graph.SimpleCenterlineGraph,
        sgm_tuple: tuple[list, int],
        tree: hcatnetwork.node.ArteryNodeSide,
        connections):
    start_node_id_int = graph.number_of_nodes()
    p_list, seg_id = sgm_tuple
    connections = numpy.array(connections)
    is_alone = ((connections[:,0] == seg_id) == (connections[:,1] == seg_id)).all()
    # Nodes
    node_idx_int = start_node_id_int
    for i_, p in enumerate(p_list):
        node_attributes = hcatnetwork.node.SimpleCenterlineNodeAttributes()
        node_attributes.set_vertex_and_radius(p)
        node_attributes["t"] = 0.0
        if i_ == 0 and not ((seg_id == connections[:,1]).any() and not is_alone):
            topology = hcatnetwork.node.ArteryNodeTopology.OSTIUM
        elif i_ == len(p_list)-1:
            if not (seg_id == connections[:,0]).any() or is_alone:
                topology = hcatnetwork.node.ArteryNodeTopology.ENDPOINT
            else:
                topology = hcatnetwork.node.ArteryNodeTopology.INTERSECTION
        else:
            topology = hcatnetwork.node.ArteryNodeTopology.SEGMENT
        node_attributes["topology"] = topology
        node_attributes["side"] = tree
        graph.add_node(str(node_idx_int), **node_attributes)
        end_node_id_int = node_idx_int
        node_idx_int += 1
    # Edges
    for i_ in range(start_node_id_int, end_node_id_int):
        edge_features = hcatnetwork.edge.SimpleCenterlineEdgeAttributes()
        node1_v = hcatnetwork.node.SimpleCenterlineNodeAttributes(**graph.nodes[str(i_)]).get_vertex_numpy_array()
        node2_v = hcatnetwork.node.SimpleCenterlineNodeAttributes(**graph.nodes[str(i_+1)]).get_vertex_numpy_array()
        edge_features["euclidean_distance"] = float(numpy.linalg.norm(node1_v-node2_v))
        edge_features.update_weight_from_euclidean_distance()
        graph.add_edge(str(i_), str(i_+1), **edge_features)
    # Plot  ###################### keep only for debugging - use at the end to plot graph, try in 3d ###############
    if 0:
        color_list__ = []
        pos_dict__ = {}
        for n in graph.nodes:
            n_scn = hcatnetwork.node.SimpleCenterlineNodeAttributes(**(graph.nodes[n]))
            pos_dict__.update(**{n: n_scn.get_vertex_list()[:2]})
            color_list__.append(n_scn["topology"].value)
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph, backend="networkx")
    # Out
    # note that the graph object gets modified in place, no need to return it
    segment_first_node_id = str(start_node_id_int)
    segment_last_node_id = str(end_node_id_int)
    return {seg_id: {"s":segment_first_node_id, "e":segment_last_node_id}}


def buildAndConnectGraph(
        graph,
        t_final_segments_tuples_list: list[tuple[numpy.ndarray,float]],
        centerlines_cat_conn: list[tuple[int,int]],
        tree_class: hcatnetwork.node.ArteryNodeSide,
        graph_nodes_target_spacing_mm: float) -> None:
    """
    This utility method creates the graph starting from the segments
    tuples.
    it resamples all curves so that the points are "graph_nodes_target_spacing_mm" distant from each other.
    It takes as input the target graph, so it has no output, the graph is updated and modified in place.
    """ 
    if len(t_final_segments_tuples_list) == 1:
        # just one segment in the whole arterial tree
        # resample it
        arclen_ = getCentelrineArcLength(t_final_segments_tuples_list[0][0])
        t_vec = numpy.arange(0,arclen_, graph_nodes_target_spacing_mm, dtype="float")
        segment_new = numpy.array(
            [getCenterlinePointFromArcLength(t_final_segments_tuples_list[0][0], t_) for t_ in t_vec]
        )
        t_final_segments_tuples_list[0] = (segment_new, t_final_segments_tuples_list[0][1])
        # populate graph with intra-connected segments
        connectGraphIntersegment(
            graph=graph,
            sgm_tuple=t_final_segments_tuples_list[0],
            tree=tree_class,
            connections=centerlines_cat_conn
        )
    else:
        # add cubic bezier extension to middle segments towards previous segments
        cent_conn_ = numpy.array(centerlines_cat_conn)
        for [in_, out_] in cent_conn_:
            if in_ == out_:
                continue
            # find in and out segments in tuple list (with dict was much easier... noted for the future me...)
            tpl_index_in = None
            tpl_index_out = None
            for i_, tpl_ in enumerate(t_final_segments_tuples_list):
                if tpl_[1] == in_:
                    tpl_index_in = i_
                if tpl_[1] == out_:
                    tpl_index_out = i_
            # get start and end control points
            p0 = t_final_segments_tuples_list[tpl_index_in][0][-1]
            p3 = t_final_segments_tuples_list[tpl_index_out][0][0]
            # get two middle control points
            d = numpy.linalg.norm(p0 - p3)
            v0 = p3 - p0
            v0 /= numpy.linalg.norm(v0)
            p1 = p0 + v0*0.333*d
            v1 = numpy.mean([t_final_segments_tuples_list[tpl_index_out][0][0] - t_final_segments_tuples_list[tpl_index_out][0][j_] for j_ in [1,2,3]], axis=0)
            v1 /= numpy.linalg.norm(v1)
            p2 = p3 + v1*0.333*d
            # make bezier
            n_linspace = max(20, int(1.5*d/graph_nodes_target_spacing_mm))
            bez = numpy.array([cubiBezier(p0,p1,p2,p3,t_) for t_ in numpy.linspace(0,1,n_linspace)])
            # attach it to out_ segment, in the front
            bez_plus_out_segment = numpy.zeros((bez.shape[0]-1+t_final_segments_tuples_list[tpl_index_out][0].shape[0],bez.shape[1]))
            bez_plus_out_segment[:bez.shape[0]-1,:] = bez[:-1,:]
            bez_plus_out_segment[bez.shape[0]-1:,:] = t_final_segments_tuples_list[tpl_index_out][0]
            # resample it
            arclen_ = getCentelrineArcLength(bez_plus_out_segment)
            t_vec = numpy.arange(0,arclen_, graph_nodes_target_spacing_mm, dtype="float")
            bez_plus_out_segment_new = numpy.array([getCenterlinePointFromArcLength(bez_plus_out_segment, t_) for t_ in t_vec])
            bez_plus_out_segment = bez_plus_out_segment_new[1:] # the first point would coincide with the last point of the previous segment, so we do not include it in the recieving segment
            if 0:
                # PLOT for DEBUGGING and EXPLAINABILITY
                # useful plot to explain how the procedure works, visually
                # It is better to show the bezier not resampled, or both (before and after resampling)
                plt.plot(bez[:,0], bez[:,1],".-")
                plt.title("Bezier only")
                plt.show()
                # bezier plus segment
                plt.plot(bez_plus_out_segment[:,0], bez_plus_out_segment[:,1],".-")
                plt.plot(bez[:,0], bez[:,1],"o--", color="purple")
                p_ = numpy.array([p0,p1,p2,p3])
                plt.scatter(p_[:,0], p_[:,1], c="orange")
                plt.plot([p0[0], p0[0]+v0[0]], [p0[1], p0[1]+v0[1]], color="grey")
                plt.plot([p3[0], p3[0]+v1[0]], [p3[1], p3[1]+v1[1]], color="grey")
                plt.show()
            # assign extended curve to the results storage, ready for the next step
            t_final_segments_tuples_list[tpl_index_out] = (bez_plus_out_segment, t_final_segments_tuples_list[tpl_index_out][1])
        # populate graph with intra-connected segments
        segment_start_end_nodes_dict = {}
        for tpl_ in t_final_segments_tuples_list:
            segment_start_end_nodes_dict.update(
                connectGraphIntersegment(
                    graph=graph,
                    sgm_tuple=tpl_,
                    tree=tree_class,
                    connections=centerlines_cat_conn
                )
            )
        # connect segments
        for conn_tuple in centerlines_cat_conn:
            if conn_tuple[0] == conn_tuple[1]:
                continue
            # take last node of first segment in conn_tuple and connect it to first node of last segment in conn_tuple
            edge_features = hcatnetwork.edge.SimpleCenterlineEdgeAttributes()
            node1 = segment_start_end_nodes_dict[conn_tuple[0]]["e"]
            node2 = segment_start_end_nodes_dict[conn_tuple[1]]["s"]
            node1_v = hcatnetwork.node.SimpleCenterlineNodeAttributes(**graph.nodes[node1]).get_vertex_numpy_array()
            node2_v = hcatnetwork.node.SimpleCenterlineNodeAttributes(**graph.nodes[node2]).get_vertex_numpy_array()
            edge_features["euclidean_distance"] = float(numpy.linalg.norm(node1_v-node2_v))
            edge_features.update_weight_from_euclidean_distance()
            graph.add_edge(node1, node2, **edge_features)



###############
# VISUALISATION
###############

colors=["red", "blue", "orange", "green", "black", "pink", "yellow"]

def plotCenterlineList(centerlines_list: list):
    ax1 = plt.subplot(221)
    n = len(centerlines_list)
    for i_ in range(n):
        ax1.plot(centerlines_list[i_][:,0],centerlines_list[i_][:,1],
                 ".-", label=str(i_), color=colors[i_])
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.legend()
    #
    ax2 = plt.subplot(222, projection="3d")
    for i_ in range(n):
        ax2.plot(centerlines_list[i_][:,0],centerlines_list[i_][:,1], centerlines_list[i_][:,2],
                 color="black", linewidth=0.7)
        ax2.scatter(centerlines_list[i_][:,0],centerlines_list[i_][:,1], centerlines_list[i_][:,2],
                 s=centerlines_list[i_][:,3]**2, label=str(i_), c=colors[i_])
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_zlabel("z [mm]")
    ax2.legend()
    #
    ax3 = plt.subplot(212)
    for i_ in range(n):
        d = numpy.linalg.norm(centerlines_list[i_][0,:3] - centerlines_list[i_][:,:3], axis=1)
        ax3.plot(range(d.shape[0]), d, label=str(i_))
    ax3.set_xlabel("idx")
    ax3.set_ylabel("distance from first point")
    ax3.legend()
    plt.show()

def plotCenterlineListWithIndexes(centerlines_list: list):
    n = len(centerlines_list)
    # x, y
    ax1 = plt.subplot(121)
    for i_ in range(n):
        ax1.plot(centerlines_list[i_][:,0], centerlines_list[i_][:,1], ".-", label=str(i_)+f" : n_points={centerlines_list[i_].shape[0]}", color=colors[i_], alpha=0.4)
        for i_p in range(0, centerlines_list[i_].shape[0], 10):
            ax1.text(centerlines_list[i_][i_p,0], centerlines_list[i_][i_p,1], str(i_p), color=colors[i_])
        ax1.text(centerlines_list[i_][-1,0], centerlines_list[i_][-1,1], str(centerlines_list[i_].shape[0]-1), color=colors[i_])
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.legend()
    ax1.axis("equal")
    # x, z
    ax2 = plt.subplot(222)
    for i_ in range(n):
        ax2.plot(centerlines_list[i_][:,0], centerlines_list[i_][:,2], ".-", label=str(i_), color=colors[i_])
        for i_p in range(0, centerlines_list[i_].shape[0], 20):
            ax2.text(centerlines_list[i_][i_p,0], centerlines_list[i_][i_p,2], str(i_p), color=colors[i_])
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("z [mm]")
    ax2.axis("equal")
    # y, z
    ax3 = plt.subplot(224)
    for i_ in range(n):
        ax3.plot(centerlines_list[i_][:,1], centerlines_list[i_][:,2], ".-", label=str(i_), color=colors[i_])
    ax3.set_xlabel("y [mm]")
    ax3.set_ylabel("z [mm]")
    ax3.axis("equal")
    # out
    plt.show()
    del ax1, ax2, ax3
    # 3d
    ax0 = plt.subplot(111, projection="3d")
    for i_ in range(n):
        ax0.plot(centerlines_list[i_][:,0], centerlines_list[i_][:,1], centerlines_list[i_][:,2], ".-", label=str(i_)+f" : n_points={centerlines_list[i_].shape[0]}", color=colors[i_], alpha=0.4)
        for i_p in range(0, centerlines_list[i_].shape[0], 10):
            ax0.text(centerlines_list[i_][i_p,0], centerlines_list[i_][i_p,1], centerlines_list[i_][i_p,2], str(i_p), color=colors[i_])
        ax0.text(centerlines_list[i_][-1,0], centerlines_list[i_][-1,1], centerlines_list[i_][-1,2],str(centerlines_list[i_].shape[0]-1), color=colors[i_])
    ax0.set_xlabel("x [mm]")
    ax0.set_ylabel("y [mm]")
    ax0.set_zlabel("z [mm]")
    ax0.legend()
    ax0.axis("equal")
    # out
    plt.show()

def plot_curve_ptop_distances(curve):
    pass