import numpy
import matplotlib.pyplot as plt
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader 

import networkx
import hcatnetwork


def read_vtk_centerlines(filename) -> list[numpy.ndarray]:
    """Read the VTK files in which the centerlines are stored.

    Parameters
    ----------
    filename : str
        Path to the VTK file.

    Returns
    -------
    list[numpy.ndarray]
        List of numpy arrays, each array is a segment (ostium to endpoint) of the centerline,
        each point appearing sequentially in the segment.
        Each point is a 4D array, containing the x,y,z coordinates and the radius of the point.
        Points in the same segment section overlap perfectly.
    """
    # Read all the data from the file
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    # Extract data
    polydata = reader.GetOutput()
    point_data = polydata.GetPointData() # from this we get the radius
    cell_data = polydata.GetCellData()   # from this we get the nodes positions
    print(point_data); print(); print(cell_data); quit()
    # - extract the radius
    data_array = point_data.GetArray("MaximumInscribedSphereRadius")
    MaximumInscribedSphereRadius_list = [data_array.GetValue(i) for i in range(data_array.GetSize())]
    # -- # debug - keep it, might be useful in future
    # -- # point_array_names = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
    # -- # cell_array_names = [cell_data.GetArrayName(i) for i in range(cell_data.GetNumberOfArrays())]
    # - extract the nodes positions
    #   from here on, code is like magic, no idea how it works
    #   at the end, you get a list of numpy arrays, each array is a segment of the centerline
    lines_celldata = polydata.GetLines()
    offset_array = lines_celldata.GetOffsetsArray()
    cells_idx_list = [offset_array.GetValue(i) for i in range(offset_array.GetSize())]
    connectivity_array = lines_celldata.GetConnectivityArray()
    connectivity_list = [connectivity_array.GetValue(i) for i in range(connectivity_array.GetSize())]
    points = polydata.GetPoints()
    # get the final list of segments
    points_nparray_container = []
    for s, e_excl in zip(cells_idx_list[:-1], cells_idx_list[1:]):
        segment = []
        for i in connectivity_list[s:e_excl]:
            # Data with coordinates + radius     
            point = numpy.array(points.GetPoint(i))
            radius = MaximumInscribedSphereRadius_list[i]
            point_with_radius = numpy.append(point, radius)
            segment.append(point_with_radius)
        points_nparray_container.append(numpy.array(segment))
    return points_nparray_container

def plod_segments_container(segments_container):
    surf=2**6
    zorder=1
    ax = plt.subplot(111)
    for line in segments_container:
        ax.scatter(line[:,0], line[:,1], s=surf, zorder=zorder, label=f"{zorder-1}")
        # segments_container[a][b,c]
        # a: indice del singolo tracciato di centerline
        # b: indice del singolo punto
        # c: 0->x, 1->y, 2->z
        surf /= 1.5
        zorder+=1
    ax.set_xlabel("Right -> Left (mm)")
    ax.set_ylabel("Anterior -> Posterior (mm)")
    ax.legend()
    plt.show()


def process_points_to_graph(points_nparray_container, graph: hcatnetwork.graph.SimpleCenterlineGraph) -> None:
    """
    Processes points and their corresponding attributes to create nodes and edges in a graph
    the updated graph with added nodes and edges.
    """
    node_id_ = -1 
    nodes_string_to_node_id_map_ = []
    already_added_nodes_positions_list_ = []
    current_tree = hcatnetwork.node.ArteryNodeSide.RIGHT
    # cycle through all the segments
    for i_arr_, arr_ in enumerate(points_nparray_container):
        # Skip segments with less than 10 points
        if len(arr_) < 10:
            continue
        # cycle through all the points in the segment
        for i_p_, p_ in enumerate(arr_):
            # Check if the point is already in the graph
            # To do so, we create a string with the coordinates of the point,
            # and check if it is in the list of already added points.
            # This is a peculiarity of the ASOCA dataset, where the same point
            # is repeated in different segments exactly at the same coordinates.
            p_string = f"{p_[0]:.3f} {p_[1]:.3f} {p_[2]:.3f} {p_[3]:.3f}"
            if p_string in already_added_nodes_positions_list_:
                continue
            else:
                already_added_nodes_positions_list_.append(p_string)
                node_id_ += 1
                nodes_string_to_node_id_map_.append(str(node_id_))
                # Determine the topology
                if i_p_ == 0:
                    # first point of the segment is always an ostium
                    topology = hcatnetwork.node.ArteryNodeTopology.OSTIUM
                elif i_p_ == len(arr_)-1:
                    # last point of the segment is always an endpoint
                    topology = hcatnetwork.node.ArteryNodeTopology.ENDPOINT
                else:
                    # in-between points are always initialized as segments
                    # outside this loop, when the graph is full, we will check if
                    # they are intersections.
                    topology = hcatnetwork.node.ArteryNodeTopology.SEGMENT
                # Determine if it is the left or right tree
                # We assume the first tree is the left one. This is not always the case,
                # but we will check and swap the trees later based on empirical rules
                # appliable only to this dataset.
                # Here we check if the current point is the first point of the first segment,
                # and if it is, we save the string of the point to compare it later with the
                # first point of the other ostium. If they are different, we swap trees and,
                # from there on, consider all points as left.
                # This has errors, because sometimes the segments are not ordered, but it is
                # the best we can do now. In any case, the correct tree is corrected later,
                # by checking the tree of the ostium to which it is connected. 
                if node_id_ == 0:
                    p_string_first_ostium = p_string
                elif i_p_ == 0 and p_string != p_string_first_ostium: 
                    current_tree = hcatnetwork.node.ArteryNodeSide.LEFT
                # Create and set attributes for the node
                node_attributes = hcatnetwork.node.SimpleCenterlineNodeAttributes()
                node_attributes.set_vertex_and_radius(p_)
                node_attributes["t"] = 0.0
                node_attributes["topology"] = topology
                node_attributes["side"] = current_tree
                graph.add_node(str(node_id_), node_attributes)
                # Create edges and adds them
                if node_id_ == 0:
                    pass
                elif i_p_ == 0 and p_string != p_string_first_ostium: 
                    pass
                else:
                    # find the node in the graph to which the newly created node is connected
                    p_string_node_before_ = f"{arr_[i_p_-1][0]:.3f} {arr_[i_p_-1][1]:.3f} {arr_[i_p_-1][2]:.3f} {arr_[i_p_-1][3]:.3f}"
                    loc_node_before_ = already_added_nodes_positions_list_.index(p_string_node_before_)
                    id_target_node_ = nodes_string_to_node_id_map_[loc_node_before_]
                    # Make edge attributes
                    edge_features = hcatnetwork.edge.SimpleCenterlineEdgeAttributes()
                    position_of_target_node_ = hcatnetwork.node.SimpleCenterlineNodeAttributes(**graph.nodes[id_target_node_]).get_vertex_numpy_array()
                    edge_features["euclidean_distance"] = float(numpy.linalg.norm(position_of_target_node_ - p_[:3]))
                    edge_features.update_weight_from_euclidean_distance()
                    graph.add_edge(str(node_id_), str(id_target_node_), edge_features)
    # Graph has been populated
    # It is not complete, because the ASOCA dataset is not always ordered and consistent.
    # Further post-processing is needed to complete the graph.
    return graph


def check_and_swap_ostia_if_needed(graph: hcatnetwork.graph.SimpleCenterlineGraph, left_ostium_node_id, right_ostium_node_id):
    # Determine reachable nodes from each ostium
    reachable_from_left_ostium = [k for k in networkx.single_source_dijkstra_path_length(graph, left_ostium_node_id)]
    reachable_from_right_ostium = [k for k in networkx.single_source_dijkstra_path_length(graph, right_ostium_node_id)]
    # Calculate mean x-coordinate for nodes reachable from each ostium
    mean_x_left = sum(graph.nodes[node]['x'] for node in reachable_from_left_ostium) / len(reachable_from_left_ostium)
    mean_x_right = sum(graph.nodes[node]['x'] for node in reachable_from_right_ostium) / len(reachable_from_right_ostium)
    # Check if the mean x-coordinate of the left ostium is lower than that of the right ostium
    mean_x_check = mean_x_left < mean_x_right
    # Swap if the check is verified
    if mean_x_check:
        # Swap the labels of the two ostia
        graph.nodes[left_ostium_node_id]["side"] = hcatnetwork.node.ArteryNodeSide.RIGHT
        graph.nodes[right_ostium_node_id]["side"] = hcatnetwork.node.ArteryNodeSide.LEFT
        return right_ostium_node_id, left_ostium_node_id
