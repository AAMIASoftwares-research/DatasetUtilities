# This program zips, or sews together, the centerline graph near the intersections of the graph.
# This is necessary because, in the ASOCA dataset, the intersections are labelled far before the actual intersection.
# So, we want to move the intersection further down the line to the actual intersection.
# To find the actual intersection, we will use an aprioristic rule:
#     when the angle between the two direction vectors
#     pointing from the intersection point towards the
#     next two points is less than 30 degrees, we will
#     move the intersection further up the line.
# This will also rename the graph appending "intersections" to the filename and "fixed intersections 30 deg" to the
# graph name.
#
# This script works on the graphs resampled at 0.5 mm between each node

import os, time, sys
import numpy
import matplotlib.pyplot as plt
import networkx
import hcatnetwork

from ..dataset import DATASET_ASOCA_GRAPHS_RESAMPLED_05MM

MIN_INTERSECTION_ANGLE_DEG = 40 # degrees (30, 35, 40)
SAVE_FIXED_GRAPHS = True

ASOCA_GRAPHS_DIR = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\"
ASOCA_GRAPHS_DIR = os.path.normpath(ASOCA_GRAPHS_DIR)

def main():
    print("Zipping up all ASOCA graphs. 30 degrees.")
    graphs_files = [os.path.join(ASOCA_GRAPHS_DIR, n) for n in DATASET_ASOCA_GRAPHS_RESAMPLED_05MM]
    graphs = []
    for graph_file in graphs_files:
        print("Loading graph:", graph_file)
        graph: hcatnetwork.graph.SimpleCenterlineGraph = hcatnetwork.io.load_graph(graph_file, output_type=hcatnetwork.graph.SimpleCenterlineGraph)
        graphs.append(graph)
    fixed_graphs = []
    # process
    time_ = time.time()
    for graph in graphs:
        # First, fix all endpoints that may have been not classified as such
        for n in graph.nodes():
            if (graph.degree(n) == 1) and (graph.nodes[n]["topology"] != hcatnetwork.node.ArteryNodeTopology.OSTIUM):
                graph.nodes[n]["topology"] = hcatnetwork.node.ArteryNodeTopology.ENDPOINT
        # find and list all intersection nodes
        intersections = [n for n in graph.nodes() if graph.degree(n) > 2]
        intersections_ostia = [graph.get_relative_coronary_ostia_node_id(n)[0] for n in intersections]
        intersections_distances_from_ostium = [networkx.dijkstra_path_length(graph, i, o) for i, o in zip(intersections,intersections_ostia)]
        intersections_order_of_management = numpy.argsort(intersections_distances_from_ostium)
        # process intersections
        for intersection_idx_ in intersections_order_of_management:
            intersection = intersections[intersection_idx_]
            intersection_ostium = intersections_ostia[intersection_idx_]
            original_intersection_node_id = intersection
            # Fix intersection
            # This is done by iteratively moving the intersection node to the middle of the two next nodes
            # until the angle between the two direction vectors pointing from the intersection point towards the
            # next two points is less than MIN_INTERSECTION_ANGLE_DEG degrees
            #
            # - initialize
            intersection_fixed = False
            current_intersection_node_id = original_intersection_node_id
            while not intersection_fixed:
                # useful data
                cidfo = float(networkx.dijkstra_path_length(graph, current_intersection_node_id, intersection_ostium)) # current intersection distance from ostium
                # get the two next nodes
                ngh = list(graph.neighbors(current_intersection_node_id))
                nghdfo = [networkx.dijkstra_path_length(graph, n, intersection_ostium) for n in ngh] # neighbours nodes distance from ostium
                next_nodes = [ngh[i_] for i_ in range(len(ngh)) if nghdfo[i_] > cidfo]
                # raise error if i have more (or less) than 2 next nodes - should not happen in ASOCA, we leverage this to simplify the algorithm
                if len(next_nodes) != 2:
                    raise ValueError(f"More (or less) than 2 next nodes found for intersection node {current_intersection_node_id}")
                # If one of the next nodes is an intersection, stop the zipping
                if any([graph.degree(n) > 2 for n in next_nodes]):
                    intersection_fixed = True
                elif any([graph.degree(n) == 1 for n in next_nodes]):
                    # Skip this intersection
                    intersection_fixed = True
                else:
                    # get the two next nodes
                    next_node_1 = next_nodes[0]
                    next_node_2 = next_nodes[1]
                    # get nodes position
                    current_intersection_position = numpy.array(
                        [graph.nodes[current_intersection_node_id]["x"], 
                         graph.nodes[current_intersection_node_id]["y"], 
                         graph.nodes[current_intersection_node_id]["z"]]
                    )
                    next_node_1_position = numpy.array(
                        [graph.nodes[next_node_1]["x"], 
                         graph.nodes[next_node_1]["y"], 
                         graph.nodes[next_node_1]["z"]]
                    )
                    next_node_2_position = numpy.array(
                        [graph.nodes[next_node_2]["x"], 
                         graph.nodes[next_node_2]["y"], 
                         graph.nodes[next_node_2]["z"]
                    ])
                    # get the direction vectors
                    direction_vector_1 = next_node_1_position - current_intersection_position
                    direction_vector_1_l = numpy.linalg.norm(direction_vector_1)
                    direction_vector_1_n = direction_vector_1 / direction_vector_1_l
                    direction_vector_2 = next_node_2_position - current_intersection_position
                    direction_vector_2_l = numpy.linalg.norm(direction_vector_2)
                    direction_vector_2_n = direction_vector_2 / direction_vector_2_l
                    # get the angle
                    angle_deg = numpy.rad2deg(
                        numpy.arccos(numpy.dot(direction_vector_1_n, direction_vector_2_n))
                    )
                    if angle_deg >= MIN_INTERSECTION_ANGLE_DEG:
                        intersection_fixed = True
                    else:
                        # get new intersection node position
                        new_intersection_position = current_intersection_position + numpy.mean([direction_vector_1_l, direction_vector_2_l])*(direction_vector_1 + direction_vector_2) / 2
                        # get the neighbours of the two next nodes
                        next_node_1_neighbours = list(graph.neighbors(next_node_1))
                        next_node_1_next_neighbours = [n for n in next_node_1_neighbours if n != current_intersection_node_id] # this list should be of length 1
                        if len(next_node_1_next_neighbours) != 1:
                            raise ValueError(f"More (or less) than 1 next node found for next node 1 {next_node_1}")
                        next_node_1_next_neighbour = next_node_1_next_neighbours[0]
                        next_node_2_neighbours = list(graph.neighbors(next_node_2))
                        next_node_2_next_neighbours = [n for n in next_node_2_neighbours if n != current_intersection_node_id] # this list should be of length 1
                        if len(next_node_2_next_neighbours) != 1:
                            raise ValueError(f"More (or less) than 1 next node found for next node 2 {next_node_2}")
                        next_node_2_next_neighbour = next_node_2_next_neighbours[0]
                        # ZIP is about to happen here
                        # - set current intersection node to be a normal (segment) node
                        graph.nodes[current_intersection_node_id]["topology"] = hcatnetwork.node.ArteryNodeTopology.SEGMENT
                        # - estimate radius of new intersection node
                        new_r = numpy.mean([graph.nodes[n]["r"] for n in [current_intersection_node_id, next_node_1, next_node_2]])
                        new_side = graph.nodes[current_intersection_node_id]["side"]
                        # - remove the two next nodes (and the edges with them)
                        graph.remove_node(next_node_1)
                        graph.remove_node(next_node_2)
                        # - add the new intersection node
                        new_intersection_node_id = str(max([int(n) for n in graph.nodes()]) + 1)
                        graph.add_node(
                            node_for_adding=new_intersection_node_id,
                            x=new_intersection_position[0],
                            y=new_intersection_position[1],
                            z=new_intersection_position[2],
                            r=new_r,
                            t=0.0,
                            topology=hcatnetwork.node.ArteryNodeTopology.INTERSECTION,
                            side=new_side
                        )
                        # - add the new edges
                        # - - current intersection to new intersection
                        edge_attributes = hcatnetwork.edge.SimpleCenterlineEdgeAttributes()
                        edge_attributes["euclidean_distance"] = numpy.linalg.norm(new_intersection_position - current_intersection_position)
                        edge_attributes.update_weight_from_euclidean_distance()
                        graph.add_edge(current_intersection_node_id, new_intersection_node_id, attributes_dict=edge_attributes)
                        # - - new intersection to next node 1 next neighbour
                        next_node_1_next_neighbour_position = numpy.array(
                            [graph.nodes[next_node_1_next_neighbour]["x"], 
                             graph.nodes[next_node_1_next_neighbour]["y"], 
                             graph.nodes[next_node_1_next_neighbour]["z"]]
                        )
                        edge_attributes = hcatnetwork.edge.SimpleCenterlineEdgeAttributes()
                        edge_attributes["euclidean_distance"] = numpy.linalg.norm(new_intersection_position - next_node_1_next_neighbour_position)
                        edge_attributes.update_weight_from_euclidean_distance()
                        graph.add_edge(new_intersection_node_id, next_node_1_next_neighbour, attributes_dict=edge_attributes)
                        # - - new intersection to next node 2 next neighbour
                        next_node_2_next_neighbour_position = numpy.array(
                            [graph.nodes[next_node_2_next_neighbour]["x"], 
                             graph.nodes[next_node_2_next_neighbour]["y"], 
                             graph.nodes[next_node_2_next_neighbour]["z"]]
                        )
                        edge_attributes = hcatnetwork.edge.SimpleCenterlineEdgeAttributes()
                        edge_attributes["euclidean_distance"] = numpy.linalg.norm(new_intersection_position - next_node_2_next_neighbour_position)
                        edge_attributes.update_weight_from_euclidean_distance()
                        graph.add_edge(new_intersection_node_id, next_node_2_next_neighbour, attributes_dict=edge_attributes)
                        # - update current intersection node id
                        current_intersection_node_id = new_intersection_node_id
                        # evaluation of the new intersection is dlegated to the next iteration
        # For some (yet unknown) reason, some intersections are not removed by the previous algorithm
        # This is a quick fix for that
        for n in graph.nodes():
            if graph.degree(n) > 2:
                graph.nodes[n]["topology"] = hcatnetwork.node.ArteryNodeTopology.INTERSECTION
            if graph.degree(n) > 3:
                raise ValueError(f"Node {n} has degree {graph.degree(n)}, maximum for this dataset is 3.")
            if graph.nodes[n]["topology"] == hcatnetwork.node.ArteryNodeTopology.INTERSECTION and graph.degree(n) == 2:
                graph.nodes[n]["topology"] = hcatnetwork.node.ArteryNodeTopology.SEGMENT
                position = numpy.array([graph.nodes[n]["x"], graph.nodes[n]["y"], graph.nodes[n]["z"]])
                print(f"Node {n} has been fixed (graph {graph.graph['image_id']}), position {position}.")
        # Final resample (it is already sampled at 0.5 mm between nodes, but we want to resample due to the new nodes added)
        new_graph = graph.resample(mm_between_nodes=0.5, update_image_id=False)
        # Rename graph
        old_name = new_graph.graph["image_id"]
        new_name = old_name + f", intersections {int(MIN_INTERSECTION_ANGLE_DEG)}°"
        new_graph.graph["image_id"] = new_name
        # Save graph
        fixed_graphs.append(new_graph)
    print(f"Processed all graphs in {time.time() - time_:.2f} seconds.")
    # Save fixed graphs - same folder where the original graphs are
    if SAVE_FIXED_GRAPHS:
        for i_, graph in enumerate(fixed_graphs):
            old_graph_name = DATASET_ASOCA_GRAPHS_RESAMPLED_05MM[i_]
            new_graph_name = old_graph_name.replace(".GML", f"_intersections_{int(MIN_INTERSECTION_ANGLE_DEG)}deg.GML")
            graph_file = os.path.join(ASOCA_GRAPHS_DIR, new_graph_name)
            print("Saving graph: ", graph_file)
            hcatnetwork.io.save_graph(graph, graph_file)



if __name__ == "__main__":
    main()