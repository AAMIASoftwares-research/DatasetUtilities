import os, sys
import numpy 
import matplotlib.pyplot as plt
import networkx
import hcatnetwork
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

import HearticDatasetManager.ASOCA.convert_to_hcatnetwork.utils as util


def process_image_and_generate_graph(asoca_folder: str, dataset: str, im_number: int):
    """Process the image and generate the graph.

    Parameters
    ----------
    asoca_folder : str
        Path to the ASOCA folder.
    dataset : str
        Dataset name: "Normal" or "Diseased".
    im_number : int
        Image number in [1;20]
    """
    # Input clean
    im_number = int(im_number)
    if im_number < 1 or im_number > 20:
        raise ValueError("Image number must be in [1;20].")
    if dataset not in ["Normal", "Diseased"]:
        raise ValueError("Dataset must be either 'Normal' or 'Diseased'.")
    
    # Read all the data from vtk centerline file
    filename = os.path.join(
        asoca_folder,
        dataset,
        "Centerlines",
        dataset + f"_{im_number}.vtp")
    points_nparray_container = util.read_vtk_centerlines(filename)
    
    # Plot raw data
    if 0:
        util.plod_segments_container(points_nparray_container)
        quit()

    ##################
    # CREATE THE GRAPH
    ##################
    
    # Initialize graph
    graph = hcatnetwork.graph.SimpleCenterlineGraph(
        image_id = f"ASOCA/{dataset}_{im_number}",
        are_left_right_disjointed = True
    )

    # Fill the graph with nodes and edges
    util.process_points_to_graph(points_nparray_container, graph)

    if 0:
        # Plot the graph
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph, backend="debug")
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph, backend="networkx")
        quit()

    ##########################
    # CLEANING - INTERSECTIONS
    ##########################

    for n in graph.nodes:
        if graph.degree[n] < 1:
            raise RuntimeError(f"node {n} has no connections. Cannod be.")
        if graph.degree[n] > 2:
                graph.nodes[n]["topology"] = hcatnetwork.node.ArteryNodeTopology.INTERSECTION
        if graph.degree[n] > 3:
            print(f"WARNING: node {n} has more than 3 connections. Allowed, but do not expect this in this dataset.\nPlease check.")
            hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph)

    #################################
    # CLEANING - OSTIA AND TREE SIDES
    #################################

    # interactive checking
    left_ostium_node_id, right_ostium_node_id = graph.get_coronary_ostia_node_id()
    hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph, backend="networkx")
    # Check and swap ostia if needed by using the position of the x coordinate
    is_ok = input("Are the graph right/left sides ok? [y/n]   ")
    if is_ok != "y":
        left_ostium_node_id_old = left_ostium_node_id
        right_ostium_node_id_old = right_ostium_node_id
        left_ostium_node_id, right_ostium_node_id = util.check_and_swap_ostia_if_needed(
             graph,
             left_ostium_node_id,
             right_ostium_node_id
        )
        if left_ostium_node_id == left_ostium_node_id_old:
            # swap anyways
            graph.nodes[left_ostium_node_id]["side"] = hcatnetwork.node.ArteryNodeSide.RIGHT
            graph.nodes[right_ostium_node_id]["side"] = hcatnetwork.node.ArteryNodeSide.LEFT
            right_ostium_node_id, left_ostium_node_id = left_ostium_node_id_old, right_ostium_node_id_old
    # Reset all left and right tree flags based on their connected ostium since the dataset has not ordered data
    reachable_from_left_ostium = [k for k in networkx.single_source_dijkstra_path_length(graph, left_ostium_node_id)]
    reachable_from_right_ostium = [k for k in networkx.single_source_dijkstra_path_length(graph, right_ostium_node_id)]
    for n in reachable_from_left_ostium:
        graph.nodes[n]["side"] = hcatnetwork.node.ArteryNodeSide.LEFT
    for n in reachable_from_right_ostium:
        graph.nodes[n]["side"] = hcatnetwork.node.ArteryNodeSide.RIGHT
    # Final check to see if there are ronin nodes
    for n in graph.nodes:
            if not n in reachable_from_left_ostium and not n in reachable_from_right_ostium:
                raise RuntimeError(f"Node {n} is not connected to any ostium. Please check.")
    # user check
    hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph)
    is_ok = input("Is the graph ready to be saved? [y/n]   ")
    if is_ok != "y":
        raise RuntimeError("Quitting under user command...")
    
    ###############
    #Save the graph
    ###############

    if not os.path.exists(os.path.join(asoca_folder, dataset, "Centerlines_graphs")):
        os.makedirs(os.path.join(asoca_folder, dataset, "Centerlines_graphs"))
    # save graph original
    graph_save_path = os.path.join(
        asoca_folder,
        dataset,
        "Centerlines_graphs",
        dataset + f"_{im_number}.GML")
    hcatnetwork.io.save_graph(graph, graph_save_path)
    print(f"Graph saved at {graph_save_path}.")
    # save resampled graph (at 0.5 mm)
    graph_resampled = graph.resample(0.5)
    graph_save_path = os.path.join(
        asoca_folder,
        dataset,
        "Centerlines_graphs",
        dataset + f"_{im_number}_0.5mm.GML")
    hcatnetwork.io.save_graph(graph_resampled, graph_save_path)
    print(f"Resampled graph saved at {graph_save_path}.\n")
            
    

    

if __name__ == "__main__":
    # Loop to process all the images of the two datatsets
    ASOCA_FOLDER = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\"
    DATASTES = ["Normal", "Diseased"]
    IMAGE_NUMBERS = {
        "Normal": range(1, 21),
        "Diseased": range(1, 21)
    }
    
    print("\n\n")
    for DATASET in DATASTES:
        for IM_NUMBER in IMAGE_NUMBERS[DATASET]:
            print(f"Processing {DATASET} image {IM_NUMBER}...")
            process_image_and_generate_graph(ASOCA_FOLDER, DATASET, IM_NUMBER)
    
