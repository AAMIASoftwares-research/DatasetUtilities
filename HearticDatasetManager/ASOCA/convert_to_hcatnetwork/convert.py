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
        image_id = f"ASOCA/{dataset}/{im_number}",
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
    hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph, backend="networkx")
    is_ok = input("Is the graph sides ok? [y/n]")
    if is_ok != "y":
        left_ostium_node_id, right_ostium_node_id = graph.get_coronary_ostia_node_id(graph)
        # Check and swap ostia if needed by using the position of the x coordinate
        left_ostium_node_id, right_ostium_node_id = util.check_and_swap_ostia_if_needed(graph, left_ostium_node_id, right_ostium_node_id)
        # Reset all left and right tree flags based on their connected ostium since the dataset has not ordered data
        reachable_from_left_ostium = [k for k in networkx.single_source_dijkstra_path_length(graph, left_ostium_node_id)]
        reachable_from_right_ostium = [k for k in networkx.single_source_dijkstra_path_length(graph, right_ostium_node_id)]
        for n in reachable_from_left_ostium:
                graph.nodes[n]["side"] = hcatnetwork.node.ArteryNodeSide.LEFT
        for n in reachable_from_left_ostium:
            graph.nodes[n]["side"] = hcatnetwork.node.ArteryNodeSide.RIGHT
        # Final check to see if there are ronin nodes
        for n in graph.nodes:
             if not n in reachable_from_left_ostium and not n in reachable_from_right_ostium:
                  raise RuntimeError(f"Node {n} is not connected to any ostium. Please check.")
        # user check
        hcatnetwork.draw.draw_simple_centerlines_graph_2d(graph, backend="networkx")
        is_ok = input("Is the graph sides ok now? [y/n]")
        if is_ok != "y":
            raise RuntimeError("Graph sides are not ok. Please check.")
        
    
    ###############
    #Save the graph
    ###############

    graph_save_path = f"C:\\Users\\madda\\Desktop\\Codice_temp\\Data\\ASOCA\\Grafi_ASOCA_Diseased_orig\\Diseased_{dataset, im_number:02d}_orig.GML"
    HCATNetwork.graph.saveGraph(graph, graph_save_path)
            
    ###############
    #Plot the graph
    ###############
    
    ostia = HCATNetwork.graph.BasicCenterlineGraph.get_coronary_ostia_node_id(graph)
    print(ostia)

    for n in graph.nodes:
        if graph.nodes[n]['topology'].value == HCATNetwork.node.ArteryPointTopologyClass.OSTIUM.value:
                print(graph.nodes(data=True)[n])


    # Visualize each graph
    drawCenterlinesGraph2D(graph, backend="hcatnetwork")
    return

    

if __name__ == "__main__":
    # Loop to process all the images of the two datatsets
    ASOCA_FOLDER = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\"
    DATASTES = ["Normal", "Diseased"]
    IMAGE_NUMBERS = range(1, 21)
    for DATASET in DATASTES:
        for IM_NUMBER in IMAGE_NUMBERS:
            process_image_and_generate_graph(ASOCA_FOLDER, DATASET, IM_NUMBER)
    
