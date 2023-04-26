import os, sys
import numpy


if __name__ == "__main__":
    print("Running 'DatasetUtilities.utils' module")
    
    ## view a saved graph
    import matplotlib.pyplot as plt
    import networkx, HCATNetwork
    graph_save_path = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\DatasetUtilities\\DatasetUtilities\\CAT08\\aaa_graph_prova.GML"
    graph_2 = HCATNetwork.graph.loadGraph(graph_save_path)
    if 1:
        color_list__ = []
        color_tree_list__ = []
        pos_dict__ = {}
        for n in graph_2.nodes:
            n_scn = HCATNetwork.node.SimpleCenterlineNode(**(graph_2.nodes[n]))
            pos_dict__.update(**{n: n_scn.getVertexList()[:2]})
            color_list__.append(n_scn["topology_class"].value)
            color_tree_list__.append("red" if n_scn["arterial_tree"].value == HCATNetwork.node.ArteryPointTree.RIGHT.value else "blue")
        ax = plt.subplot(111)
        networkx.draw_networkx_nodes(
            graph_2,
            **{"node_color": color_tree_list__, 
                "node_size": 100,
                "pos": pos_dict__,
                "ax": ax
                }
        )
        networkx.draw(
            graph_2,
            **{"with_labels": False, 
                "node_color": color_list__, 
                "node_size": 20,
                "pos": pos_dict__,
                "ax": ax
                }
        )
        ax.grid("on")
        plt.show()