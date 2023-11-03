import os, sys
import numpy


if __name__ == "__main__":
    print("Running 'DatasetUtilities.utils' module")
    
    ## view a saved centerline graph
    import hcatnetwork
    graph_save_path = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\dataset00\\dataset00.GML"
    g = hcatnetwork.graph.loadGraph(graph_save_path)
    hcatnetwork.draw.draw2DCenterlinesGraph(g)
    hcatnetwork.draw.draw3DCenterlinesGraph(g)