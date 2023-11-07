import os, sys
import numpy 
import matplotlib.pyplot as plt
import networkx
import hcatnetwork
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

import HearticDatasetManager.deepvesselnet.convert_to_hcatnetwork.utils as util


VTK_FILE = "C:\\Users\\lecca\\Desktop\\DeepVesselNet\\centerline_graphs_vtk\\synthetic_arteries_graph_59.vtk"

centerlines = util.read_vtk_centerlines(VTK_FILE)

print(centerlines)

for centerline in centerlines:
    plt.plot(centerline[:,0], centerline[:,1])
plt.show()
    