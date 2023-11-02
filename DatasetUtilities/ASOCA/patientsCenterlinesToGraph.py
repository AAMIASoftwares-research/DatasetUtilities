import os, sys
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import networkx, HCATNetwork
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

# Read all the data from the file
filename = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\normal_prova\\Centerlines\\Normal_1.vtp"

reader = vtkXMLPolyDataReader()
reader.SetFileName(filename)
reader.Update()

polydata = reader.GetOutput()
print(type(polydata))
# read https://stackoverflow.com/questions/33838986/how-to-read-data-in-vtkdataarray

# vtkPointData - Points Radiuses
point_data = polydata.GetPointData()
print(type(point_data))
data_array = point_data.GetArray("MaximumInscribedSphereRadius")
print(type(data_array))
MaximumInscribedSphereRadius_list = [data_array.GetValue(i) for i in range(data_array.GetSize())]
print(len(MaximumInscribedSphereRadius_list), type(MaximumInscribedSphereRadius_list))

# Get centerlines connectivity infos
# vtkCellArray - stores dataset topologies as an explicit connectivity table listing the point ids that make up each cell.
# https://vtk.org/doc/nightly/html/classvtkCellArray.html#details look at extended description in web page
lines_celldata = polydata.GetLines()
print(type(lines_celldata))
# centelrines points divisions
offset_array = lines_celldata.GetOffsetsArray()
cells_idx_list = [offset_array.GetValue(i) for i in range(offset_array.GetSize())]
# all centerlines points
connectivity_array = lines_celldata.GetConnectivityArray()
connectivity_list = [connectivity_array.GetValue(i) for i in range(connectivity_array.GetSize())]

# get centerline points coords
# vtkPoints
points = polydata.GetPoints()
points_nparray_container = []
#Points_list = [points.GetPoint(i) for i in connectivity_list]
surf=2**8
zorder=1
ax = plt.subplot(111)
for s,e_excl in zip(cells_idx_list[:-1], cells_idx_list[1:]):
    print(s, e_excl)
    points_nparray_container.append(
        numpy.array(
            [points.GetPoint(i) for i in connectivity_list[s:e_excl]]
        )
    )
    ###
    ###  VISUALIZE DATA
    ###
    ax.scatter(points_nparray_container[-1][:,0], points_nparray_container[-1][:,1], s=surf, zorder=zorder)
    # points_nparray_container[a][b,c]
    # a: indice del singolo tracciato di centerline
    # b: indice del singolo punto
    # c: 0->x, 1->y, 2->z
    surf /= 2
    zorder+=1
ax.set_xlabel("Right -> Left (mm)")
ax.set_ylabel("Anterior -> Posterior (mm)")
plt.show()




quit()







## EXAMPLE FROM VTK WEBSITE
"""Below example found on vtk website, precooked.
It works, but absolutely no idea of why."""
# noinspection PyUnresolvedReferences
import vtkmodules.vtkInteractionStyle
# noinspection PyUnresolvedReferences
import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def main():
    colors = vtkNamedColors()

    # Read all the data from the file
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()

    # Visualize
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(colors.GetColor3d('NavajoWhite'))

    renderer = vtkRenderer()
    renderWindow = vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d('DarkOliveGreen'))
    renderer.GetActiveCamera().Pitch(90)
    renderer.GetActiveCamera().SetViewUp(0, 0, 1)
    renderer.ResetCamera()

    renderWindow.SetSize(600, 600)
    renderWindow.Render()
    renderWindow.SetWindowName('ReadPolyData')
    renderWindowInteractor.Start()


if __name__ == '__main__':
    main()

