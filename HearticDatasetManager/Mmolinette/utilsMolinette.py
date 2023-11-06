import os
import numpy
import matplotlib.pyplot as plt

def importCenterlineAndNormals(centerline_file:str):
    out = numpy.loadtxt(centerline_file, delimiter=",", skiprows=1)
    return out

def importCenterline(centerline_file:str):
    out = numpy.loadtxt(centerline_file, delimiter=",", skiprows=1, usecols=range(3))
    return out

def splitOverlappingCenterlines(centerline: numpy.ndarray):
    d = numpy.linalg.norm(centerline[1:,:3]-centerline[0:-1,:3], axis=1)
    idxs = numpy.argwhere(d > 15).flatten()
    si = idxs + 1
    si = numpy.insert(si, 0, 0)
    ee = idxs + 1
    ee = numpy.append(ee, centerline.shape[0])
    out_list = []
    for s, e in zip(si, ee):
        out_list.append(
            centerline[s:e,:].copy()
        )
    return out_list


### VISUALISATION ###

def plotCenterlines3D(centerline: numpy.ndarray, ax=None, title:str=" "):
    if ax is None:
        ax = plt.subplot(111, projection="3d")
    ax.plot(centerline[:,0], centerline[:,1], centerline[:,2], ".-", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    plt.show()

def plotCenterlines2D(centerline: numpy.ndarray, ax=None, title:str=" "):
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(centerline[:,0], centerline[:,1], ".-", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.show()

### VISUALIZATION IN SLICER ###

def centerlineToMarkerListSlicer(centerline: numpy.ndarray, outfile):
    '''
    Assumes the centerline to be at least x, y, z colums
    '''
    header_string = """# Markups fiducial file version = 5.2
# CoordinateSystem = LPS
# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID
"""
    c_id = numpy.arange(centerline.shape[0], dtype="int")
    c_data = centerline[:,:3]
    c_ow_ox_oy_oz_vis_sel_lock = numpy.array([[0,0,0,1,1,1,0] for i in range(c_data.shape[0])], dtype="int")
    label = numpy.array([f"point{i:04d}" for i in range(c_data.shape[0])])
    desc = numpy.array(["" for i in range(c_data.shape[0])])
    assoc_node_id = numpy.array(["vtkMRMLScalarVolumeNode" for i in range(c_data.shape[0])])
    last_cols = numpy.array([[2,0] for i in range(c_data.shape[0])], dtype="int")
    # merge all
    fcsv_string = header_string
    for i in range(c_data.shape[0]):
        fcsv_string += f"{c_id[i]:d},{c_data[i,0]:.5f},{c_data[i,1]:.5f},{c_data[i,2]:.5f},"\
            + f"{c_ow_ox_oy_oz_vis_sel_lock[i,0]:d},{c_ow_ox_oy_oz_vis_sel_lock[i,1]:d},{c_ow_ox_oy_oz_vis_sel_lock[i,2]:d},{c_ow_ox_oy_oz_vis_sel_lock[i,3]:d},{c_ow_ox_oy_oz_vis_sel_lock[i,4]:d},{c_ow_ox_oy_oz_vis_sel_lock[i,5]:d},{c_ow_ox_oy_oz_vis_sel_lock[i,6]:d},"\
            + f"{label[i]},{desc[i]},{assoc_node_id[i]},{last_cols[i,0]},{last_cols[i,1]}\n"
    # out
    f = open(outfile, "w")
    f.write(fcsv_string)
    f.close()

