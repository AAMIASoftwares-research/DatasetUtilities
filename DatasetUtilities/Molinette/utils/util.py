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