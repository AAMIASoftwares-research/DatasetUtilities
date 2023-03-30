import os
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def importCenterlineList(cat08_im_folder_path:str):
    v_list = []
    for i_ in range(4):
        v_file_path = os.path.join(cat08_im_folder_path, f"vessel{i_}", "reference.txt")
        v_list.append(numpy.loadtxt(v_file_path, delimiter=" ", usecols=range(4)))
    return v_list


def getCentelrineArcLength(centerline: numpy.ndarray, startIdx: int|None = None, endIdx: int|None = None) -> float:
        """ Function to get the length of a centerline, defined as a 3D point sequence
        starting from the ostium and ending at its endpoint.
        :param centerline: NxM numpy.ndarray with M>=3
        :param startIdx: index of the starting point of the arc length measure. If None, 0
        :param endIdx: index of the end point of the arc length measure. If None: end
        """
        if centerline.shape[1] < 3:
            raise RuntimeWarning(f"Shape of the passed numpy.ndarray is not acceptable, must be NxM with M>=3, instead it is: {centerline.shape[0]}")
        if startIdx is None:
            startIdx = 0
        if endIdx is None:
            endIdx = centerline.shape[0] - 1
        if startIdx > endIdx - 1:
            raise RuntimeError(f"Invalid startIdx={startIdx} and endIdx={endIdx} with centerline of shape={centerline.shape}")
        return float(numpy.sum(numpy.linalg.norm(centerline[startIdx+1:endIdx,:3]-centerline[startIdx:endIdx-1,:3], axis = 1)))

def getCenterlinePointFromArcLength(reference_centerline: numpy.ndarray, arc_length:float):
    """Linear interpolation"""
    total_length = getCentelrineArcLength(reference_centerline)
    if arc_length == 0:
        return reference_centerline[0]
    if arc_length == total_length:
        return reference_centerline[-1]
    if arc_length > total_length or arc_length < 0:
        return None
    idx_before = 0
    len_before = 0.0
    last_len_before = 0.0
    while len_before < arc_length:
        last_len_before = numpy.linalg.norm(reference_centerline[idx_before+1,:3]-reference_centerline[idx_before,:3])
        len_before += last_len_before
        idx_before += 1
    # Reached the point for which the s is between before and after
    exceeded_len = arc_length - (len_before-last_len_before)
    covered_percentage = exceeded_len/last_len_before if last_len_before != 0 else 0.0
    out_point =  covered_percentage*reference_centerline[idx_before+1,:] + (1-covered_percentage)*reference_centerline[idx_before,:]
    return out_point

def divideArterialTreesFromOriginalCenterlines(centerlines_list: list) -> tuple:
    ## preprocess the centerlines: necessary because some of them have the ostium very
    # different from one another in the common segments
    # 1 - find the two arterial trees right away: we use just the first points of the centerlines
    ostia = numpy.array([l[0,:3] for l in centerlines_list])
    min_distance_between_two_trees = 20 #mm
    db = DBSCAN(eps=min_distance_between_two_trees, min_samples=1).fit(ostia)
    tree1_idxs = numpy.argwhere(db.labels_==0).flatten()
    tree1_list = [centerlines_list[i] for i in tree1_idxs]
    tree1_ostia = ostia[tree1_idxs,:]
    tree2_idxs = numpy.argwhere(db.labels_==1).flatten()
    tree2_list = [centerlines_list[i] for i in tree2_idxs]
    tree2_ostia = ostia[tree2_idxs,:]
    return (tree1_idxs, tree1_list, tree2_idxs, tree2_list)



def getPointToCenterlinePointsMinDistance(p:numpy.ndarray, centerline: numpy.ndarray) -> tuple:
    """Description later..."""
    d = numpy.linalg.norm(p[:3] - centerline[:,:3], axis= 1)
    idx_min = numpy.argmin(d).flatten()[0]
    dist_min = d[idx_min]
    return (idx_min, dist_min)

def getCentelrinesFurthestConnectionIndex(c_i: numpy.ndarray, c_j: numpy.ndarray, thresh_radius_multiplier=0.4):
    """
    """
    conn_index = int(-1)
    # work in c_i
    d_last = 1e9
    for i_ in range(len(c_i)):
        # get dist and compute logic
        _, dmin = getPointToCenterlinePointsMinDistance(
            p=c_i[i_],
            centerline=c_j
        )
        thresh = 0.5 * c_i[i_][3]
        if dmin < thresh:
            d_last = dmin
            conn_index = i_
    # go back a couple of indexes to have smoother intersections
    if conn_index > 30:
        for i_ in range(conn_index, conn_index - 30, -1):
            _, dmin = getPointToCenterlinePointsMinDistance(
                p=c_i[i_],
                centerline=c_j
            )
            if dmin < 0.1*c_i[i_][3]:
                conn_index = i_ - 3
                break
            if dmin < d_last:
                d_last = dmin
                conn_index = i_
    return conn_index



###############
# VISUALISATION 
###############

colors=["red", "blue", "orange", "green", "black", "pink", "yellow"]

def plotCenterlineList(centerlines_list: list):
    ax1 = plt.subplot(221)
    n = len(centerlines_list)
    for i_ in range(n):
        ax1.plot(centerlines_list[i_][:,0],centerlines_list[i_][:,1],
                 ".-", label=str(i_), color=colors[i_])
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.legend()
    #
    ax2 = plt.subplot(222, projection="3d")
    for i_ in range(n):
        ax2.plot(centerlines_list[i_][:,0],centerlines_list[i_][:,1], centerlines_list[i_][:,2],
                 color="black", linewidth=0.7)
        ax2.scatter(centerlines_list[i_][:,0],centerlines_list[i_][:,1], centerlines_list[i_][:,2],
                 s=centerlines_list[i_][:,3]**2, label=str(i_), c=colors[i_])
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    ax2.set_zlabel("z [mm]")
    ax2.legend()
    #
    ax3 = plt.subplot(212)
    for i_ in range(n):
        d = numpy.linalg.norm(centerlines_list[i_][0,:3] - centerlines_list[i_][:,:3], axis=1)
        ax3.plot(range(d.shape[0]), d, label=str(i_))
    ax3.set_xlabel("idx")
    ax3.set_ylabel("distance from first point")
    ax3.legend()
    plt.show()

def plotCenterlineListWithIndexes(centerlines_list: list):
    n = len(centerlines_list)
    # x, y
    ax1 = plt.subplot(121)
    for i_ in range(n):
        ax1.plot(centerlines_list[i_][:,0], centerlines_list[i_][:,1], ".-", label=str(i_)+f" : n_points={centerlines_list[i_].shape[0]}", color=colors[i_], alpha=0.4)
        for i_p in range(0, centerlines_list[i_].shape[0], 10):
            ax1.text(centerlines_list[i_][i_p,0], centerlines_list[i_][i_p,1], str(i_p), color=colors[i_])
        ax1.text(centerlines_list[i_][-1,0], centerlines_list[i_][-1,1], str(centerlines_list[i_].shape[0]-1), color=colors[i_])
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("y [mm]")
    ax1.legend()
    ax1.axis("equal")
    # x, z
    ax2 = plt.subplot(222)
    for i_ in range(n):
        ax2.plot(centerlines_list[i_][:,0], centerlines_list[i_][:,2], ".-", label=str(i_), color=colors[i_])
        for i_p in range(0, centerlines_list[i_].shape[0], 20):
            ax2.text(centerlines_list[i_][i_p,0], centerlines_list[i_][i_p,2], str(i_p), color=colors[i_])
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("z [mm]")
    ax2.axis("equal")
    # y, z
    ax3 = plt.subplot(224)
    for i_ in range(n):
        ax3.plot(centerlines_list[i_][:,1], centerlines_list[i_][:,2], ".-", label=str(i_), color=colors[i_])
    ax3.set_xlabel("y [mm]")
    ax3.set_ylabel("z [mm]")
    ax3.axis("equal")
    # out
    plt.show()
