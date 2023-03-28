import os, sys
import numpy
from sklearn.cluster import DBSCAN
import utils.utils as util

CAT08_IM_folder = os.path.normpath(
    "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\CAT08\\dataset05\\"
)


if __name__ == "__main__":
    # load centelrines
    centerlines_list = util.importCenterlineList(CAT08_IM_folder)
    if 1:
        util.plotCenterlineList(centerlines_list)

    # divide left and right arterial tree
    t1_list, t2_list = util.divideArterialTreesFromOriginalCenterlines(centerlines_list)
    if 0:
        util.plotCenterlineList(t1_list)
        util.plotCenterlineList(t2_list)

    # get connection matrix
    # the connection matrix N x N, with N the number of centerlines in the single arterial tree,
    # stores the index of the ith (row) centerline for which, after that index, the centelrine is
    # no more connected to the jth (column) centelrine.
    # Should be symmetric if the analysed centerline are all from the same tree (same ostium), 
    # but not symmetric if centerlines of different trees are used.
    # tree 1
    t1_connection_matrix = numpy.zeros((len(t1_list), len(t1_list)), dtype="int") - 1
    for i_v in range(len(t1_list)):
        for j_v in range(len(t1_list)):
            if i_v != j_v:
                # work in i_v
                t1_connection_matrix[i_v, j_v] = util.getCentelrinesFurthestConnectionIndex(
                    c_i=t1_list[i_v],
                    c_j=t1_list[j_v], 
                    thresh_radius_multiplier=0.3
                )
    # tree 2
    t2_connection_matrix = numpy.zeros((len(t2_list), len(t2_list)), dtype="int") - 1
    for i_v in range(len(t2_list)):
        for j_v in range(len(t2_list)):
            if i_v != j_v:
                # work in i_v
                t2_connection_matrix[i_v, j_v] = util.getCentelrinesFurthestConnectionIndex(
                    c_i=t2_list[i_v],
                    c_j=t2_list[j_v], 
                    thresh_radius_multiplier=0.3
                )
        
    if 0:
        print("debug plot")
        import matplotlib.pyplot as plt
        for i in range(len(t2_list)):
            plt.scatter(t2_list[i][:,0], t2_list[i][:,1], s=20)
        for i_v in range(len(t2_list)):
            for j_v in range(len(t2_list)):
                if t2_connection_matrix[i_v, j_v] != -1:
                    idx = t2_connection_matrix[i_v, j_v]
                    plt.scatter(
                        t2_list[i_v][idx,0],
                        t2_list[i_v][idx,1],
                        s=10, c="red")
                    plt.plot(
                        [t2_list[i_v][t2_connection_matrix[i_v, j_v],0], t2_list[j_v][t2_connection_matrix[j_v, i_v],0]],
                        [t2_list[i_v][t2_connection_matrix[i_v, j_v],1], t2_list[j_v][t2_connection_matrix[j_v, i_v],1]],
                        color="red"
                    )
        plt.axis("equal")
        plt.show()

    # mean for each centerline with the closest point on the connected centelrines

    # tests on tree 2
    points_list = []
    for i_v in range(len(t2_list)):
        for i_p in range(len(t2_list[i_v])):
            if (t2_connection_matrix[i_v, :] != -1).any():
                p_to_mean = [t2_list[i_v][i_p].tolist()]
                for j_v in range(len(t2_list)):
                    if t2_connection_matrix[i_v, j_v] != -1 and i_p < t2_connection_matrix[i_v, j_v]:
                        idx_min, _ = util.getPointToCenterlinePointsMinDistance(
                            p=t2_list[i_v][i_p],
                            centerline=t2_list[j_v]
                        )
                        if idx_min < t2_connection_matrix[j_v, i_v]:
                            p_to_mean.append(t2_list[j_v][idx_min].tolist())
                p_mean = numpy.mean(p_to_mean, axis=0).tolist() if len(p_to_mean) > 1 else p_to_mean[0]
            else:
                p_mean = t2_list[i_v][i_p].tolist()
            if not p_mean in points_list:
                points_list.append(p_mean)
    points_list = numpy.array(points_list)

    if 1:
        print("debug plot")
        import matplotlib.pyplot as plt
        for i in range(len(t2_list)):
            plt.scatter(t2_list[i][:,0], t2_list[i][:,1], s=40, alpha=0.7)
        plt.scatter(points_list[:,0], points_list[:,1], s=10)
        plt.show()
        






    sys.exit()
    # old method
    

    # Now find the almost-mean-shift
    spacing=0.29
    max_length = numpy.max([getCentelrineLength(c) for c in v_list])
    tuple_list = []
    r_thresh_modifier_count = 0
    len_dbscan_set_previous = 0 
    d_ostium_array = numpy.linspace(0,max_length, int(max_length/spacing))
    i_ = 0
    while i_ < d_ostium_array.shape[0]:
        d_ostium = d_ostium_array[i_]
        p_d = [getContinuousCenterline(c, d_ostium) for c in v_list]
        p_d_pruned = [x for x in p_d if x is not None]
        r = numpy.mean([p[3] for p in p_d_pruned])
        if r_thresh_modifier_count == 0:
            r_thresh = 0.75*numpy.exp(-0.8*d_ostium/10) + 0.25
        else:
            r_thresh = 0.12
            r_thresh_modifier_count -= 1
        db = DBSCAN(
            eps=r_thresh*r,
            min_samples=1
        ).fit(numpy.array(p_d_pruned)[:,:3])
        if i_ > 0 and r_thresh_modifier_count==0:
            if len(set(db.labels_)) > len_dbscan_set_previous:
                # This means that, since the last step, something branched off
                # To make sure that the point in which it branches off is very near to the previous segment
                # (no hard turns), we go back 3 < n < 5 steps and temporarily set the 
                # segments-branching threshold (eps in DBSCAN) to a much lower value, so that
                # the interested segments can branch off before they would do with the previous threshold.
                n_backwards = 4
                i_ = max(0, i_ - n_backwards - 1)
                r_thresh_modifier_count = n_backwards + 1
                for ii in range(
                    min(len(tuple_list),n_backwards+1)
                    ):
                    tuple_list.pop()
                len_dbscan_set_previous = len(set(db.labels_))
                continue
        if r_thresh_modifier_count == 0:
            len_dbscan_set_previous = len(set(db.labels_))
        p_out_list = []
        for index in set(db.labels_):
            idx_positions = numpy.argwhere(db.labels_ == index).flatten()                
            if len(idx_positions) < 1:
                raise RuntimeError("Should always be at least 1")
            elif len(idx_positions) == 1:
                p_out_list.append(p_d_pruned[idx_positions[0]])
            else:
                p_d_pruned_out = numpy.array(
                    [p_d_pruned[i] for i in idx_positions]
                )
                p_new = numpy.mean(
                    p_d_pruned_out,
                    axis=0
                )
                p_out_list.append(p_new)
        tuple_list.append((d_ostium, p_out_list))
        # Update iterator
        i_ += 1
        print(f"{100*i_/len(d_ostium_array):3.1f}", end="\r")
    print("")

    vl, cl = [], []
    for t in tuple_list:
        for v in t[1]:
            vl.append([v[0], v[1], v[2], v[3]])
            cl.append(t[0])
    vl = numpy.array(vl)
    cl = numpy.array(cl).flatten()
    # plots
    if 0:
        for i_ in range(4):
            plt.plot(v_list[i_][:,0],v_list[i_][:,1])
        plt.scatter(vl[:,0], vl[:,1], c=cl)
        plt.show()

    ## now, we do a final dbscan so we can separate the two arterial trees (L and R)
    # Note that this step is possible under the assumption that the arterial trees are disconnected
    # This happens in 95% of humans, but not in all of them
    # This assumption i smore than ok for CAT08 and any other dataset annotations probably,
    # but for an inferred arterial tree this should not be taken for granted.
    min_distance_between_two_trees = 2.5 #mm
    db = DBSCAN(eps=min_distance_between_two_trees, min_samples=10).fit(vl[:,:3])
    tree1 = vl[numpy.argwhere(db.labels_==0).flatten(),:]
    d_ostium1 = cl[numpy.argwhere(db.labels_==0).flatten()]
    tree2 = vl[numpy.argwhere(db.labels_==1).flatten(),:]
    d_ostium2 = cl[numpy.argwhere(db.labels_==1).flatten()]

    fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
    ax.scatter(tree1[:,0], tree1[:,1], tree1[:,2], s=10*tree1[:,3], c=d_ostium1, cmap="plasma")
    ax.scatter(tree2[:,0], tree2[:,1], tree2[:,2], s=10*tree2[:,3], c=d_ostium2, cmap="rainbow")
    plt.show()

    # now, it only remains to create the final graph




   
        






