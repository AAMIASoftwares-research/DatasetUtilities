import os
import utils.util as util

base_path = os.path.normpath(
    "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Workbench\\Stefanati_centerlines\\Data_centerline"
)

patients_list = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

if __name__ == "__main__":
    centerlines_list = []
    for p in patients_list:
        c = util.importCenterline(
            os.path.join(base_path, p, "centerplanes.csv")
        )
        centerlines_list.append(c)
        if 0:
            util.plotCenterlines3D(c, title=p)

    centerlines_list = util.splitOverlappingCenterlines(centerlines_list[0])