import os
import matplotlib.pyplot as plt
import utilsMolinette as util
""" ### mATTACK PLAN ###
Molinette data are very "sparse", incomplete, sometimes slightly wrong and sometimes
completely wrong. There is very little structure, if not in the pazNAME files, which
should be anyway integrated with previously obtained centerlines.
Centerlines are more extended than CAT08, however they are not super-complete and
often times they miss not only small branches, but also big branches.

Optimal pipeline:
    1. For each patient, check all centerlines in Slicer, and add missing branches/
       correct slightly wrong branches.
       (Adding more branches should be done also in CAT08 to say the truth)
    2. Create ostium-endpoints filaments with common points if possible (easier to create a graph).
    3. Create the complete graphs.
       Having a super-complete graph is not strictly necessary for an iterative centerline tracker,
       however it is very necessary for graphs neural networks.

Bottom line, a lot of manual work is still needed. And then there have to be a way to create
graphs from such a sparse data representation.
"""

base_path = os.path.normpath(
    "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\Molinette\\Medis_Centerlines_STLs"
)


PATIENTS = [
    'AC', 'BC', 'BF', 'BG', 'BM', 'BR', 'CA', 'CC', 'CN', 'CV', 'DRV',
    'DTA', 'EFA', 'FPV', 'GG', 'GGI', 'GL', 'GP', 'GR', 'GRO', 'IA', 'ID',
    'LA', 'LG', 'LMG', 'LU', 'MA', 'MG', 'MM', 'MT', 'NR', 'PEG', 'PS',
    'RI', 'SA', 'SAN', 'SGF', 'TL', 'ZG', 'ZGI'
]
ARTERIES_TO_BE_MANUALLY_REVISED = { # "patient": ["artery", "list"]
    "BF": ["RCA"],
    "BR": ["RCA"],
    "EFA": ["LCX", "LAD"],
    "GGI": ["LCX"],
    "ID": ["LAD", "LCX"], # ostium
    "LMG":["RCA"], # ostium, try to fix it
    "MT": ["LAD", "LCX", "RCA"], # everything is shifted - can be corrected probably
    "NR": ["LAD", "LCX", "RCA"],  # everything is shifted + other errors
    "RI": ["RCA"], # distal

}
ARTERIES_WITH_IRREPARABLE_ISSUES = { # "patient": ["artery", "list"]
    "DTA":["LAD", "LCX"],
    "GP": ["LCX"],
    "GR": ["LCX"],
    "LA": ["RCA"],
    "RI": ["LCX"]

}
ARTERY_NAMES = ["RCA", "LAD", "LCX", "D1"]
# KEEP IN MIND! 
#   The name of the arteries IS NOT INDICATIVE of the real artery classification name.
#   Also, the tracked arteries often enter the ostium too much!

if __name__ == "__main__":

    if 0:
        ### first try
        ''' code does not work, it worked but i changes some variables
        centerlines_list = []
        for p in [os.path.join(base_path, PATIENTS[i]) for i in len(PATIENTS)]:
            c = util.importCenterline(
                os.path.join(base_path, p, "centerplanes.csv")
            )
            centerlines_list.append(c)
            if 0:
                util.centerlineToMarkerListSlicer(c, os.path.join(base_path, p, "centerplanes.SlicerView.fcsv"))
            if 0:
                util.plotCenterlines3D(c, title=p)

        centerlines_list = util.splitOverlappingCenterlines(centerlines_list[1])
        
        ax = plt.subplot(111, projection="3d")
        for i, l in enumerate(centerlines_list):
            ax.plot(l[:,0]+i,l[:,1],l[:,2], ".-")
        plt.show()
        '''

    ### For each patient, save in a list all "single" centerlines
    # a single centerline is one that goes from the ostium to a single endpoint,
    # without any intersection or bifurcation
    
    
