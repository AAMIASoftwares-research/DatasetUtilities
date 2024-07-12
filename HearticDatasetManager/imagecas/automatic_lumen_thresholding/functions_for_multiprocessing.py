import os
import numpy
from ..image import ImagecasImageCT, ImagecasLabelCT
from scipy.ndimage import binary_erosion, gaussian_filter
import nibabel
from sklearn.cluster import DBSCAN

INTENSITIES_DISTR_DICT = {
    0: [238., 301., 364.],
    1: [241., 319., 398.],
    2: [246., 331., 420.],
    3: [245., 334., 426.],
    4: [241., 331., 426.],
    5: [237., 325., 421.],
    6: [230., 320., 416.],
    7: [224., 315., 412.],
    8: [216., 310., 409.],
    9: [210., 305., 404.],
    10: [204., 300., 398.],
    11: [199., 296., 393.],
    12: [193., 290., 390.],
    13: [189., 286., 386.],
    14: [185., 283., 384.],
    15: [180., 278., 379.],
    16: [175., 274., 375.],
    17: [172., 271., 373.],
    18: [170., 268., 371.],
    19: [169., 266., 369.],
    20: [168., 265., 368.],
    21: [166., 263., 366.],
    22: [164., 261., 364.],
    23: [163., 259., 362.],
    24: [162., 257., 361.],
    25: [161., 256., 360.],
    26: [160., 256., 360.],
    27: [160., 256., 361.],
    28: [160., 255., 360.],
    29: [161., 255., 361.],
    30: [161., 254., 362.],
    31: [160., 253., 361.],
    32: [160., 252., 359.],
    33: [159., 251., 357.],
    34: [158., 250., 355.],
    35: [158., 248., 354.],
    36: [157., 248., 355.],
    37: [158., 249., 355.],
    38: [159., 248., 356.],
    39: [158., 249., 357.],
    40: [158., 249., 358.],
    41: [160., 251., 359.],
    42: [161., 252., 363.],
    43: [162., 253., 365.],
    44: [162., 253., 364.],
    45: [163., 254., 365.],
    46: [164., 254., 364.],
    47: [165., 255., 363.],
    48: [166., 254., 362.],
    49: [166., 254., 361.],
    50: [165., 253., 361.],
    51: [165., 253., 361.],
    52: [165., 252., 360.],
    53: [164., 250., 358.],
    54: [163., 248., 357.],
    55: [162., 246., 354.],
    56: [161., 245., 353.],
    57: [160., 244., 352.],
    58: [160., 244., 352.],
    59: [160., 244., 351.],
    60: [160., 242., 349.],
    61: [159., 242., 347.],
    62: [159., 240., 346.],
    63: [159., 240., 345.],
    64: [159., 239., 345.],
    65: [160., 239., 344.],
    66: [158., 237., 342.],
    67: [157., 236., 341.],
    68: [156., 234., 339.],
    69: [154., 233., 338.],
    70: [153., 232., 337.],
    71: [153., 232., 337.],
    72: [153., 231., 336.],
    73: [153., 231., 336.],
    74: [152., 230., 336.],
    75: [153., 231., 337.],
    76: [152., 230., 337.],
    77: [152., 230., 336.],
    78: [151., 229., 336.],
    79: [151., 229., 336.],
    80: [150., 228., 336.],
    81: [149., 226., 335.],
    82: [148., 225., 333.],
    83: [147., 225., 332.],
    84: [146., 223., 330.],
    85: [145., 222., 330.],
    86: [144., 222., 329.],
    87: [144., 221., 328.],
    88: [144., 221., 328.],
    89: [145., 222., 327.],
    90: [145., 222., 328.],
    91: [144., 221., 328.],
    92: [142., 220., 327.],
    93: [142., 219., 326.],
    94: [140., 219., 324.],
    95: [140., 217., 323.],
    96: [140., 217., 322.],
    97: [139., 217., 321.],
    98: [139., 217., 321.],
    99: [139., 216., 321.],
    100: [139., 216., 320.],
    101: [136., 214., 317.],
    102: [134., 212., 318.],
    103: [133., 213., 318.],
    104: [135., 214., 319.],
    105: [135., 214., 321.],
    106: [136., 214., 321.],
    107: [134., 214., 321.],
    108: [134., 214., 321.],
    109: [134., 216., 322.],
    110: [133., 216., 321.],
    111: [132., 214., 321.],
    112: [131., 214., 321.],
    113: [130., 214., 322.],
    114: [130., 214., 323.],
    115: [129., 215., 324.],
    116: [131., 216., 325.],
    117: [130., 214., 324.],
    118: [130., 215., 325.],
    119: [131., 215., 326.],
    120: [131., 215., 326.],
    121: [133., 217., 326.],
    122: [134., 219., 329.],
    123: [136., 221., 331.],
    124: [137., 223., 334.],
    125: [139., 225., 336.],
    126: [140., 227., 340.],
    127: [142., 230., 344.],
    128: [144., 232., 348.],
    129: [146., 234., 350.],
    130: [148., 235., 348.],
    131: [147., 234., 346.],
    132: [145., 234., 345.],
    133: [146., 235., 345.],
    134: [146., 236., 344.],
    135: [146., 236., 345.],
    136: [148., 237., 345.],
    137: [147., 236., 341.],
    138: [145., 234., 340.],
    139: [145., 234., 341.],
    140: [142., 232., 340.],
    141: [142., 232., 339.],
    142: [143., 233., 341.],
    143: [144., 233., 342.],
    144: [142., 230., 340.],
    145: [139., 228., 340.],
    146: [138., 229., 339.],
    147: [137., 227., 338.],
    148: [136., 226., 340.],
    149: [135., 227., 339.],
    150: [138., 228., 338.],
    151: [137., 228., 338.],
    152: [134., 226., 338.],
    153: [131., 223., 335.],
    154: [128., 219., 330.],
    155: [126., 217., 328.],
    156: [125., 215., 327.],
    157: [122., 211., 325.],
    158: [119., 208., 323.],
    159: [117., 204., 322.],
    160: [114., 201., 318.],
    161: [111., 197., 311.],
    162: [109., 192., 302.],
    163: [106., 186., 295.],
    164: [102., 179., 287.],
    165: [ 99., 175., 277.],
    166: [ 98., 170., 267.],
    167: [ 92., 163., 258.],
    168: [ 89., 162., 253.],
    169: [ 86., 157., 246.],
    170: [ 87., 155., 240.],
    171: [ 82., 149., 232.],
    172: [ 83., 148., 230.],
    173: [ 85., 150., 233.],
    174: [ 85., 150., 231.],
    175: [ 84., 148., 231.],
    176: [ 80., 145., 227.],
    177: [ 83., 146., 228.],
    178: [ 84., 148., 230.],
    179: [ 82., 146., 224.],
    180: [ 80., 141., 214.],
    181: [ 73., 130., 201.],
    182: [ 68., 125., 188.],
    183: [ 65., 120., 181.],
    184: [ 64., 117., 176.],
    185: [ 68., 118., 177.],
    186: [ 66., 118., 177.],
    187: [ 67., 118., 176.],
    188: [ 64., 116., 177.],
    189: [ 68., 121., 184.],
    190: [ 70., 124., 188.],
    191: [ 67., 120., 187.],
    192: [ 72., 124., 187.],
    193: [ 77., 132., 192.],
    194: [ 84., 139., 202.],
    195: [ 86.,  144.5, 210. ],
    196: [ 86., 148., 211.],
    197: [ 76., 137., 201.],
    198: [ 71.25, 131.,   201.75],
    199: [ 66., 129., 189.],
    200: [ 69., 123., 180.],
    201: [ 61., 115., 173.],
    202: [ 65., 123., 180.],
    203: [ 65., 126., 185.],
    204: [ 68., 131., 188.],
    205: [ 65., 125., 185.],
    206: [ 80., 135., 195.],
    207: [ 76., 130., 201.],
    208: [ 70., 128., 185.],
    209: [ 68., 118., 166.],
    210: [ 75.25, 131.,   196.  ],
    211: [ 62.25, 140.,   227.  ],
    212: [ 56.5, 129.,  228. ],
    213: [ 59., 128., 214.],
    214: [ 48., 123., 215.],
    215: [ 90.75, 191.,   255.25],
    216: [108.25, 195.,   253.  ],
    217: [106., 179., 236.],
    218: [ 88.5, 155.,  225.5],
    219: [ 79.,   148.5,  224.75],
    220: [ 76.,   141.,   217.75],
    221: [ 90., 166., 222.],
    222: [120., 168., 211.],
    223: [ 86.25, 145.,   203.75],
    224: [ 68.25, 134.5,  241.75],
    225: [ 57.,  108.5, 203. ],
    226: [ 47.5,  81.,  149.5],
    227: [ 12.5,   72.,   121.75],
    228: [ 13.75,  72.5,  128.75],
    229: [ 23.,   69.,  104.5],
    230: [ 34.,   72.,  123.5],
    231: [ 43.,  67., 103.],
    232: [20.75, 47.,   85.25],
    232: [20.75, 47.,   85.25],
    233: [ 11.75,  48.5,  127.75],
    234: [ 14.,  55., 133.],
    235: [ 33.5,  72.,  139. ],
    236: [ 15.,  49., 123.],
    237: [-13.5,   38.5,  104.25],
    238: [-42.5,  31.,   86. ],
    239: [-122.,    69.,   141.5],
    240: [-101.75,  105.,    178.  ],
    241: [129.,  155.,  186.5],
    242: [120.,  169.,  207.5],
    243: [113., 148., 175.]
}

def get_histogram(image_file_path, label_file_path) -> list[float]:
        image = ImagecasImageCT(image_file_path)
        label = ImagecasLabelCT(label_file_path)
        pixel_intensities_histogram = [0]*3001 # from -1000 to 2000 included
        where = numpy.argwhere(label.data > 0)
        for x, y, z in where:
            if (image.data[x, y, z] >= -1000) and (image.data[x, y, z] <= 2000):
                pixel_intensities_histogram[image.data[x, y, z]+1000] += 1
        return pixel_intensities_histogram



def make_wall_lumen_label(image_file_path, label_file_path, save_path, lumen_thresh=150, lumen_label=2, wall_label=1, null_label=0, intensities_distr_dict=None):
    image = ImagecasImageCT(image_file_path)
    label = ImagecasLabelCT(label_file_path)
    # apply smoothing to the image so that the lumen is more uniform and less dependent on the noise
    image.data = gaussian_filter(image.data, sigma=1)
    # create the new label file, that has 0 and 1 as labels for Null and Wall
    new_label = ImagecasLabelCT(label_file_path)
    # outer layer erosion
    new_label.data = binary_erosion(label.data.astype(bool), iterations=1)
    new_label.data = new_label.data.astype(int)
    new_label.data += label.data
    # now, erode the lumen label so that a pixel does not have neighbours of Null label
    # in the slice
    where = numpy.argwhere(new_label.data == lumen_label)
    for x, y, z in where:
        if null_label in new_label.data[x-1:x+2, y-1:y+2, z-1:z+2]:
            new_label.data[x, y, z] = wall_label
    
    if intensities_distr_dict is not None:
        # if the intensities dictionary is given (key 0 at the highest z, 1 at z_max-1, etc.)
        # (values list of 25th, 50th, 75th percentiles)
        # we will use the 25th percentile as the threshold for the lumen if it is lower than the given threshold
        where = numpy.argwhere(new_label.data == lumen_label)
        z_min, z_max = numpy.min(where[:, 2]), numpy.max(where[:, 2])
        low_thresh = 60
        for x, y, z in where:
            z_norm = z_max - z
            if z_norm in intensities_distr_dict:
                is_lumen_condition = (image.data[x, y, z] >= intensities_distr_dict[z_norm][0] or image.data[x, y, z] >= lumen_thresh) and image.data[x, y, z] >= low_thresh
            else:
                closest_idx_ = min(intensities_distr_dict.keys(), key=lambda x: abs(x - z_norm))
                is_lumen_condition = (image.data[x, y, z] >= intensities_distr_dict[closest_idx_][0] or image.data[x, y, z] >= lumen_thresh) and image.data[x, y, z] >= low_thresh
            if not is_lumen_condition:
                new_label.data[x, y, z] = wall_label
    else:
        # intensity-based erosion of the wall label
        # only in the locations where the previously found lumen is
        # after finding the pixels we have to search,
        # the lumen label is reset to be all just wall label
        where = numpy.argwhere(new_label.data == lumen_label)
        new_label.data = new_label.data.astype(bool).astype(int)
        new_label.data[where[:, 0], where[:, 1], where[:, 2]] = numpy.where(
            image.data[where[:, 0], where[:, 1], where[:, 2]] >= lumen_thresh,
            lumen_label,
            new_label.data[where[:, 0], where[:, 1], where[:, 2]]
        )
    if 0:
        # if a vessel wall pixel is surrounded, on the slice, 
        # by lumen pixels, it is a lumen pixel
        # this is to prevent holes in the lumen label
        for _ in range(3):
            # XY plane
            where = numpy.argwhere(new_label.data == wall_label)
            for x, y, z in where:
                square_ = new_label.data[x-1:x+2, y-1:y+2, z].copy()
                square_[1, 1] = 100
                if wall_label not in square_:
                    new_label.data[x, y, z] = lumen_label  
    else:
        # use DBSCAN to keep all the vessel wall pixels
        # connected with each other
        # if there are unconnected vessel wall pixels,
        # they are lumen pixels
        where = numpy.argwhere(new_label.data == wall_label)
        dbscan = DBSCAN(eps=numpy.sqrt(2)+0.00001, min_samples=5) # also include diagonal connections
        dbscan.fit(where)
        labels = dbscan.labels_
        for i, label in enumerate(labels):
            if label == -1:
                new_label.data[where[i, 0], where[i, 1], where[i, 2]] = lumen_label
    # return or save with nibabel to nii.gz
    if save_path == "":
        return new_label
    nib_label = nibabel.load(label_file_path)
    nib_label_new = nibabel.Nifti1Image(
        numpy.flip(new_label.data, axis=0).astype(numpy.uint8),
        nib_label.affine
    )
    print("Saved: ", os.path.basename(save_path))
    nibabel.save(nib_label_new, save_path)