"""Here are defined the base classes that will hold the images and their
metadata.
Standardized methods are defined here to allow for easy access to the
images and their metadata.
"""

import os
import numpy

#####################
# CT/CCTA BASIC IMAGE
#####################

class ImageCT(object):
    """Base class for CT/CCTA single images.
    """
    data: numpy.ndarray # dimensions: (x, y, z) in RAS coordinate system
    origin: numpy.ndarray # dimensions: (x, y, z) in RAS coordinate system
    spacing: numpy.ndarray # dimensions: (x, y, z) in RAS coordinate system
    name: str # Name of the image


class ImageSequenceCT(object):
    pass
