""" Heartic Dataset Manager

This module helps working with the data available to the HEARTIC project
from the AAMIA Softwares research group.

Available datasets are:

* cat08
* asoca

while other datasets we are working on are:

* molinette
* monzino
* deepvesselnet

"""

from . import cat08, asoca, affine
__all__ = ['cat08', 'asoca', "affine"]