# Utility to annotate coronary ostia on a given image
# With matplotlib, user can click on the image to mark the coronary ostia
# A slider allow to go through slices of the image
# Another lider allows to change the axis of visualization of the image)

import os, sys
import numpy
import tkinter
from tkinter import filedialog
import SimpleITK
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import Button, RadioButtons


FRA = 200