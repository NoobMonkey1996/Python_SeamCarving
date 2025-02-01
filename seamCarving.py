# USAGE:
# python seam_carving.py (-resize | -remove) -im IM -out OUT [-mask MASK]
#                        [-rmask RMASK] [-dy DY] [-dx DX] [-vis] [-hremove] [-backward_energy]
# Examples:
# python seam_carving.py -resize -im demos/ratatouille.jpg -out ratatouille_resize.jpg 
#        -mask demos/ratatouille_mask.jpg -dy 20 -dx -200 -vis
# python seam_carving.py -remove -im demos/eiffel.jpg -out eiffel_remove.jpg 
#        -rmask demos/eiffel_mask.jpg -vis

import argparse
import numpy as np
import cv2 as cv
from numba import jit
from scipy import ndimage as ndim


# to create features: Resizing, Object removal, and Both in combination

