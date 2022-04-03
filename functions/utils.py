#!/usr/bin/env python3
import sys
import os
import numpy as np
import math
from colormap import rgb2hex
from colorutils import Color

# Logger that duplicates output to terminal and to file
# Not portable on Windows though...
class Logger(object):

    def __init__(self,logfile):
        import warnings
        warnings.filterwarnings("default")

        # Works but we loose colors in the terminal
        import subprocess
        self.tee = subprocess.Popen(["tee", logfile], stdin=subprocess.PIPE)
        os.dup2(self.tee.stdin.fileno(), sys.stdout.fileno())
        os.dup2(self.tee.stdin.fileno(), sys.stderr.fileno())

    def __del__(self):
        self.tee.stdin.close()
        sys.stdout.close()
        sys.stderr.close()


# Helper function for filenames
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'k', 'M', 'B', 'T'][magnitude])

# Given vectors of x and y coordinates returns a color gradient where x ranges over
# red and y ranges over red, resulting in colors from black to purple.
def euclidean_color(x, y):
    x = (x - min(x))/(max(x) - min(x))
    y = (y - min(y))/(max(y) - min(y))
    return [rgb2hex(int(255*xi), 0, int(255*yi)) for xi, yi in zip(x,y)]

# Given vectors of x and y coordinates and the center point returns colors by angle with
# contour lines every 5% of the max radius
def radial_color(x, y, x0=0, y0=0):
    spatial_coords_polar = np.array([np.arctan2(y - y0, x - x0), np.sqrt((y - y0)**2 + (x - x0)**2)]).T

    max_r = max(spatial_coords_polar[:,1])
    def polar_rainbow(theta, r):
        if int(100*r/max_r) % 10 == 0:
            return '#000000'
        else:
            return Color(hsv=(360*(theta+math.pi)/(2*math.pi), 0.8, 0.8)).hex

    polar_rainbow_vec = np.vectorize(polar_rainbow)
    return polar_rainbow_vec(spatial_coords_polar[:,0], spatial_coords_polar[:,1])
