#!/usr/bin/env python3
import sys
import os
import numpy as np
import math
from colormap import rgb2hex
from colorutils import Color
from matplotlib import pyplot as plt

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

# Given the ground truth and aligned coordinates, produces three plots showing
# the ground truth, the reconstruction, and the reconstruction colored by distance
# to the ground truth.
# PARAMS:
# -- adata: anndata containing the ground truth coordinates in obs
# -- aligned_coords: reconstruction coordinates after the affine transformation
# -- dist_gt: distance to the ground truth for each bead
def ground_truth_comparison(adata, aligned_coords, dist_gt):
    fig = plt.figure(figsize=(12,4))
    gs = fig.add_gridspec(1,16)
    plt.subplot(gs[:,:5])
    plt.title('Ground Truth')
    plt.scatter(adata.obs.x, adata.obs.y, c=adata.obs.spatial_color, s=1)
    plt.axis('off')

    plt.subplot(gs[:,5:10])
    plt.title('Reconstruction')
    indices = np.arange(adata.shape[0])
    np.random.shuffle(indices)
    plt.scatter(aligned_coords[indices][:,0], aligned_coords[indices][:,1], c=adata[indices].obs.spatial_color, s=1)
    plt.axis('off')

    plt.subplot(gs[:,10:])
    plt.title('Accuracy (median = {:.2f} $\mu m$)'.format(np.median(dist_gt)))
    plt.scatter(aligned_coords[indices][:,0], aligned_coords[indices][:,1], c=dist_gt, s=1, vmax=100)
    plt.colorbar(label='Distance to Ground Truth ($\mu m$)', orientation = 'vertical')
    plt.axis('off')

    plt.show()

# Given a vector of distances to the ground truth and a maximum distance produces
# a complementary cummulative distribution plot with horizontal lines for the 50th
# and 90th percentiles.
def complementary_cummulative_distribution(dist_gt, max_dist):
    bins = np.arange(0, max_dist, 1)
    bins = np.append(bins, max(max(dist_gt), max_dist))
    hist = plt.hist(dist_gt, bins=bins, density=True, cumulative=True)
    vals = np.insert(hist[0],0,0)
    plt.close()

    fig = plt.figure(figsize=(4,4))
    plt.title('Complementary Cummulative Distribution ($P(x \leq d)$)')
    plt.plot(bins, 1 - vals)
    plt.ylabel('Proportion')
    plt.xlabel('Distance (d) to Ground Truth ($\mu m$)')

    plt.xlim([0,max_dist])
    plt.ylim([0,1])
    xticks = np.arange(0,max_dist+1,10)
    xlabels = ['{}+'.format(t) if t == max(xticks) else '{}'.format(t) if t % 20 == 0 else '' for t in xticks]
    plt.xticks(ticks=xticks, labels=xlabels)

    plt.axhline(0.1, c='k', linestyle='dashed', linewidth=0.5, zorder=0)
    plt.axhline(0.5, c='k', linestyle='dashed', linewidth=0.5, zorder=0)
    plt.show()
