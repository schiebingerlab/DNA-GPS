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


# Function to create a grid with n_big major and n_small minor steps along each axis
def _fancy_grid(ax_min, ax_max, n_big, n_small):
    step_big = (ax_max - ax_min)/n_big
    step_small = (ax_max - ax_min)/n_small

    ax = plt.gca()
    major_ticks = np.arange(ax_min, ax_max, step_big)
    minor_ticks = np.arange(ax_min, ax_max, step_small)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='major', alpha=0.8)
    ax.grid(which='minor', alpha=0.3)

# Function to visualize a colony grid both by presence/absence of bateria and by colony id
def colony_grid(colony_grid, beadwidth=10, step_um=500):
    n = colony_grid.shape[0]
    n_steps = int(np.ceil(beadwidth*n/step_um))
    n_big = n / (int(n/50)*10)

    fig = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('Bacteria Presence/Absence')
    plt.imshow(_color_by_thresh(colony_grid, 0))
    _fancy_grid(0, n, n_big, 5*n_big)
    plt.xticks(np.arange(0, n+1, n/n_steps), range(0, step_um*n_steps + 1, step_um))
    plt.yticks(np.arange(0, n+1, n/n_steps), range(0, step_um*n_steps + 1, step_um))
    plt.xlabel('Position $(\mu m)$')

    plt.subplot(1,2,2)
    plt.title('Bacteria by Colony')
    grid_copy = colony_grid.copy()
    grid_copy[grid_copy == 0] = float('nan')
    plt.imshow(grid_copy, cmap='hsv')
    _fancy_grid(0, n, n_big, 5*n_big)
    plt.xticks(np.arange(0, n+1, n/n_steps), range(0, step_um*n_steps + 1, step_um))
    plt.yticks(np.arange(0, n+1, n/n_steps), range(0, step_um*n_steps + 1, step_um))
    plt.xlabel('Position $(\mu m)$')

    plt.show()

# Convert an image specified by a grid to an RGB image colored by threshold where
# pixels above the threshold are red and cells below the threshold are blue.
def _color_by_thresh(img, thresh):
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))
    thresh = thresh

    def r_thresh(x):
        return 1 if x > thresh else 0

    def b_thresh(x):
        return 1 if x <= thresh else 0

    r_vec = np.vectorize(r_thresh)
    b_vec = np.vectorize(b_thresh)

    img_rgb[:,:,0] = r_vec(img)
    img_rgb[:,:,2] = b_vec(img)

    return img_rgb


# Given an anndata with a satellite barcode read matrix visualizes the reads from
# the first n_colonies colonies.
# PARAMS:
# -- adata: Anndata with satellite barcode read matrix
# -- n_colonies: number of colonies to plot
# -- edge_length_um: Length of a plot edge in um, usually n*beadwidth
# -- step_um: tick step in um
def satellite_barcode_counts(adata, n_colonies, edge_length_um, step_um=500):
    # Normalize data to be in [0,1]^2
    x = (adata.obs.x - min(adata.obs.x)) / (max(adata.obs.x) - min(adata.obs.x))
    y = (adata.obs.y - min(adata.obs.y)) / (max(adata.obs.y) - min(adata.obs.y))
    n_steps = int(np.ceil(edge_length_um/step_um))

    n_rows = np.ceil(n_colonies/3).astype(int)
    fig = plt.figure(figsize=(19,5*n_rows))
    for i in range(n_colonies):
        plt.subplot(n_rows,3,i + 1)
        plt.scatter(x, y, c=adata.X.A[:,i], s=1)
        plt.colorbar(label='Reads')
        plt.xticks(np.arange(0, (n_steps + 1)/n_steps, 1/n_steps), range(0, step_um*n_steps + 1, step_um))
        plt.yticks(np.arange(0, (n_steps + 1)/n_steps, 1/n_steps), range(0, step_um*n_steps + 1, step_um))
        plt.xlabel('Position ($\mu m$)')
    plt.show()

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
