import numpy as np
from matplotlib import pyplot as plt
import anndata
import math
from colormap import rgb2hex
from scipy.stats import pearsonr
from datetime import datetime
from colorutils import Color
from distances import sparse_euclidean_dist, calculate_all_distances, calculate_subset_distances, calculate_embedding_distances, compute_small_umap_dists, compute_large_umap_dists

# Visualize barcodes in a dataset by plotting
#   i)   a heatmap of barcode size,
#   ii)  a histogram of the total number of reads
#   iii) a histogram of number of non-zero elements
# barcodes: AnnData object with barcodes in X and spatial coordinates in
#           obs.x and obs.y
# suptitle: Optional suptitle for the plots
def plot_barcodes(barcodes, suptitle=False):
    fig = plt.figure(figsize=(17,5.5))
    if suptitle != False:
        plt.suptitle(suptitle)
    gs = fig.add_gridspec(1,17)
    barcode_sizes = np.zeros(barcodes.shape[0])
    barcode_length = np.zeros(barcodes.shape[0])
    for j in range(barcodes.shape[0]):
        barcode_sizes[j] = np.sum(barcodes.X[j,:].data)
        barcode_length[j] = len(barcodes.X[j,:].data)

    if len(barcode_sizes[barcode_sizes == 0]) > 0:
        print("Warning {} beads have 0 reads.".format(len(barcode_sizes[barcode_sizes == 0])))
    plt.subplot(gs[:,:7])
    plt.scatter(barcodes.obs.x, barcodes.obs.y, c=barcode_sizes, s=4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    if max(barcode_sizes) > 10**4:
        plt.colorbar(format='%.1e')
    else:
        plt.colorbar()
    plt.title('Reads by Spatial Position')

    plt.subplot(gs[:,7:12])
    plt.hist(barcode_sizes)
    plt.title('Reads per Bead')
    plt.xlabel('Total Reads')
    plt.ylabel('# Barcodes')
    if max(barcode_length) > 10**4:
        plt.ticklabel_format(axis="x", style="sci")

    plt.subplot(gs[:,12:])
    plt.hist(barcode_length)
    plt.title('Non-Zero Colonies per Bead')
    plt.xlabel('Non-Zero Elements')
    plt.ylabel('# Barcodes')

    plt.tight_layout()
    plt.show()

# Plot n_comps randomly selected pairwise distances between beads stored in
# an anndata object. The first subfigure shows all distances while the
# second only shows distances within the provided threshold.
# barcodes: Anndata object with barcodes in X and spatial coordinates in obs.x
#           and obs.y
# thresh: Cut-off for ground truth bead distances in second subplot
# n_comps: Number of bead pairs considered if dense is not true
# all: If true calculate distances between all pairs
# verbose: If true print timing information
# returns: the array of distances calculated for the plot
def plot_pairwise_distances (barcodes, thresh, n_comps=10**4, all=False, verbose=False):
    start = datetime.now()
    if all:
        dists = calculate_all_distances(barcodes)
    else:
        dists = calculate_subset_distances(barcodes, n_comps)

    fig = plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.scatter(dists[:,1], dists[:,0], alpha=0.1, rasterized=True)
    plt.xlabel('Spatial Distance')
    plt.ylabel('Barcode Distance')

    plt.subplot(1, 2, 2)
    plt.scatter(dists[:,1], dists[:,0], alpha=1, rasterized=True)
    plt.xlabel('Spatial Distance')
    plt.ylabel('Barcode Distance')
    plt.xlim([0, thresh])
    plt.title("$x \in [0, {}]$".format(thresh))
    if verbose:
        print("Elapsed time: {}".format(datetime.now() - start))

    return dists

# Find and compare the distances in the original space and a projected space.
# barcodes: Anndata object with barcodes in X and spatial coordinates in obs.x
#           and obs.y
# n_colony_barcodes: Array of number of colony barcodes in projected space
# n_comps: Number of bead pairs considered
# method: Method for binning barcodes into projected space. If uniform
#         all bins will be roughly the same size. If normal bin size will
#         roughly follow a normal distribution.
# suptitle: Optional suptitle for the plot
# verbose: If true print timing information
# returns: the array of distances calculated for the plot
def plot_projected_distances(barcodes, n_colony_barcodes, n_comps=10**4, method='uniform', suptitle=False, verbose=False):
    start = datetime.now()
    dists = np.zeros((n_comps, len(n_colony_barcodes) + 1))
    for k in range(n_comps):
        i = np.random.randint(low=0, high=barcodes.shape[0])
        j = np.random.randint(low=0, high=barcodes.shape[0])
        dists[k,0] = sparse_euclidean_dist(barcodes, i, j)
        for n in range(len(n_colony_barcodes)):
            if method == 'normal':
                dists[k,n+1] = sparse_euclidean_dist(barcodes, i, j, mod=n_colony_barcodes[n], method='normal')
            else:
                dists[k,n+1] = sparse_euclidean_dist(barcodes, i, j, mod=n_colony_barcodes[n])


    fig = plt.figure(figsize=(5*len(n_colony_barcodes),5))
    if suptitle != False:
        plt.suptitle(suptitle)
    for n in range(len(n_colony_barcodes)):
        plt.subplot(1, len(n_colony_barcodes), n+1)
        plt.scatter(dists[:,0], dists[:,n+1], alpha=0.1, rasterized=True)
        plt.xlabel('Original Distance')
        plt.ylabel('Projected Distance')
        plt.title("projected barcodes = {:.0e}, (corr = {:.2f})".format(n_colony_barcodes[n], np.corrcoef(dists[:,0], dists[:,n+1])[0,1]))
        if verbose:
            print("Elapsed time: {}".format(datetime.now() - start))

    return dists


# Plot a UMAP compared to the underlying spatial locations, coloring both
# embeddings with a gradient that varies in both x and y
# adata: Anndata object with umap embedding in obsm['X_umap'] and spatial
#        coordinates in obs.x and obs.y
# suptitle: Optional suptitle for the subplots
def plot_umap(adata, suptitle=None):
    fig = plt.figure(figsize=(10,5))
    if suptitle != None:
        plt.suptitle(suptitle)

    if 'spatial_color' not in list(adata.obs.columns):
        adata.obs['spatial_color'] = adata.obs.apply(lambda x: rgb2hex(int(255*x['x']), 0, int(255*x['y'])), axis=1)

    plt.subplot(1,2,1)
    plt.title('Spatial Locations')
    plt.axis('off')
    plt.scatter(adata.obs.x, adata.obs.y, c=adata.obs.spatial_color, s=4, rasterized=True)

    plt.subplot(1,2,2)
    plt.title('UMAP')
    plt.axis('off')
    plt.scatter(adata.obsm['X_umap'][:,0], adata.obsm['X_umap'][:,1], c=adata.obs.spatial_color, s=4, rasterized=True)


# Plot and calculate the correlation for distances between random pairs of beads
# in the underlying space and in the UMAP embedding. The first subplot shows the correlation
# for all distances while the second shows only distances near the edge
def plot_correlations(adata, suptitle=False, edge_thresh=0.05, dist_thresh=0.1, n_reps=10**3, savepath=None):
    # Move the umap coordinates to obs
    adata.obs['umap_x'] = adata.obsm['X_umap'][:,0]
    adata.obs['umap_y'] = adata.obsm['X_umap'][:,1]

    # Generate the random distances from the whole field
    dists = calculate_embedding_distances(adata, n_reps)

    # Generate the distances for points along the edge
    edge_adata = adata[((adata.obs.x < edge_thresh) | (adata.obs.x > (1-edge_thresh))) & ((adata.obs.y < edge_thresh) | (adata.obs.y > (1-edge_thresh)))]
    edge_dists = calculate_embedding_distances(edge_adata, dist_thresh=dist_thresh)

    # Plot a two panel figure showing (left) the distance between random pairs of points in the ground truth (x)
    # versus the UMAP embedding 9y) and (right) the distance between pairs of points near the edges close to each
    # other
    fig = plt.figure(figsize=(10,5))
    if suptitle != False:
        plt.suptitle(suptitle)
    plt.subplot(1,2,1)
    plt.scatter(dists[:,0], dists[:,1], rasterized=True)
    plt.xlabel('Ground Truth Distance')
    plt.ylabel('UMAP Distance')
    plt.title("All points, corr = {:.4f}".format(pearsonr(dists[:,0], dists[:,1])[0]))

    plt.subplot(1,2,2)
    plt.scatter(edge_dists[:,0], edge_dists[:,1], rasterized=True)
    plt.xlabel('Ground Truth Distance')
    plt.ylabel('UMAP Distance')
    plt.title("Edges, corr = {:.4f}".format(pearsonr(edge_dists[:,0], edge_dists[:,1])[0]))
    
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()
    return dists
        
# Compare a UMAP embedding to the ground truth spatial distances by finding the distances in each space
# between pairs or points or the areas between triplets of points. Plots a scatter plot of all distances/areas
# as well as only those where points as within a small number of sigma of each other as well as returns the
# correlation.
# umap_coords: Nx2 array of umap coordinates
# spatial_coords: Nx2 array with ground truth spatial distances
# sigma: Standard deviation of Gaussian kernel used to generate data
# x0, y0: Centre of the annulus of considered points
# r0, r1: Inner and outer radius of the annulus of considered points
# comp: Method for comparing points. Use `dist` to compare pairwise distances and `area` to compute the area of
#       a triangle formed by three points
# n_sigma_thresh: Maximum distance between points shown on the x-axis of the second plot, specified in
#                 number of sigmas.
# max_plot: Maximum number of coordinates to plot for the plots showing points on the UMAP/ground truth
#           embeddings.
def compare_umap_dists_annulus(umap_coords, spatial_coords, sigma, r0, r1, x0=0.5, y0=0.5, comp="dist", n_sigma_thresh=10, max_plot = 10**5):
    # Check if the comparison method is valid and set the plot label variable
    if comp == 'dist':
        dist_label = 'Pairwise Distances'
    elif comp == 'area':
        dist_label = 'Triangle Areas'
    else:
        print('Invalid comparison type "{}"'.format(comp))
        return

    # Convert UMAP distances to polar, centering around (x0, y0)
    spatial_coords_polar = np.array([np.arctan2(spatial_coords[:,1] - y0, spatial_coords[:,0] - x0),
                                 np.sqrt((spatial_coords[:,1] - y0)**2 + (spatial_coords[:,0] - x0)**2)]).T
    # Create a filter for points within the annulus
    annulus_filter = (spatial_coords_polar[:,1] >= r0) & (spatial_coords_polar[:,1] <= r1)

    # Create the plot
    fig = plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    if len(spatial_coords) > max_plot:
        plt.scatter(spatial_coords[:,0][:max_plot], spatial_coords[:,1][:max_plot], color='lightgrey', s=4)
        plt.scatter(spatial_coords[annulus_filter][:,0][:max_plot], spatial_coords[annulus_filter][:,1][:max_plot],s=4)
    else:
        plt.scatter(spatial_coords[:,0], spatial_coords[:,1], color='lightgrey', s=4)
        plt.scatter(spatial_coords[annulus_filter][:,0], spatial_coords[annulus_filter][:,1],s=4)
    plt.axis('off')
    plt.title('Points in Annulus - Ground Truth')

    plt.subplot(2,2,2)
    if len(spatial_coords) > max_plot:
        plt.scatter(umap_coords[:,0][:max_plot], umap_coords[:,1][:max_plot], color='lightgrey', s=4)
        plt.scatter(umap_coords[annulus_filter][:,0][:max_plot], umap_coords[annulus_filter][:,1][:max_plot],s=4)
    else:
        plt.scatter(umap_coords[:,0], umap_coords[:,1], color='lightgrey', s=4)
        plt.scatter(umap_coords[annulus_filter][:,0], umap_coords[annulus_filter][:,1],s=4)
    plt.axis('off')
    plt.title('Points in Annulus - UMAP')

    plt.subplot(2,2,3)
    # Make the first plot by chossing a point from the filtered set and a point/two points from the full set 10^4 times
    large_dists = compute_large_umap_dists(umap_coords, spatial_coords, annulus_filter, comp)
    plt.scatter(large_dists[:,1], large_dists[:,0], s=4)
    plt.ylabel('UMAP')
    plt.xlabel('Ground truth')
    d0 = large_dists[:,0]
    d1 = large_dists[:,1]
    filt = ~(np.isnan(d0) | np.isnan(d1)) 
    plt.title('Large {} (corr={:.2f})'.format(dist_label, pearsonr(d0[filt], d1[filt])[0]))

    # Create a second filter for points within the threshold of distance of the annulus, removing points
    # that will fail to be within the threshold for every point in the annulus.
    thresh = sigma*n_sigma_thresh
    large_annulus_filter = (spatial_coords_polar[:,1] >= (r0 - thresh)) & (spatial_coords_polar[:,1] <= (r1 + thresh))

    plt.subplot(2,2,4)
    small_dists = compute_small_umap_dists(umap_coords, spatial_coords, large_annulus_filter, annulus_filter, comp, thresh)
    plt.scatter(small_dists[:,1], small_dists[:,0], s=4)
    plt.ylabel('UMAP')
    plt.xlabel('Ground truth $[0, {}\sigma]$'.format(n_sigma_thresh))
    d0 = small_dists[:,0]
    d1 = small_dists[:,1]
    filt = ~(np.isnan(d0) | np.isnan(d1)) 
    plt.title('Small {} (corr={:.2f})'.format(dist_label, pearsonr(d0[filt], d1[filt])[0]))
    
    
# Compare a UMAP embedding to the ground truth spatial distances by finding the distances in each space
# between pairs or points or the areas between triplets of points. Plots a scatter plot of all distances/areas
# as well as only those where points as within a small number of sigma of each other as well as returns the
# correlation.
# umap_coords: Nx2 array of umap coordinates
# spatial_coords: Nx2 array with ground truth spatial distances
# sigma: Standard deviation of Gaussian kernel used to generate data
# x0, y0: Centre of the annulus of considered points
# max_plot: Maximum number of coordinates to plot for the plots showing points on the UMAP/ground truth
#           embeddings.
def compare_umap_dists(umap_coords, spatial_coords, sigma, x0=0.5, y0=0.5, max_plot = 10**5):
    # Convert UMAP distances to polar, centering around (x0, y0)
    spatial_coords_polar = np.array([np.arctan2(spatial_coords[:,1] - y0, spatial_coords[:,0] - x0),
                                 np.sqrt((spatial_coords[:,1] - y0)**2 + (spatial_coords[:,0] - x0)**2)]).T
    
    max_r = max(spatial_coords_polar[:,1])

    # Create a function to color points by theta and add isolines every 5% of the radius
    def polar_rainbow(theta, r): 
        if int(100*r/max_r) % 10 == 0:
            return '#000000'
        else:
            return Color(hsv=(360*(theta+math.pi)/(2*math.pi), 0.8, 0.8)).hex

    polar_rainbow_vec = np.vectorize(polar_rainbow)
    colors = polar_rainbow_vec(spatial_coords_polar[:,0], spatial_coords_polar[:,1])
    
    # Create the plot
    fig = plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    if len(spatial_coords) > max_plot:
        plt.scatter(spatial_coords[colors != '#000000'][:,0][:max_plot], spatial_coords[colors != '#000000'][:,1][:max_plot], 
                    c=colors[colors != '#000000'][:max_plot], s=2, rasterized=True)
        plt.scatter(spatial_coords[colors == '#000000'][:,0][:max_plot], spatial_coords[colors == '#000000'][:,1][:max_plot], 
                    c='#000000', s=2, rasterized=True)
    else:
        plt.scatter(spatial_coords[colors != '#000000'][:,0], spatial_coords[colors != '#000000'][:,1], 
                    c=colors[colors != '#000000'], s=2, rasterized=True)
        plt.scatter(spatial_coords[colors == '#000000'][:,0], spatial_coords[colors == '#000000'][:,1], 
                    c='#000000', s=2, rasterized=True)
    plt.axis('off')
    plt.title('Ground Truth')

    plt.subplot(2,2,2)
    if len(spatial_coords) > max_plot:
        plt.scatter(umap_coords[colors != '#000000'][:,0][:max_plot], umap_coords[colors != '#000000'][:,1][:max_plot],
                    c=colors[colors != '#000000'][:max_plot], s=2, rasterized=True)
        plt.scatter(umap_coords[colors == '#000000'][:,0][:max_plot], umap_coords[colors == '#000000'][:,1][:max_plot],
                    c='#000000', s=2, rasterized=True)
    else:
        plt.scatter(umap_coords[colors != '#000000'][:,0], umap_coords[colors != '#000000'][:,1], c=colors[colors != '#000000'], s=2, rasterized=True)
        plt.scatter(umap_coords[colors == '#000000'][:,0], umap_coords[colors == '#000000'][:,1], c='#000000', s=2, rasterized=True)
    plt.axis('off')
    plt.title('UMAP')



# Get the radial rainbow colormap from coordinates
def radial_rainbow_colormap(coords, center=[0.5,0.5]):
    # Convert to polar coordinates, centering around center
    coords_polar = np.array([np.arctan2(coords[:,1] - center[1], coords[:,0] - center[0]),
                                 np.sqrt((coords[:,1] - center[1])**2 + (coords[:,0] - center[0])**2)]).T

    max_r = max(coords_polar[:,1])

    # Create a function to color points by theta and add isolines every 5% of the radius
    def polar_rainbow(theta, r):
        if int(100*r/max_r) % 10 == 0:
            return '#000000'
        else:
            return Color(hsv=(360*(theta+math.pi)/(2*math.pi), 0.8, 0.8)).hex

    polar_rainbow_vec = np.vectorize(polar_rainbow)

    return polar_rainbow_vec(coords_polar[:,0], coords_polar[:,1])


# This function is inspired from compare_umap_dists above, but made to be more general.
# Compare a reconstruction with the corresponding ground truth, using colors that identify point positions
# coords_gt is required to plot colors, for all values of which_plot
# coords_rec is not required when which_plot=0
# which_plot: 0: GT, 1: Rec, 2: GT + Rec
# center: center of the ground truth coordinates necessary for radial coloring.
def compare_gt_rec(coords_gt, coords_rec=None, which_plot=2, point_size=5, center=(0.5,0.5),
                   max_plot=10**5, title_gt="Ground truth", title_rec="Reconstruction",
                   hide_axes=True):

    # Get rainbow radial colormap
    colors = radial_rainbow_colormap(coords_gt,center)

    # Create the plot
    if which_plot == 2:
        fig = plt.figure(figsize=(20,10))
    else:
        fig = plt.figure(figsize=(10,10))

    if which_plot == 2:
        plt.subplot(1,2,1)
    if which_plot == 0 or which_plot == 2:
        if len(coords_gt) > max_plot:
            plt.scatter(coords_gt[colors != '#000000'][:,0][:max_plot], coords_gt[colors != '#000000'][:,1][:max_plot],
                        c=colors[colors != '#000000'][:max_plot], s=point_size)
            plt.scatter(coords_gt[colors == '#000000'][:,0][:max_plot], coords_gt[colors == '#000000'][:,1][:max_plot],
                        c='#000000', s=point_size)
        else:
            plt.scatter(coords_gt[colors != '#000000'][:,0], coords_gt[colors != '#000000'][:,1],
                        c=colors[colors != '#000000'], s=point_size)
            plt.scatter(coords_gt[colors == '#000000'][:,0], coords_gt[colors == '#000000'][:,1],
                        c='#000000', s=point_size)
        if hide_axes: plt.axis('off')
        plt.title(title_gt)

    if which_plot == 2:
        plt.subplot(1,2,2)
    if which_plot == 1 or which_plot == 2:
        if len(coords_gt) > max_plot:
            plt.scatter(coords_rec[colors != '#000000'][:,0][:max_plot], coords_rec[colors != '#000000'][:,1][:max_plot],
                        c=colors[colors != '#000000'][:max_plot], s=point_size)
            plt.scatter(coords_rec[colors == '#000000'][:,0][:max_plot], coords_rec[colors == '#000000'][:,1][:max_plot],
                        c='#000000', s=point_size)
        else:
            plt.scatter(coords_rec[colors != '#000000'][:,0], coords_rec[colors != '#000000'][:,1], c=colors[colors != '#000000'], s=point_size)
            plt.scatter(coords_rec[colors == '#000000'][:,0], coords_rec[colors == '#000000'][:,1], c='#000000', s=point_size)
        if hide_axes: plt.axis('off')
        plt.title(title_rec)
