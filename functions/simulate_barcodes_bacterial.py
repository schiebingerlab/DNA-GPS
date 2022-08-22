import numpy as np
import pandas as pd
import gc
import anndata
import math
import scipy
import scanpy
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from colormap import rgb2hex
from colorutils import Color
from tqdm import tqdm

# Create an n x n grid and initialize n_init colonies.
# Optionally provide a seed to numpy for reproducibility.
# Colonies are assigned an ID from 1 to n_init. Cells with
# no colony have a value of 0.
# If 'circle' is provided as the shape, bacteria will only
# be placed in a diameter n circle.
def init_grid(n, n_init, seed=0, shape='square'):
    grid = np.zeros((n,n))

    y,x = np.meshgrid(np.arange(0,n), np.arange(0,n))
    y = y.flatten()
    x = x.flatten()

    if shape == 'circle':
        in_circle = [np.sqrt((x[i]-n/2)**2 + (y[i]-n/2)**2) < n/2 for i in range(len(x))]
        x = x[in_circle]
        y = y[in_circle]

    indices = np.arange(len(x))

    if n_init > len(indices):
        print('Error: More colonies requested than grid squares')
        return

    np.random.seed(seed)
    np.random.shuffle(indices)
    for m in range(n_init):
        i = indices[m]
        grid[x[i], y[i]] = m + 1
    return grid

# Perform a single iteration of growth, returning the updated grid.
# Iterates colony-pixel by pixel, allowing each pixel to spread to an
# unoccupied neighbor with probability PROB_SPREAD. Optionally, set
# the numpy seed for reproducibility.
# If 'circle' is provided as the shape, bacteria will only
# be placed in a diameter n circle.
def growth_iter(grid, PROB_SPREAD = 0.1, seed=0, shape='square'):
    np.random.seed(seed)
    xs, ys = np.nonzero(grid)
    new_grid = grid.copy()

    for i,j in zip(xs, ys):
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                # Check that the pixel you are considering is within the grid
                valid = (i + di < grid.shape[0]) & (i + di >= 0) & (j + dj < grid.shape[1]) & (j + dj >= 0)
                if shape == 'circle':
                    r = int(grid.shape[0]/2)
                    valid = valid & (np.sqrt((i + di - r)**2 + (j + dj - r)**2) < r)
                if valid:
                    # Check that the pixel you are considering is unoccupied and if so randomly generate
                    # a value to see if the colony succeeds in spreading.
                    if (grid[i+di, j+dj] == 0) & (np.random.rand() > 1 - PROB_SPREAD):
                        new_grid[i+di, j+dj] = grid[i, j]

    xs, ys = np.nonzero(new_grid)
    if shape == 'circle':
        coverage = len(xs)/int(np.pi*(new_grid.shape[0]/2)**2)
    else:
        coverage = len(xs)/(new_grid.shape[0]*new_grid.shape[1])
    return coverage, new_grid

# Simulates C colonies on an n x n grid, growing colonies until PROP_COVERAGE
# of the surface is covered.
# n: The number of beads in each direction
# C: The number of colonies to simulate
# sigma: The sigma of the diffusion Gaussian in bead-widths
# n_sigma_border: The number of sigma-widths the colonies should extend beyond the bead grid to prevent edge effects
# PROP_COVERAGE: The simulation will stop when bacteria cover this proportion of the stamp surface
# verbose: If true will print progress information
# seed: The random seed for reproducibility
# If 'circle' is provided as the shape, bacteria will only
# be placed in a diameter n circle.
def grow_colonies(n, C, sigma, n_sigma_buffer=3, PROP_COVERAGE = 0.9, verbose=False, shape='square', seed=0):
    colony_grid = init_grid(n=n + 2*n_sigma_buffer*sigma, n_init=C, shape=shape, seed=seed)
    reached_max = False
    i = 1

    while not reached_max:
        prop, colony_grid = growth_iter(colony_grid, shape=shape)
        if verbose:
            if i % 10 == 0:
                print('Finished iter {}. Prop. coverage = {:.2f}'.format(i, prop))
        if prop >= PROP_COVERAGE:
            reached_max = True
            if verbose:
                print('Final Coverage = {:.2f}'.format(prop))
        i = i+1

    return colony_grid


# Convert an image specified by a grid to an RGB image colored
# by threshold where pixels above the threshold are red and cells
# below the threshold are blue
def color_by_thresh(img, thresh):
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
def visualize_colonies(colony_grid, beadwidth=10, step_size=500):
    n = colony_grid.shape[0]
    n_steps = int(np.ceil(beadwidth*n/step_size))
    n_big = n / (int(n/50)*10)

    fig = plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title('Bacteria Presence/Absence')
    plt.imshow(color_by_thresh(colony_grid, 0))
    _fancy_grid(0, n, n_big, 5*n_big)
    plt.xticks(np.arange(0, n+1, n/n_steps), range(0, step_size*n_steps + 1, step_size))
    plt.yticks(np.arange(0, n+1, n/n_steps), range(0, step_size*n_steps + 1, step_size))
    plt.xlabel('Position $(\mu m)$')

    plt.subplot(1,2,2)
    plt.title('Bacteria by Colony')
    grid_copy = colony_grid.copy()
    grid_copy[grid_copy == 0] = float('nan')
    plt.imshow(grid_copy, cmap='hsv')
    _fancy_grid(0, n, n_big, 5*n_big)
    plt.xticks(np.arange(0, n+1, n/n_steps), range(0, step_size*n_steps + 1, step_size))
    plt.yticks(np.arange(0, n+1, n/n_steps), range(0, step_size*n_steps + 1, step_size))
    plt.xlabel('Position $(\mu m)$')

    plt.show()

# Given a barcodes distribution, resamples it for
# num_needed barcodes. Ie. creates a new distribution
# with num_needed unique barcodes.
def _resample_overlaps(barcodes_dist, num_needed):
    # Sort the existing distribution and fit a spline to it
    sorted_dist = np.sort(barcodes_dist)
    num_orig = len(sorted_dist)
    dist_spline = scipy.interpolate.CubicSpline(range(num_orig), sorted_dist)

    # Sample the needed amount of barcodes along the existing spline
    sampled_bc = num_orig*np.random.sample(size=num_needed)

    # Evaluate the spline on our samples and re-normalize
    new_probs = dist_spline(sampled_bc)
    new_probs = new_probs/new_probs.sum()

    return new_probs


# Function to generate an expression matrix from a colony grid, given that the beads are arranged in an
# n x n square with the colonies extending sigma_pixels x buffer_pixels beyond the beads on each side
# to prevent edge effects. The amplitude specifies the number of reads that a bead directly under a bacterial
# pixel will receive. If 'circle' is provided as the shape, beads will only be generated in a diameter n circle.
def __generate_expression_matrix(colony_grid, C, n, sigma_pixels, reads_dist, rpb, overlaps_dist, n_sigma_buffer,
                                 overlap_complexity=None, seed=42, shape='square', scaling=False, verbose=False):
    # Generate beads coordinates
    buffer_pixels = sigma_pixels*n_sigma_buffer
    vec = np.arange(0,n+2*buffer_pixels)
    y,x = np.meshgrid(vec, vec)
    x = x.flatten()
    y = y.flatten()

    valid = [(x[b] >= buffer_pixels) & (x[b] < n + buffer_pixels) & (y[b] >= buffer_pixels) & (y[b] < n + buffer_pixels) for b in range(len(x))]
    if shape == 'circle':
        valid = [np.sqrt((x[i]-(n+2*buffer_pixels)/2)**2 + (y[i]-(n+2*buffer_pixels)/2)**2) < n/2 for i in range(len(x))]

    x = x[valid].astype(int)
    y = y[valid].astype(int)

    B = len(x)

    if verbose:
        print('Generated beads.\nGenerating reads...')

    # Randomly choose the number of rpb per bead
    np.random.seed(seed)

    N_rpb = np.random.choice(reads_dist['reads'], size=B, p=reads_dist['p'])
    mean_of_dist = np.average(reads_dist['reads'], weights=reads_dist['p'])
    if scaling:
        scaling_factor = rpb/mean_of_dist
        N_rpb = (N_rpb*scaling_factor).astype(int)

    # Choose the amount of overlaps
    if overlap_complexity is not None:
        overlaps_dist = _resample_overlaps(overlaps_dist, overlap_complexity)

    # Get the mapping for colonies to colony barcodes
    colony_mapping = np.random.choice(range(len(overlaps_dist)), size=C, p=overlaps_dist)

    # Store only the nnz indexes and values at each iteration,
    # and only build the sparse matrix after the loop.
    row_ind_l = []
    col_ind_l = []
    vals_l = []

    # Iterate over the beads, creating sparse vectors of counts by colony
    for b in tqdm(range(len(x))):
        colony_signal = {}
        reads_bead = N_rpb[b]

        # Iterate over the pixels within 3 sigma of the bead, creating vectors for non-zero entries
        for i in range(-3*sigma_pixels, 3*sigma_pixels + 1):
            for j in range(-3*sigma_pixels, 3*sigma_pixels + 1):
                idx_x = x[b] + i
                idx_y = y[b] + j
                if (idx_x >= 0) and (idx_y >= 0):
                    # Adjust the colony id by -1 as 0 is a colony in the read matrix but the no colony-symbol in the colony grid
                    col_id = int(colony_grid[idx_x, idx_y] - 1)

                    # Only add reads if the pixel is occupied by a colony
                    if col_id >= 0:
                        signal = np.exp(-(i**2 + j**2)/(2*sigma_pixels**2))

                        # Apply the overlap mapping
                        col_id = colony_mapping[col_id]

                        # If you've already gotten signal from that colony ID, just increment the count
                        if col_id in colony_signal:
                            colony_signal[col_id] += signal
                        # If you have not seen the colony, add it to the list and initialize the signal
                        else:
                            colony_signal[col_id] = signal

        # Convert the signal and inds to np arrays
        signal_vals = np.fromiter(colony_signal.values(), dtype=np.float32)
        signal_inds = np.fromiter(colony_signal.keys(), dtype=np.int32)

        # Check that the bead has enough signal to get reads
        if np.sum(signal_vals) == 0:
            continue

        # Normalize the signal so we can sample
        signal_vals = signal_vals/np.sum(signal_vals)

        # Sample reads and translate to indices
        count_locs = np.random.choice(len(signal_vals), size=reads_bead, p=signal_vals)
        count_locs = signal_inds[count_locs]

        # Count how many times each value should occur
        values = {loc: 0 for loc in count_locs}

        for loc in count_locs:
            values[loc] += 1

        b_count_vals = np.fromiter(values.values(), dtype=np.int16)
        b_col_ind = np.fromiter(values.keys(), dtype=np.int32)
        b_row_ind = b*np.ones_like(b_col_ind)

        row_ind_l.append(b_row_ind)
        col_ind_l.append(b_col_ind)
        vals_l.append(b_count_vals)

    # Concatenate all the counts
    counts_mat = scipy.sparse.csr_matrix((np.concatenate(vals_l),
                             (np.concatenate(row_ind_l),
                              np.concatenate(col_ind_l))),
                              shape=(B,overlaps_dist.shape[0]), dtype=np.int16)

    # Compress the matrix
    counts_mat.eliminate_zeros()
    row_sum = counts_mat.sum(axis=0)
    nz = np.array(row_sum > 0).flatten()
    counts_mat = counts_mat[:, nz]

    return counts_mat


# Generate reads by iterating over the beads and adding reads from all
# pixels within 3 sigma of the bead. Returns an anndata with beads on
# the rows and colonies on the columns with the original colony layout
# stored in uns.
# NOTE: This function assumes each pixel is a bead, i.e. the scale is
# 10um. The colony grid is taken to be aligned with the bead grid, so
# each cell of the colony grid is also 10um and the beads are aligned
# so element [i,j] overlaps in both grids. Both grids are extended
# 3*sigma_pix to both sides to simulate colonies extending beyond beads.
# colony_grid: the grid containing colony ids for each pixel and 0 for
#              no colony.
# C: the number of unique ccolony IDs
# n: the number of width/height of the plate in beads, assumes beads
#    are in a grid
# sigma_pixels: the value of sigma in pixels, i.e. multiples of 10um
# reads_dist: A dataframe with two columns. The first is the number of reads, and the
#             second is the probability of them.
# rpb: The target reads per bead
# overlaps_dist: An array of probabilities for that colony to occur
# overlap_complexity: The total number of barcodes for colonies
# verbose: If true, print progress messages
# shape: If 'circle' is provided as the shape, beads will only be generated in a diameter n circle.
def generate_reads(colony_grid, C, n, sigma_pixels, reads_dist, rpb, overlaps_dist, seed=42, overlap_complexity=None, n_sigma_buffer=3,
                   shape='square', scaling=False, verbose=True):
    # Generate the expression matrix
    expr_matrix = __generate_expression_matrix(colony_grid, C, n, sigma_pixels,
                                               reads_dist, rpb, overlaps_dist, n_sigma_buffer, seed=seed,
                                               overlap_complexity=overlap_complexity, shape=shape, scaling=scaling, verbose=verbose)

    if verbose:
        print('Generated expression matrix.')

    # Create an anndata object from the expression matrix
    y, x = np.meshgrid(np.arange(0,n), np.arange(0,n))
    x = x.flatten()
    y = y.flatten()

    if shape == 'circle':
        in_circle = [np.sqrt((x[i]-n/2)**2 + (y[i]-n/2)**2) < n/2 for i in range(len(x))]
        x = x[in_circle]
        y = y[in_circle]
    obs = pd.DataFrame(columns=['x', 'y'], data=np.array([x,y]).T)
    adata = anndata.AnnData(X=expr_matrix, obs=obs, var=pd.DataFrame(index=range(expr_matrix.shape[1])))
    if verbose:
        print('Generated anndata.\nAdding metadata...')

    # Add data to obs, var, and uns about the size of each colony, the number of non-zero colonies per bead,
    # and the original colony grid
    adata.obs['total_reads'] = np.sum(adata.X, axis=1)
    adata.obs['total_colonies'] = [adata.X[i,:].count_nonzero() for i in range(adata.shape[0])]
    adata.obs['spatial_color'] = euclidean_color(adata.obs.x, adata.obs.y)

    adata.var['area'] = [len(colony_grid[colony_grid == i + 1]) for i in range(adata.shape[1])]

    adata.uns['colony_grid'] = colony_grid
    if verbose:
        print('Finished read generation')

    return adata

# Generate reads based on stamping each plate in the list colony_grids once.
# Returns an anndata with beads on the rows and colonies on the columns with
# the original colony grids stored in uns.
def stamp_colonies(colony_grids, C, n, amplitude, sigma_pixels, n_sigma_buffer=3, shape='square', verbose=False):
    # Generate the expression matrices
    expression_matrices = []
    n_stamps = len(colony_grids)
    for i in range(n_stamps):
        if verbose:
            print('\nGenerating reads for stamp {}...'.format(i))
        expression_matrices.append(__generate_expression_matrix(colony_grids[i], C, n, amplitude, sigma_pixels, n_sigma_buffer, shape=shape, verbose=verbose))

    # Build var
    var = pd.DataFrame(columns=['stamp', 'area'])
    for i in range(n_stamps):
        temp = pd.DataFrame(index=np.arange(expression_matrices[i].shape[1]))
        temp['stamp'] = i + 1
        temp['area'] = [len(colony_grids[i][colony_grids[i] == j + 1]) for j in range(expression_matrices[i].shape[1])]

        var = var.append(temp, ignore_index=True)

    # Build the anndata
    y, x = np.meshgrid(np.arange(0,n), np.arange(0,n))
    y, x = np.meshgrid(np.arange(0,n), np.arange(0,n))
    x = x.flatten()
    y = y.flatten()

    if shape == 'circle':
        in_circle = [np.sqrt((x[i]-n/2)**2 + (y[i]-n/2)**2) < n/2 for i in range(len(x))]
        x = x[in_circle]
        y = y[in_circle]
    obs = pd.DataFrame(columns=['x', 'y'], data=np.array([x,y]).T)
    adata = anndata.AnnData(X=expr_matrix, obs=obs, var=pd.DataFrame(index=range(1, C + 1)))
    if verbose:
        print('Generated anndata\nAdding metadata...')

    # Add the obs fields
    adata.obs['total_reads'] = np.sum(adata.X, axis=1)
    adata.obs['total_colonies'] = [adata.X[i,:].count_nonzero() for i in range(adata.shape[0])]

    # Add the colony grids to uns
    adata.uns['colony_grids'] = colony_grids
    if verbose:
        print('Finished read generation')

    return adata


# Adds layers to adata where beads are randomly downsampled to an average of each value of rpbs provided in the list.
# Returns a copy of adata with the new downsampled layers.
# adata: AnnData to be downsampled
# rpbs: List of mean reads per bead where a downsampled layer is created for each value.
# copy: If true, a copy of adata will be returned. Otherwise, layers will be added inplace.
def downsample_reads(adata, rpbs, copy=False, inplace=False):
    # If we're modifying inplace we should only have one rpb
    if inplace and len(rpbs) > 1:
        rpbs = rpbs[:1]
        print('Warning: inplace modification only supports a single rpb value. Using first value.')

    if copy == True:
        new_adata = adata.copy()
    else:
        new_adata = adata

    adata_total = np.sum(adata.X, dtype=int)
    for rpb in rpbs:
        total = rpb*adata.shape[0]
        if total > adata_total:
            print('Warning: {} rpb is greater than counts in the data. ({:.2e} > {:.2e})\n Shape = {}'.format(rpb, total, adata_total, adata.shape))

        if inplace:
            scanpy.pp.downsample_counts(adata, total_counts=total, replace=True, copy=False)
            adata.obs['{}_rpb_total_reads'.format(rpb)] = np.sum(adata.X, axis=1)
            adata.obs['{}_rpb_total_colonies'.format(rpb)] = [adata.X[i,:].count_nonzero() for i in range(adata.shape[0])]
        else:
            downsampled = scanpy.pp.downsample_counts(adata, total_counts=total, replace=False, copy=True)
            new_adata.layers['{}_rpb'.format(rpb)] = downsampled.X
            new_adata.obs['{}_rpb_total_reads'.format(rpb)] = np.sum(downsampled.X, axis=1)
            new_adata.obs['{}_rpb_total_colonies'.format(rpb)] = [downsampled.X[i,:].count_nonzero() for i in range(downsampled.shape[0])]

    if copy == True:
        return new_adata

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

# Contruct a matrix from an old matrix by summing the overlapping columns specified in overlaps.
def _merge_overlaps(old_counts, overlaps):
    new_columns = []

    for columns in tqdm(overlaps):
        overlapping = old_counts[:, columns]

        if len(columns) > 1:
            new_columns.append(scipy.sparse.csc_matrix(overlapping.sum(axis=1)))
        else:
            new_columns.append(overlapping)

    new_counts = scipy.sparse.hstack(new_columns).tocsr()

    return new_counts

# Given a counts matrix returns the same data subject to overlapping barcodes
# according to barcode_dist
def add_overlap_to_counts(counts_mtx, barcode_dist):
    if type(counts_mtx) == scipy.sparse.csc.csc_matrix:
        counts = counts_mtx
    else:
        counts = counts_mtx.tocsc()

    # Generate a barcode for each colony following the provided distribution
    colony_ids = np.random.choice(len(barcode_dist), size=counts_mtx.shape[1], p=barcode_dist)

    # Get the indices where each barcode occurs
    unique_barcodes = np.unique(colony_ids)
    indices = []
    for barcode in unique_barcodes:
        indices.append(np.where(colony_ids == barcode)[0])

    # Apply the overlaps to the counts matrix
    return _merge_overlaps(counts, indices)
