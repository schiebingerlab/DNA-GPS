import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import gc
import anndata
import scipy
from tqdm import tqdm
import numexpr as ne
import visuals

# Given amplitude and variance return the corresponding Gaussian.
def _make_gauss_kernel(a, var):
    def gauss(x2):
        return a*np.exp(-x2/(2*var))

    return gauss


# Creates a kernel that accepts n x d matrices (n samples in dimension d).
# PARAMS:
#  -- amp is the amplitude
#  -- theta: angle (in radians) of the anisotropy
#  -- sigma: diffusion level of the first direction (pointing at the angle theta)
#  -- anisotropy: strength of the anisotropy: how much less the second direction is diffused. A value of 1 means isotropic.
def _make_gauss_kernel_uniform_anisotropic(amp=1, theta=0, sigma=0.1, anisotropy=2.0):
    # https://janakiev.com/blog/covariance-matrix/
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    S2_inv = np.diag([1/anisotropy**2, 1])/(2*sigma**2)
    C_inv = rot_mat @ S2_inv @ rot_mat.T    # C = RSSR^{-1}  -> C^1 = RS^{-2}R^T

    def gauss(x):
        return amp*np.exp(-(x*(C_inv@x.T).T).sum(-1))
    return gauss


# Given a barcodes distribution, resamples it for num_needed barcodes,
# ie. creates a new distribution with num_needed unique barcodes.
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

# Create an n x n grid and initialize n_init colonies. Optionally provide a seed
# to numpy for reproducibility. Colonies are assigned an ID from 1 to n_init.
# Cells with no colony have a value of 0. If 'circle' is provided as the shape,
# bacteria will only be placed in a diameter n circle.
def _init_grid(n, n_init, seed=0, shape='square'):
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

# Perform a single iteration of growth, returning the updated grid. Iterates colony-pixel
# by pixel, allowing each pixel to spread to an unoccupied neighbor with probability
# PROB_SPREAD. Optionally, set the numpy seed for reproducibility. If 'circle' is provided
# as the shape, bacteria will only be placed in a diameter n circle.
def _growth_iter(grid, PROB_SPREAD = 0.1, seed=0, shape='square'):
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
# PARAMS:
# -- n: The number of beads in each direction, not accounting for the three sigma buffer to prevent edge effects
# -- C: Number of colonies to simulate
# -- sigma_um: The width of the diffusion kernel in um
# -- beadwidth_um: the width of a bead in um
# -- n_sigma_border: The number of sigma-widths the colonies should extend beyond the bead grid to prevent edge effects
# -- PROP_COVERAGE: The simulation will stop when bacteria cover this proportion of the stamp surface
# -- verbose: If true will print progress information
# -- seed: The random seed for reproducibility
# -- shape: If circle, bacteria will only be placed in a diameter n circle.
def grow_colonies(n, C, sigma_um, beadwidth_um, n_sigma_buffer=3, PROP_COVERAGE = 0.9, verbose=False, shape='square', seed=0):
    sigma_pixels = int(np.ceil(sigma_um/beadwidth_um))
    colony_grid = _init_grid(n=n + 2*n_sigma_buffer*sigma_pixels, n_init=C, shape=shape, seed=seed)
    reached_max = False
    i = 1

    while not reached_max:
        prop, colony_grid = _growth_iter(colony_grid, shape=shape)
        if verbose:
            if i % 10 == 0:
                print('Finished iter {}. Prop. coverage = {:.2f}'.format(i, prop))
        if prop >= PROP_COVERAGE:
            reached_max = True
            if verbose:
                print('Final Coverage = {:.2f}'.format(prop))
        i = i+1

    return colony_grid


# Computes colony counts matrix for a subset of indices.
# PARAMS:
# -- sigma: The standard deviation for the Gaussian kernel
# -- beads_coords: The coordinates of the beads
# -- colonies_coords: The coordinates of the colonies
# -- sub_indices: The indices of the beads that will be included in the counts matrix (optional)
# -- overlap_dist: An array of probabilities for each rank of barcode to be drawn.
# -- reads_dist: A pd dataframe with two columns 'reads' and 'p', where p gives the probability of each count
#                occuring
# -- scaling: Boolean for whether or not to scale the distribution to a mean number of rpbs
# -- overlap_scaling: Whether we want to resample our overlap distribution (usually to have more complexity)
# -- rpb: The mean number of rpbs if scaling or if no distribution is provided
# -- seed: Seed to be used for the numpy random number generator
# -- num_proc: The number of processors to use
# -- chunksize: The number of beads each cpu computes at a time. Too low a chunksize will severely impact performance.
def _get_colony_counts(sigma, beads_coords, colonies_coords, sub_indices=None, scaling=False,
                       overlap_scaling=False, rpb=1000, overlap_dist=None, overlap_complexity=None,
                       reads_dist=None, seed=0, num_proc=0, chunksize=1000):
    sq_dist_threshold = 9*sigma**2

    # Store only the nnz indexes and values at each iteration,
    # and only build the sparse matrix after the loop.
    row_ind_l = []
    col_ind_l = []
    vals_l = []

    rng = np.random.default_rng(seed) # Make rng instance with the seed

    gaussian_count_ker = _make_gauss_kernel(1, sigma**2) # Make the Gaussian function for counts

    # Set which bead indices we'll be considering
    if sub_indices is None:
        sub_indices = range(0, beads_coords.shape[0])

    # Get the number of beads
    N = len(sub_indices)

    # Get the number of rpb for each bead
    if reads_dist is None:
        N_rpb = [rpb for i in range(N)]
    else:
        N_rpb = np.random.choice(reads_dist['reads'], size=N, p=reads_dist['p'])

        # Apply scaling
        if scaling:
            mean_of_dist = np.average(reads_dist['reads'], weights=reads_dist['p'])
            scaling_factor = rpb/mean_of_dist

            N_rpb = (N_rpb*scaling_factor).astype(int)

    if overlap_scaling:
        overlap_dist = _resample_overlaps(overlap_dist, overlap_complexity)

    # Get the mapping for colonies to colony barcodes
    colony_mapping = rng.choice(range(len(overlap_dist)), size=colonies_coords.shape[0], p=overlap_dist)

    def worker_colony_counts(i):
        ind = sub_indices[i]
        bead_coord = beads_coords[ind]
        a = colonies_coords # This is here just so that this function keeps colonies_coords in its scope for processes.

        sqd = ne.evaluate('sum((bead_coord - colonies_coords)**2, axis=1)')  # Compute distance to all colonies
        col_ind = np.where(sqd < sq_dist_threshold)[0]  # Get indices of distances below the threshold
        dists_below_thresh = sqd[col_ind] # Get the distances below the threshold
        counts_dist = gaussian_count_ker(dists_below_thresh)  # Compute raw counts values

        # Normalize so that the counts distribution is a real distribution
        sum_vals = ne.evaluate('sum(counts_dist)')
        counts_dist = counts_dist/sum_vals

        # Perform sampling
        reads_bead = N_rpb[i] # The number of reads for this bead

        # Check if our data is too sparse to produce counts
        if len(counts_dist) == 0:
            return [[], [], []]

        # Choose colonies to get reads from below threshold
        count_locs = rng.choice(len(counts_dist), size=reads_bead, p=counts_dist)
        count_locs = col_ind[count_locs] # Translate to actual colony indices rather than relative indices

        # Count how many times each value should occur
        values = {loc: 0 for loc in colony_mapping[count_locs]}

        for loc in count_locs:
            values[colony_mapping[loc]] += 1

        count_vals = np.fromiter(values.values(), dtype=np.int16)
        col_ind = np.fromiter(values.keys(), dtype=np.int32)
        row_ind = i*np.ones_like(col_ind)

        return [col_ind, row_ind, count_vals]

    t0 = time.time()

    # Progress bar works, and same timing as without tqdm and with pathos.
    import tqdm
    import pathos.multiprocessing as mp
    with mp.Pool(num_proc) as p:
        results = list(tqdm.tqdm(p.imap(worker_colony_counts, range(len(sub_indices)), chunksize=chunksize), total=len(sub_indices)))
    # Unpack results
    for r in results:
        col_ind_l.append(r[0])
        row_ind_l.append(r[1])
        vals_l.append(r[2])

    print("Matrix build time:",time.time()-t0)

    # Concatenate all the counts
    colony_counts = scipy.sparse.csr_matrix((np.concatenate(vals_l), (np.concatenate(row_ind_l), np.concatenate(col_ind_l))), shape=(N,len(overlap_dist)), dtype=np.int16)
    colony_counts.eliminate_zeros()

    # Remove empty colonies
    row_sum = colony_counts.sum(axis=0)
    nz = np.array(row_sum > 0).flatten()
    colony_counts = colony_counts[:, nz]

    return colony_counts


def _get_colony_counts_anisotropic(sigma, beads_coords, colonies_coords,
                                   sub_indices=None, scaling=False, overlap_scaling=False,
                                   rpb=1000, overlap_dist=None, overlap_complexity=None, reads_dist=None,
                                   theta=0, anisotropy=2, seed=0, num_proc=0, chunksize=1000):
    # Store only the nnz indexes and values at each iteration,
    # and only build the sparse matrix after the loop.
    row_ind_l = []
    col_ind_l = []
    vals_l = []

    rng = np.random.default_rng(seed)  # Make rng instance with the seed

    gaussian_ker_anis = _make_gauss_kernel_uniform_anisotropic(sigma=sigma, theta=theta, anisotropy=anisotropy)
    count_threshold = np.exp(-3**2/2)

    # Set which bead indices we'll be considering
    if sub_indices is None:
        sub_indices = range(0, beads_coords.shape[0])

    # Get the number of beads
    N = len(sub_indices)

    # Get the number of rpb for each bead
    if reads_dist is None:
        N_rpb = [rpb for i in range(N)]
    else:
        N_rpb = np.random.choice(reads_dist['reads'], size=N, p=reads_dist['p'])

        # Apply scaling
        if scaling:
            mean_of_dist = np.average(reads_dist['reads'], weights=reads_dist['p'])
            scaling_factor = rpb / mean_of_dist

            N_rpb = (N_rpb * scaling_factor).astype(int)

    if overlap_scaling:
        overlap_dist = _resample_overlaps(overlap_dist, overlap_complexity)

    # Get the mapping for colonies to colony barcodes
    colony_mapping = rng.choice(range(len(overlap_dist)), size=colonies_coords.shape[0], p=overlap_dist)

    def worker_colony_counts(i):
        ind = sub_indices[i]
        bead_coord = beads_coords[ind]

        # sqd = ne.evaluate('sum((bead_coord - colonies_coords)**2, axis=1)')  # Compute distance to all colonies
        # col_ind = np.where(sqd < sq_dist_threshold)[0]  # Get indices of distances below the threshold
        # dists_below_thresh = sqd[col_ind]  # Get the distances below the threshold
        # counts_dist2 = gaussian_count_ker(dists_below_thresh)  # Compute raw counts values

        # It doesn't make sense to threshold it on (Mahalanobis) distance, since it's only an exp() away from the counts
        counts_dist = gaussian_ker_anis(bead_coord - colonies_coords)
        col_ind = np.where(counts_dist > count_threshold)[0]
        counts_dist = counts_dist[col_ind]

        #         if stdev_noise > 0:
        #             counts_dist = noise_kernel(counts_dist)  # Add noise to counts
        #             counts_dist = counts_dist[counts_dist > 0]  # Make sure no counts are below zero

        # Normalize so that the counts distribution is a real distribution
        sum_vals = ne.evaluate('sum(counts_dist)')
        counts_dist = counts_dist / sum_vals

        # Perform sampling
        reads_bead = N_rpb[i]  # The number of reads for this bead

        # Check if our data is too sparse to produce counts
        if len(counts_dist) == 0:
            return [[], [], []]

        # Choose colonies to get reads from below threshold
        count_locs = rng.choice(len(counts_dist), size=reads_bead, p=counts_dist)
        count_locs = col_ind[count_locs]  # Translate to actual colony indices rather than relative indices

        # Count how many times each value should occur
        values = {loc: 0 for loc in colony_mapping[count_locs]}

        for loc in count_locs:
            values[colony_mapping[loc]] += 1

        count_vals = np.fromiter(values.values(), dtype=np.int16)
        col_ind = np.fromiter(values.keys(), dtype=np.int32)
        row_ind = i * np.ones_like(col_ind)

        return [col_ind, row_ind, count_vals]

    t0 = time.time()

    results = []
    for i in range(len(sub_indices)):
        results.append(worker_colony_counts(i))

    # Unpack results
    for r in results:
        col_ind_l.append(r[0])
        row_ind_l.append(r[1])
        vals_l.append(r[2])

    print("Matrix build time:", time.time() - t0)

    # Concat all the counts
    colony_counts = scipy.sparse.csr_matrix(
        (np.concatenate(vals_l), (np.concatenate(row_ind_l), np.concatenate(col_ind_l))), shape=(N, len(overlap_dist)),
        dtype=np.int16)
    colony_counts.eliminate_zeros()

    # Remove empty colonies
    row_sum = colony_counts.sum(axis=0)
    nz = np.array(row_sum > 0).flatten()
    colony_counts = colony_counts[:, nz]

    return colony_counts

# Computes colony counts matrix assuming barcodes are dispersed using bacterial colonies.
# PARAMS:
# -- colony_grid: colony grid specifying the the bacterial colony growing in each grid cell
# -- sigma_pixels: The width of the diffusion kernel in micro meters
# -- beadwidth_um: Beadwidth in micro meters
# -- reads_dist: A pd dataframe with two columns 'reads' and 'p', where p gives the probability of each count
#                occuring
# -- rpb: The mean number of rpbs if scaling or if no distribution is provided
# -- overlap_dist: An array of probabilities for each rank of barcode to be drawn.
# -- n_sigma_border: The number of sigma-widths the colonies should extend beyond the bead grid to prevent edge effects
# -- overlap_scaling: Whether we want to resample our overlap distribution (usually to have more complexity)
# -- overlap_complexity: Whether we want to resample our overlap distribution (usually to have more complexity)
# -- seed: Seed for the numpy random number generator
# -- shape: If set to circle, beads will only be generated in a diameter n circle,
#           otherwise the full square will be covered in beads.
# -- scaling:
def _get_colony_counts_bacterial(colony_grid, sigma_um, beadwidth_um, reads_dist, overlaps_dist, n_sigma_buffer=3,
                                 rpb=0, overlap_complexity=None, seed=0, shape='square', scaling=False, verbose=False):
    # Calculate parameters
    C = len(np.unique(colony_grid)) - 1 # Caclulate the number of unique colonies
    sigma_pixels = int(np.ceil(sigma_um/beadwidth_um))
    n = colony_grid.shape[0] - 2*n_sigma_buffer*sigma_pixels

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

    rng = np.random.seed(seed)

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
        for i in range(-3*sigma_pixels, 3*sigma_pixels):
            for j in range(-3*sigma_pixels, 3*sigma_pixels):
                idx_x = x[b] + i
                idx_y = y[b] + j
                #if (idx_x >= 0) and (idx_x < n) and (idx_y >= 0) and (idx_y < n):
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


# Generates an anndata whose read matrix contains satellite barcode counts,
# using either uniform or anisotropic diffusion.
# PARAMS:
# -- sigma: The standard deviation for the Gaussian in unit normalized coordinates
# -- beads_coords: Coordinates of the beads in unit normalized coordinates
# -- colonies_coords: Coordinates of the beads in unit normalized coordinates
# -- diffusion: Whether to use isotropic or anisotropic diffusion
# -- sub_indices: The indices of the beads that will be included in the counts matrix (optional)
# -- overlap_dist: An array of probabilities for each rank of barcode to be drawn.
# -- overlap_complexity: Whether we want to resample our overlap distribution (usually to have more complexity)
# -- reads_dist: A pandas dataframe with two columns 'reads' and 'p', where p gives the probability of
#                each count occuring
# -- anisoptropy: Level of anisotropy if using anisotropic diffusion
# -- theta: Angle for the direction of anisotropy if using anisotropic diffusion
# -- seed: Seed for the numpy random number generator
# -- num_proc: The number of processors to use
# -- chunksize: The number of beads each cpu computes at a time. Too low a chunksize will severely impact performance.
# -- scaling: Boolean for whether or not to scale the distribution to a mean number of rpbs
# -- rpb: The mean number of rpbs if scaling or if no distribution is provided
def generate_reads(sigma, bead_coords, satellite_coords, diffusion='base', sub_indices=None, overlap_dist=None,
                   overlap_complexity=None, reads_dist=None, anisotropy=1, theta=0, num_proc=32, seed=0,
                   chunksize=30000, scaling=False, rpb=0):
    if anisotropy != 1:
        satellite_counts = _get_colony_counts_anisotropic(sigma, bead_coords, satellite_coords, num_proc=num_proc,
                                                          overlap_dist=overlap_dist, reads_dist=reads_dist, anisotrophy=anisotropy,
                                                          theta=theta, chunksize=chunksize, seed=seed, overlap_complexity=overlap_complexity,
                                                          scaling=scaling)
    else:
        satellite_counts = _get_colony_counts(sigma, bead_coords, satellite_coords, num_proc=num_proc,
                                              overlap_dist=overlap_dist, reads_dist=reads_dist, chunksize=chunksize,
                                              seed=seed, overlap_complexity=overlap_complexity, scaling=scaling)


    beads_df = pd.DataFrame(data=bead_coords, columns=['x', 'y'])
    adata = anndata.AnnData(X=satellite_counts, obs=beads_df, var=pd.DataFrame(index=range(satellite_counts.shape[1])), dtype=np.int32)

    x0 = (max(adata.obs.x) - min(adata.obs.x))/2 + min(adata.obs.x)
    y0 = (max(adata.obs.y) - min(adata.obs.y))/2 + min(adata.obs.y)
    adata.obs['spatial_color'] = visuals.radial_color(adata.obs.x, adata.obs.y, x0=x0, y0=y0)

    return adata

# Generates an anndata whose read matrix contains satellite barcode counts,
# using dispersion by bacterial colonies.
# NOTE: This method generates a dense grid of beads and assumes each pixel of the
# colony grid represents a single beadwidth.
# PARAMS:
# -- colony_grid: the grid containing colony ids for each pixel and 0 for
#                 no colony.
# -- sigma_pixels: Width of the diffusion kernel in micro meters
# -- beadwidth_um: Beadwidth in micro meters
# -- reads_dist: A pandas dataframe with two columns 'reads' and 'p', where p gives the probability of
#                each count occuring
# -- rpb: The mean number of rpbs if scaling or if no read distribution is provided
# -- overlaps_dist: An array of probabilities for that colony to occur
# -- overlap_complexity: Whether we want to resample our overlap distribution (usually to have more complexity)
# -- verbose: If true, print progress messages
# -- shape: If set to circle, beads will only be generated in a diameter n circle,
#           otherwise the full square will be covered in beads.
def generate_bacterial_reads(colony_grid, sigma_um, beadwidth_um, reads_dist, overlaps_dist, rpb=0,
                             seed=0, overlap_complexity=None, n_sigma_buffer=3, shape='square',
                             scaling=False, verbose=True):

    # Generate the expression matrix
    expr_matrix = _get_colony_counts_bacterial(colony_grid, sigma_um, beadwidth_um, reads_dist,
                                               overlaps_dist, n_sigma_buffer, rpb=rpb, seed=seed,
                                               overlap_complexity=overlap_complexity, shape=shape,
                                               scaling=scaling, verbose=verbose)

    if verbose:
        print('Generated expression matrix.')

    # Create an anndata object from the expression matrix
    n = colony_grid.shape[0] - 2*n_sigma_buffer*int(np.ceil(sigma_um/beadwidth_um))
    y, x = np.meshgrid(np.arange(0,n), np.arange(0,n))
    x = x.flatten()
    y = y.flatten()

    if shape == 'circle':
        in_circle = [np.sqrt((x[i]-n/2)**2 + (y[i]-n/2)**2) < n/2 for i in range(len(x))]
        x = x[in_circle]
        y = y[in_circle]
    obs = pd.DataFrame(columns=['x', 'y'], data=beadwidth_um*np.array([x,y]).T)
    adata = anndata.AnnData(X=expr_matrix, obs=obs, var=pd.DataFrame(index=range(expr_matrix.shape[1])))
    if verbose:
        print('Generated anndata.\nAdding metadata...')

    # Add data to obs, var, and uns about the size of each colony, the number of non-zero colonies per bead,
    # and the original colony grid
    adata.obs['total_reads'] = np.sum(adata.X, axis=1)
    adata.obs['total_colonies'] = [adata.X[i,:].count_nonzero() for i in range(adata.shape[0])]
    x0 = (max(adata.obs.x) - min(adata.obs.x))/2 + min(adata.obs.x)
    y0 = (max(adata.obs.y) - min(adata.obs.y))/2 + min(adata.obs.y)
    adata.obs['spatial_color'] = visuals.radial_color(adata.obs.x, adata.obs.y, x0=x0, y0=y0)

    adata.var['area'] = [len(colony_grid[colony_grid == i + 1]) for i in range(adata.shape[1])]

    adata.uns['colony_grid'] = colony_grid
    if verbose:
        print('Finished read generation')

    return adata
