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
def make_gauss_kernel(a, var):
    def gauss(x2):
        return a*np.exp(-x2/(2*var))

    return gauss

# Given a distribution on reads, returns a vector of reads of length N_beads.
def distribute_reads(reads_dist, mean, N_beads):
    # Get a vector of qualities
    first_sample = np.random.choice(reads_dist['reads'], size=N_beads, p=reads_dist['p'])
    qualities = first_sample/np.sum(first_sample)

    # Sample mean*N_beads reads and assign each bead the corresponding number of reads
    read_samples = np.random.choice(len(qualities), size=int(mean)*N_beads, p=qualities)

    # Add up how many reads each bead got
    N_reads = np.zeros(shape=N_beads)

    for sampled_read in tqdm(read_samples):
        N_reads[sampled_read] += 1

    return N_reads

# Given a barcodes distribution, resamples it for num_needed barcodes,
# ie. creates a new distribution with num_needed unique barcodes.
def resample_overlaps(barcodes_dist, num_needed):
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
# -- overlap_scaling: Whether we want to resample our overlap distribution (ussually to have more complexity)
# -- rpb: The mean number of rpbs if scaling or if no distribution is provided
# -- rng: A numpy random number generator for reproducibility (optional)
# -- num_proc: The number of processors to use
# -- chunksize: The number of beads each cpu computes at a time. Too low a chunksize will severely impact performance.
def get_colony_counts(sigma, beads_coords, colonies_coords,
                      sub_indices=None, scaling=False, overlap_scaling=False,
                      rpb=1000, overlap_dist=None, overlap_complexity=None, reads_dist=None,
                      rng=None, num_proc=0, chunksize=1000):
    sq_dist_threshold = 9*sigma**2

    # Store only the nnz indexes and values at each iteration,
    # and only build the sparse matrix after the loop.
    row_ind_l = []
    col_ind_l = []
    vals_l = []

    # Check if we need to make a new random number generator
    if rng is None:
        SEED = 42 # Random seed for reproducibility
        rng = np.random.default_rng(SEED) # Make rng instance with the seed

    gaussian_count_ker = make_gauss_kernel(1, sigma**2) # Make the Gaussian function for counts

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
        overlap_dist = resample_overlaps(overlap_dist, overlap_complexity)

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


# Creates a kernel that accepts n x d matrices (n samples in dimension d).
# PARAMS:
#  -- amp is the amplitude
#  -- theta: angle (in radians) of the anisotropy
#  -- sigma: diffusion level of the first direction (pointing at the angle theta)
#  -- anisotropy: strength of the anisotropy: how much less the second direction is diffused. A value of 1 means isotropic.
def make_gauss_kernel_uniform_anisotropic(amp=1, theta=0, sigma=0.1, anisotropy=2.0):
    # https://janakiev.com/blog/covariance-matrix/
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    S2_inv = np.diag([1/anisotropy**2, 1])/(2*sigma**2)
    C_inv = rot_mat @ S2_inv @ rot_mat.T    # C = RSSR^{-1}  -> C^1 = RS^{-2}R^T

    def gauss(x):
        return amp*np.exp(-(x*(C_inv@x.T).T).sum(-1))
    return gauss


def get_colony_counts_anisotropic_diffusion(sigma, beads_coords, colonies_coords,
                                            sub_indices=None, scaling=False, overlap_scaling=False,
                                            rpb=1000, overlap_dist=None, overlap_complexity=None, reads_dist=None,
                                            theta=0, anisotropy=2, rng=None, num_proc=0, chunksize=1000):
    # Store only the nnz indexes and values at each iteration,
    # and only build the sparse matrix after the loop.
    row_ind_l = []
    col_ind_l = []
    vals_l = []

    # Check if we need to make a new random number generator
    if rng is None:
        SEED = 42  # Random seed for reproducibility
        rng = np.random.default_rng(SEED)  # Make rng instance with the seed

    gaussian_ker_anis = make_gauss_kernel_uniform_anisotropic(sigma=sigma, theta=theta, anisotropy=anisotropy)
    # gaussian_count_ker = make_gauss_kernel(1, sigma**2)  # Make the Gaussian function for counts

    # sq_dist_threshold = (3*sigma)**2    # Cut-off at 3 sigma
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
        overlap_dist = resample_overlaps(overlap_dist, overlap_complexity)

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

# Generates an anndata whose read matrix contains satellite barcode counts.
# PARAMS:
# -- sigma: The standard deviation for the Gaussian
# -- beads_coords: The coordinates of the beads
# -- colonies_coords: The coordinates of the colonies
# -- sub_indices: The indices of the beads that will be included in the counts matrix (optional)
# -- overlap_dist: An array of probabilities for each rank of barcode to be drawn.
# -- reads_dist: A pd dataframe with two columns 'reads' and 'p', where p gives the probability of each count
#                occuring
# -- scaling: Boolean for whether or not to scale the distribution to a mean number of rpbs
# -- overlap_scaling: Whether we want to resample our overlap distribution (ussually to have more complexity)
# -- rpb: The mean number of rpbs if scaling or if no distribution is provided
# -- rng: A numpy random number generator (optional)
# -- num_proc: The number of processors to use
# -- chunksize: The number of beads each cpu computes at a time. Too low a chunksize will severely impact performance.
def generate_reads(sigma, bead_coords, satellite_coords, sub_indices=None, overlap_dist=None, overlap_complexity=None, reads_dist=None,
                   num_proc=32, rng=None, chunksize=30000, scaling=False):
    satellite_counts = get_colony_counts(sigma, bead_coords, satellite_coords, num_proc=num_proc,
                                         overlap_dist=overlap_dist, reads_dist=reads_dist, chunksize=chunksize,
                                         rng=rng, overlap_complexity=overlap_complexity, scaling=scaling)

    beads_df = pd.DataFrame(data=bead_coords, columns=['x', 'y'])

    adata = anndata.AnnData(X=satellite_counts, obs=beads_df, var=pd.DataFrame(index=range(satellite_counts.shape[1])), dtype=np.int32)

    adata.obs['spatial_color'] = visuals.radial_color(adata.obs.x, adata.obs.y, x0=0.5, y0=0.5)

    return adata
