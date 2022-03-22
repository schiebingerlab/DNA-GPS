import numpy as np
import anndata
import math
from datetime import datetime
from scipy.stats import norm
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import euclidean_distances

# Calculate the Euclidean distance between two dense vectors
def euclidean_dist (x,y):
    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

# Calculates the area of the triangle specified by the three vertices x,y,z
def triangle_area(x,y,z):
    return abs((x[0]*(y[1]-z[1]) + y[0]*(z[1]-x[1]) + z[0]*(x[1]-y[1]))/2)

# Helper functions for calculating Euclidean distances on CSR matrices.
# Returns true if the elements in entry i and i+1 of data should be combined
# in the projected barcode space.
def should_combine(data, idx):
    if idx >= data.shape[1] - 1:
        return False
    elif data[0,idx] != data[0,idx+1]:
        return False
    else:
        return True

# Helper functions for calculating Euclidean distances on CSR matrices.
# Returns true if the current index in list i comes before the current index in
# list j.
def i_next(data_i, data_j, idx_i, idx_j):
    if idx_i >= data_i.shape[1]:
        return False
    elif idx_j >= data_j.shape[1]:
        return True
    elif data_i[0,idx_i] < data_j[0,idx_j]:
        return True
    else:
        return False

# Helper functions for calculating Euclidean distances on CSR matrices.
# Returns true if the current elements of list i and j are in the same bin
# in projected space.
def add_next(data_i, data_j,idx_i, idx_j):
    if idx_i >= data_i.shape[1]:
        return False
    elif idx_j >= data_j.shape[1]:
        return False
    elif data_i[0,idx_i] == data_j[0,idx_j]:
        return True
    else:
        return False

# Converts the indices of an array containing uniformly distributed indices to
# normally distributed indices, given the maximum potential index (M).
def uniform_to_normal(data, M):
    quantiles = data/M
    quantiles = [0.01 if q < 0.01 else 0.99 if q > 0.99 else q for q in quantiles]
    norm_floats = M*(norm.ppf(quantiles) + norm.ppf(0.99))/(2*norm.ppf(0.99))
    data = np.round(norm_floats)
    return data

# Calculate the Euclidean distance between two rows of an anndata object
# where X is a CSR matrix. If the mod passed is greater than 0 the projected
# data will be binned into mod bins.
# barcodes: Anndata object with barcodes in X
# I: index of first bead
# J: index of second bead
# mod: If non-negative, the size of the projected space. If negative barcodes
#      are considered to be unique.
# method: Method for binning barcodes into projected space. If uniform
#         all bins will be roughly the same size. If normal bin size will
#         roughly follow a normal distribution.
def sparse_euclidean_dist(barcodes, I, J, mod=-1, method='uniform'):
    data_i = np.array([barcodes.X[I,:].indices, barcodes.X[I,:].data])
    data_j = np.array([barcodes.X[J,:].indices, barcodes.X[J,:].data])

    # If method == normal convert the uniformly distributed bins to
    # normally distributed values
    if method == 'normal':
        data_i[0,:] = uniform_to_normal(data_i[0,:], barcodes.shape[1])
        data_i = data_i[:, data_i[0,:].argsort()]

        data_j[0,:] = uniform_to_normal(data_j[0,:], barcodes.shape[1])
        data_j = data_j[:, data_j[0,:].argsort()]

    # If the mod parameter is non-negative, take the mod of the indices
    # to project the vectors into a lower dimensional space
    if mod > 0:
        data_i[0,:] = data_i[0,:] % mod
        data_i = data_i[:, data_i[0,:].argsort()]

        data_j[0,:] = data_j[0,:] % mod
        data_j = data_j[:, data_j[0,:].argsort()]

    i = 0
    j = 0
    sqr_sum = 0

    while (i < data_i.shape[1]) | (j < data_j.shape[1]):
        # Check for bins to combine
        if should_combine(data_i, i):
            data_i[1,i+1] = data_i[1,i+1] + data_i[1,i]
            i = i + 1
        elif should_combine(data_j, j):
            data_j[1,j+1] = data_j[1,j+1] + data_j[1,j]
            j = j + 1
        # Add a new element to the sum
        else:
            if i_next(data_i, data_j, i, j):
                sqr_sum = sqr_sum + data_i[1,i]**2
                i = i + 1
            elif add_next(data_i, data_j, i, j):
                sqr_sum = sqr_sum + (data_i[1,i] - data_j[1,j])**2
                i = i + 1
                j = j + 1
            else:
                sqr_sum = sqr_sum + data_j[1,j]**2
                j = j + 1

    return np.sqrt(sqr_sum)

# Returns distances between all pairs of beads in barcode space (column 0) and
# 2D-space (column 1)
def calculate_all_distances (barcodes):
    dists = np.zeros((barcodes.shape[0]**2, 2))
    dists[:,0] = euclidean_distances(barcodes.X).reshape((barcodes.shape[0]**2, ))
    beads = np.array([barcodes.obs.x, barcodes.obs.y]).T
    dists[:,1] = euclidean_distances(beads).reshape((barcodes.shape[0]**2, ))

    return dists

# Returns distances between n_comps pairs of beads in barcode space (column 0)
# and 2D-space (column 1)
def calculate_subset_distances (barcodes, n_comps):
    beads = np.array([barcodes.obs.x, barcodes.obs.y]).T
    dists = np.zeros((n_comps, 2))
    for m in range(n_comps):
        i = np.random.randint(low=0, high=barcodes.shape[0])
        j = np.random.randint(low=0, high=barcodes.shape[0])
        dists[m, :] = [sparse_euclidean_dist(barcodes, i, j),
                       euclidean_dist(beads[i,:], beads[j,:])]
    return dists

# Calculates the distances between random pairs of points in the underlying space
# and a UMAP embedding.
def calculate_embedding_distances(adata, n_reps=10**3, dist_thresh=float('inf')):
    B = adata.shape[0]
    dists = np.zeros((n_reps,2))
    for k in range(n_reps):
        spatial_distance = float('inf')
        while spatial_distance >= dist_thresh:
            i = np.random.randint(0,B)
            j = np.random.randint(0,B)
            i_coords = np.array([float(adata[i].obs.x), float(adata[i].obs.y)])
            j_coords = np.array([float(adata[j].obs.x), float(adata[j].obs.y)])
            i_umap_coords = np.array([float(adata[i].obs.umap_x), float(adata[i].obs.umap_y)])
            j_umap_coords = np.array([float(adata[j].obs.umap_x), float(adata[j].obs.umap_y)])
            spatial_distance = euclidean_dist(i_coords, j_coords)

        dists[k, 0] = euclidean_dist(i_coords, j_coords)
        dists[k, 1] = euclidean_dist(i_umap_coords, j_umap_coords)
    return dists

# Compute ncomps distances between pairs of points or areas of triangles formed by three
# points in both spatial and umap coordinate space such that at least one point lies in
# the set of points specified by the filter.
# umap_coords: Nx2 array of UMAP coordinates
# spatial_coords: Nx2 array of ground truth spatial coordinates
# filt: Filter limiting the choices for the first point in the pair/trio of points
# comp: Comparison method, either `dist` for pairwise distances or `area` for triangle area
# ncomps: Number of pairs/trios to calculate
def compute_large_umap_dists(umap_coords, spatial_coords, filt, comp, ncomps=10**4):
    dists = np.zeros((ncomps, 2))
    umap_annulus = umap_coords[filt]
    spatial_annulus = spatial_coords[filt]
    for n in range(ncomps):
        i = np.random.randint(0, len(umap_annulus))
        j = np.random.randint(0, len(umap_coords))
        k = np.random.randint(0, len(umap_coords))

        if comp == 'dist':
            dists[n,:] = [euclidean_dist(umap_annulus[i,:], umap_coords[j,:]),
                          euclidean_dist(spatial_annulus[i,:], spatial_coords[j,:])]
        elif comp == 'area':
            dists[n,:] = [triangle_area(umap_annulus[i,:], umap_coords[j,:],  umap_coords[k,:]),
                          triangle_area(spatial_annulus[i,:], spatial_coords[j,:],  spatial_coords[k,:])]

    return dists

# Compute ncomps distances between pairs of points or areas of triangles formed by three
# points in both spatial and umap coordinate space such that at least one point lies in
# the set of points specified by the filter and the other two points are no further than
# thresh from the first point.
# umap_coords: Nx2 array of UMAP coordinates
# spatial_coords: Nx2 array of ground truth spatial coordinates
# annulus_filt: Filter limiting the choices for the first point in the pair/trio of points
# large_filt: Helper filter restricting the data the second two points are chosen from.
#             Usually taken as an annulus with radius +/- thresh from the initial annulus.
# comp: Comparison method, either `dist` for pairwise distances or `area` for triangle area
# ncomps: Number of pairs/trios to calculate
# max_attempts: Maximum number of attempts to look for a coordinate within thresh of the first
#               coordinate without giving up. If the number of times max_attempts is reached
#               is non-zero a warning message will be printed.
def compute_small_umap_dists(umap_coords, spatial_coords, large_filt, annulus_filt, comp, thresh, ncomps=10**4, max_attempts=10**3):
    dists = np.zeros((ncomps, 2))

    umap_annulus = umap_coords[annulus_filt]
    spatial_annulus = spatial_coords[annulus_filt]

    umap_filtered = umap_coords[large_filt]
    spatial_filtered = spatial_coords[large_filt]

    failures = 0

    for n in range(ncomps):
        i = np.random.randint(0, len(umap_annulus))
        j = np.random.randint(0, len(umap_filtered))

        spatial_dist = euclidean_dist(spatial_annulus[i,:], spatial_filtered[j,:])
        attempts = 1
        while (spatial_dist > thresh) & (attempts < 1000):
            j = np.random.randint(0, len(umap_filtered))
            spatial_dist = euclidean_dist(spatial_annulus[i,:], spatial_filtered[j,:])
            attempts = attempts + 1

        if attempts == 1000:
            failures = failures + 1
        else:
            if comp == 'dist':
                dists[n,:] = [euclidean_dist(umap_annulus[i,:], umap_filtered[j,:]),
                              spatial_dist]

            if comp == 'area':
                k = np.random.randint(0, len(umap_filtered))
                spatial_dist = euclidean_dist(spatial_annulus[i,:], spatial_filtered[k,:])
                attempts = 1
                while (spatial_dist > thresh) & (attempts < 1000):
                    k = np.random.randint(0, len(umap_filtered))
                    spatial_dist = euclidean_dist(spatial_annulus[i,:], spatial_filtered[k,:])
                    attempts = attempts + 1

                if attempts == 1000:
                    failures = failures + 1
                else:
                    dists[n,:] = [triangle_area(umap_annulus[i,:], umap_filtered[j,:], umap_filtered[k,:]),
                                  triangle_area(spatial_annulus[i,:], spatial_filtered[j,:], spatial_filtered[k,:])]


    if failures > 0:
        print("{}/{} points found no distance within the distance threshold in {} attempts.".format(failures, ncomps, max_attempts))
    return dists


# Taken from https://gist.github.com/nh2/bc4e2981b0e213fefd4aaa33edfb3893
# Umeyama algorithm is Kabsch algorithm + scaling:
# https://stackoverflow.com/a/32244818/4195725
# http://web.stanford.edu/class/cs273/refs/umeyama.pdf

# Rigidly + scale (+ flip) aligns two point clouds with know point-to-point correspondences
# with least-squares error. allow_flip allows a rotation matrix with a det of -1,
# which allows changing the chirality of the point cloud.
# Returns:
# - Pt: transformed P such that Pt = P*cR + t
# - tuple: (scale factor c, rotation matrix R, translation vector t)
# such that:
#   SUM over point i ( | P_i*cR + t - Q_i |^2 ) is minimised
def rigid_registration(P, Q, allow_flip=True):
    assert P.shape == Q.shape
    n, dim = P.shape

    # Center both point clouds
    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    # Build correlation matrix
    C = np.dot(np.transpose(centeredP), centeredQ) / n

    # Compute SVD
    V, S, W = np.linalg.svd(C)

    # Check whether to correct the rotation matrix to ensure
    # a right-handed coordinate system
    # -> Since we also want to allow a flip (transpose) of the image, we skip this step.
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if not allow_flip and d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Compute rotation matrix
    R = np.dot(V, W)
#     print("det(R)=",np.linalg.det(R))

    # Compute scale factor
    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S)

    # Compute translation
    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    # Compute transformed pointcloud
    Pt = c*np.matmul(P,R) + t

    return Pt, (c, R, t)


# Registers UMAP coordinates to the GT coordinates, and compute distances to the GT.
def register_to_gt_and_compute_distance(umap_coords, spatial_coords, rel_scale=None):
    reg_umap, transform = rigid_registration(umap_coords, spatial_coords)
    if rel_scale is not None:
        distances = np.sqrt(((spatial_coords - reg_umap)**2).sum(1))/rel_scale
    else:
        distances = np.sqrt(((spatial_coords - reg_umap)**2).sum(1))
    return distances, reg_umap, transform
