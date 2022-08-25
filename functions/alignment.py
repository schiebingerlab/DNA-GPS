import numpy as np
import anndata
import math

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
def align(umap_coords, spatial_coords, rel_scale=1):
    # Normalize and center the spatial coords in [0,1]^2
    # if normalize == True:
    #     xrange = max(spatial_coords[:,0]) - min(spatial_coords[:,0])
    #     yrange = max(spatial_coords[:,1]) - min(spatial_coords[:,1])
    #
    #     xbuff = max(0, yrange - xrange)/2
    #     ybuff = max(0, xrange - yrange)/2
    #
    #     spatial_coords[:,0] = (spatial_coords[:,0] - min(spatial_coords[:,0]) + xbuff) / max(xrange, yrange)
    #     spatial_coords[:,1] = (spatial_coords[:,1] - min(spatial_coords[:,1]) + ybuff) / max(xrange, yrange)

    reg_umap, transform = rigid_registration(umap_coords, spatial_coords)

    distances = np.sqrt(((spatial_coords - reg_umap)**2).sum(1))/rel_scale

    return distances, reg_umap, transform
