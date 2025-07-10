import numpy as np
import torch
from scipy.spatial.distance import euclidean

class MLSInterpolation:
    """
    Moving Least Squares (MLS) interpolation for 3D points.

    Attributes:
        x_src (array-like): Source x-coordinates.
        y_src (array-like): Source y-coordinates.
        z_src (array-like): Source z-coordinates.
        x_dst (float): Destination x-coordinate.
        y_dst (float): Destination y-coordinate.
        z_dst (float): Destination z-coordinate.
    """

    def __init__(self, x_src, y_src, z_src, x_dst, y_dst, z_dst):
        """
        Initialize the MLSInterpolation object with source and destination coordinates.

        Args:
            x_src (array-like): Source x-coordinates.
            y_src (array-like): Source y-coordinates.
            z_src (array-like): Source z-coordinates.
            x_dst (float): Destination x-coordinate.
            y_dst (float): Destination y-coordinate.
            z_dst (float): Destination z-coordinate.
        """
        self.x_src = x_src
        self.y_src = y_src
        self.z_src = z_src
        self.x_dst = x_dst
        self.y_dst = y_dst
        self.z_dst = z_dst

    def gaussian_weight(self, distance, h):
        """
        Compute the Gaussian weight for a given distance.

        Args:
            distance (float or np.ndarray): Distance(s) between points.
            h (float): Smoothing parameter.

        Returns:
            np.ndarray: Gaussian weights.
        """
        return np.exp(-distance**2 / h**2)

    def polynomial_basis(self, x):
        """
        Compute the quadratic polynomial basis for 3D coordinates.

        Args:
            x (np.ndarray): Array of shape (3, N) or (3,) representing coordinates.

        Returns:
            np.ndarray: Polynomial basis matrix.
        """
        # If input is 1D, reshape to (3, 1)
        x = np.atleast_2d(x)
        if x.shape[0] == 3 and x.shape[1] != 3:
            x = x.T
        # Each row is a point: [x, y, z]
        basis = np.column_stack([
            np.ones(x.shape[0]),
            x[:, 0],
            x[:, 1],
            x[:, 2],
            x[:, 0]**2,
            x[:, 1]**2,
            x[:, 2]**2,
            x[:, 0]*x[:, 1],
            x[:, 0]*x[:, 2],
            x[:, 1]*x[:, 2]
        ])
        return basis

    def calculate_distances(self):
        """
        Calculate Euclidean distances from each source point to the destination point.

        Returns:
            list: List of distances.
        """
        src_points = np.column_stack((self.x_src, self.y_src, self.z_src))
        dst_point = np.array([self.x_dst, self.y_dst, self.z_dst])
        distances = [euclidean(src_point, dst_point) for src_point in src_points]
        return distances

    def interpolation(self, dist=None):
        """
        Perform MLS interpolation and return the coefficients as torch tensors.

        Args:
            dist (list or np.ndarray, optional): Precomputed distances. If None, distances are calculated.

        Returns:
            list: List of torch tensors representing interpolation coefficients.
        """
        # Use provided distances or calculate them
        if dist is not None:
            distance_eucl = dist
        else:
            distance_eucl = self.calculate_distances()

        # Compute weight matrix W (diagonal)
        W = np.diag(self.gaussian_weight(np.abs(np.array(distance_eucl)), h=1.0))

        # Construct design matrix P for source points
        src_coords = np.array([self.x_src, self.y_src, self.z_src])
        P = self.polynomial_basis(src_coords)

        # Polynomial basis for the destination point
        dst_coords = np.array([self.x_dst, self.y_dst, self.z_dst])
        a = self.polynomial_basis(dst_coords)

        # Compute the MLS coefficients
        b = np.linalg.pinv(P.T @ W @ P)
        c = P.T @ W
        coefficients = a @ b @ c

        # Convert coefficients to torch tensors
        coefficients = [torch.tensor(coef, dtype=torch.float32) for coef in coefficients.flatten()]

        return coefficients
