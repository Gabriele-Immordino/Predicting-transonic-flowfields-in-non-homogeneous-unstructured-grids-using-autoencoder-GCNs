import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, distance

class WriteConnectivity:
    """
    Class to compute reduced connectivity between points in a mesh using Mahalanobis and Euclidean distances.
    """

    def __init__(self, normalized_dataset, point_id_, coordinates, num_neighbors=50, num_connection_per_node=6):
        """
        Initialize the WriteConnectivity object.

        Parameters:
        - normalized_dataset: np.ndarray
            The normalized feature dataset for each point.
        - point_id_: np.ndarray or list
            Array of point IDs corresponding to the dataset.
        - coordinates: np.ndarray
            Array of coordinates with point IDs as the last column.
        - num_neighbors: int
            Number of nearest neighbors to consider in the Euclidean space.
        - num_connection_per_node: int
            Number of connections to retain per node based on Mahalanobis distance.
        """
        self.normalized_dataset = normalized_dataset
        self.point_id_ = point_id_
        self.coordinates = coordinates
        self.num_neighbors = num_neighbors
        self.num_connection_per_node = num_connection_per_node

    def reduceSpace(self):
        """
        Reduce the space to relevant features and coordinates for each point.

        Returns:
        - reduced_space: list of tuples
            Each tuple contains (normalized features, point_id, x, y, z).
        """
        reduced_space = []
        np.random.seed(0)
        for i in range(self.normalized_dataset.shape[0]):
            # Find the coordinates for the current point ID
            coordinates_reduced = self.coordinates[np.isin(self.coordinates[:, -1], [self.point_id_[i]])][0]
            reduced_space.append((
                self.normalized_dataset[i],
                self.point_id_[i],
                coordinates_reduced[0],
                coordinates_reduced[1],
                coordinates_reduced[2]
            ))
        return reduced_space

    def reduced_connectivity(self):
        """
        Compute the reduced connectivity DataFrame for the mesh.

        Returns:
        - df: pd.DataFrame
            DataFrame with columns ['point_i', 'point_j', 'distance'] representing
            the normalized inverse Euclidean distance between connected points.
        """
        # Reduce the space to relevant features and coordinates
        reduced_space1 = self.reduceSpace()
        reduced_space1 = np.array(reduced_space1)

        # Extract coordinates and point IDs
        coarse_mesh_points = reduced_space1[:, 2:5]
        point_id_reduced = reduced_space1[:, 1].astype(int)

        # Build a KD-tree for efficient neighbor search
        tree = cKDTree(coarse_mesh_points)

        # Dictionary to store nearest neighbors for each point
        coarse_points_search_space = {}
        for i, coarse_point in enumerate(coarse_mesh_points):
            # Find the nearest neighbors in the mesh
            _, indices = tree.query(coarse_point, k=self.num_neighbors)
            coarse_points_search_space[point_id_reduced[i]] = [point_id_reduced[j] for j in indices]

        # Build a dictionary mapping point IDs to their coordinates
        coordinate_dict = {int(coord[3]): (float(coord[0]), float(coord[1]), float(coord[2])) for coord in self.coordinates}

        # Prepare fine mesh points for covariance calculation
        fine_points = np.array([list(point) for point in coordinate_dict.values()])

        # Compute covariance matrix and its pseudo-inverse
        cov_matrix = np.cov(fine_points, rowvar=False)
        cov_inv = np.linalg.pinv(cov_matrix)

        # Dictionary to store connections for each point
        new_conn = {p: [] for p in point_id_reduced}

        # Compute Mahalanobis and Euclidean distances for neighbors
        for point_id_1 in new_conn:
            x1, y1, z1 = coordinate_dict[point_id_1]
            for point_id_2 in coarse_points_search_space[point_id_1]:
                if point_id_1 != point_id_2:
                    x2, y2, z2 = coordinate_dict[point_id_2]
                    euc_dist = distance.euclidean((x1, y1, z1), (x2, y2, z2))
                    delta = np.array([x2 - x1, y2 - y1, z2 - z1])
                    m_dist = np.sqrt(np.dot(delta, np.dot(cov_inv, delta)))
                    new_conn[point_id_1].append((point_id_2, m_dist, euc_dist))

        # Build the reduced connectivity data
        reduced_connectivity_data = []
        for pt in new_conn:
            # Sort neighbors by Mahalanobis distance and keep the closest ones
            new_conn[pt].sort(key=lambda x: x[1])
            new_conn[pt] = new_conn[pt][:self.num_connection_per_node]
            for tup in new_conn[pt]:
                row = np.array([pt, tup[0], tup[2]])  # Store Euclidean distance
                reduced_connectivity_data.append(row)

        # Convert to NumPy array
        reduced_connectivity = np.array(reduced_connectivity_data)

        # Normalize the inverse Euclidean distances between 0 and 1
        max_distance = np.max(1.0 / reduced_connectivity[:, 2])
        normalized_distances = reduced_connectivity.copy()
        inv_dist = 1.0 / normalized_distances[:, 2]
        normalized_distances[:, 2] = inv_dist / max_distance

        # Build the DataFrame
        df = pd.DataFrame(normalized_distances, columns=['point_i', 'point_j', 'distance'])
        df['point_i'] = df['point_i'].astype(int)
        df['point_j'] = df['point_j'].astype(int)

        return df
