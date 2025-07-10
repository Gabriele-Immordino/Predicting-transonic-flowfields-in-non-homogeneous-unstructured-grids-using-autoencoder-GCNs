import numpy as np
from scipy.spatial import cKDTree, distance
import torch
from functions import MLS_intrp_function

class Codec:
    """
    Codec class for dimensionality reduction and connectivity computation
    using Euclidean and Mahalanobis distances, and MLS interpolation.
    """

    def __init__(self, normalized_dataset, point_id_, coordinates):
        """
        Initialize the Codec object.

        Args:
            normalized_dataset (np.ndarray): The normalized dataset.
            point_id_ (np.ndarray): Array of point IDs.
            coordinates (np.ndarray): Array of coordinates with point IDs.
        """
        self.normalized_dataset = normalized_dataset
        self.point_id_ = point_id_
        self.coordinates = coordinates

    def reduceSpace(self):
        """
        Reduce the space by associating each normalized data point with its coordinates.

        Returns:
            list: List of tuples containing (normalized_data, point_id, x, y, z).
        """
        reduced_space = []
        np.random.seed(0)
        for i in range(self.normalized_dataset.shape[0]):
            # Find the coordinates corresponding to the current point_id
            coordinates_reduced = self.coordinates[np.isin(self.coordinates[:, -1], [self.point_id_[i]])][0]
            reduced_space.append((
                self.normalized_dataset[i],
                self.point_id_[i],
                coordinates_reduced[0],
                coordinates_reduced[1],
                coordinates_reduced[2]
            ))
        return reduced_space

    def calc_euc_dist(self, src_space_coords, src_point_id, dest_space_coords, dest_point_id, num_neighbors=250):
        """
        Calculate the Euclidean distance between source and destination coordinates,
        and find the nearest neighbors.

        Args:
            src_space_coords (np.ndarray): Source coordinates.
            src_point_id (np.ndarray): Source point IDs.
            dest_space_coords (np.ndarray): Destination coordinates.
            dest_point_id (np.ndarray): Destination point IDs.
            num_neighbors (int): Number of neighbors to search.

        Returns:
            dict: Mapping from source point ID to list of nearest destination point IDs.
        """
        tree = cKDTree(dest_space_coords)
        points_search_space = {}
        for i, points in enumerate(src_space_coords):
            # Find the indices of the nearest neighbors
            _, indices = tree.query(points, k=num_neighbors)
            points_search_space[src_point_id[i]] = [dest_point_id[j] for j in indices]
        return points_search_space

    def calc_Mahal_dist(self, dest_point_id, points_search_space, num_neigh_connectivity, return_conn_matrix=False):
        """
        Calculate Mahalanobis and Euclidean distances for connectivity.

        Args:
            dest_point_id (np.ndarray): Destination point IDs.
            points_search_space (dict): Mapping from point ID to neighbor IDs.
            num_neigh_connectivity (int): Number of neighbors to keep for connectivity.
            return_conn_matrix (bool): Whether to return the connectivity matrix.

        Returns:
            tuple: (connectivity dictionary, coordinate dictionary) or
                   (connectivity matrix, coordinate dictionary)
        """
        # Build a dictionary mapping point ID to coordinates
        coordinate_dict = {int(coord[3]): (float(coord[0]), float(coord[1]), float(coord[2])) for coord in self.coordinates}
        fine_points = np.array([list(point) for point in coordinate_dict.values()])
        cov_matrix = np.cov(fine_points, rowvar=False)
        cov_inv = np.linalg.pinv(cov_matrix)
        new_conn = {p: [] for p in dest_point_id}

        for point_id_1 in new_conn:
            x1, y1, z1 = coordinate_dict[point_id_1]
            for point_id_2 in points_search_space[point_id_1]:
                x2, y2, z2 = coordinate_dict[point_id_2]
                euc_dist = distance.euclidean((x1, y1, z1), (x2, y2, z2))
                delta = np.array([x2 - x1, y2 - y1, z2 - z1])
                m_dist = np.sqrt(np.dot(delta, np.dot(cov_inv, delta)))
                new_conn[point_id_1].append((point_id_2, m_dist, euc_dist))

        connectivity_data = []
        for pt in new_conn:
            # Sort neighbors by Mahalanobis distance and keep the closest ones
            new_conn[pt].sort(key=lambda x: x[1])
            new_conn[pt] = new_conn[pt][:num_neigh_connectivity]
            for tup in new_conn[pt]:
                row = np.array([pt, tup[0], tup[2]])
                connectivity_data.append(row)

        if not return_conn_matrix:
            return new_conn, coordinate_dict

        # Convert the list of rows into a NumPy array
        connectivity = np.array(connectivity_data)
        return connectivity, coordinate_dict

    def reduced_connectivity(self, mode):
        """
        Compute reduced connectivity and MLS interpolation coefficients.

        Args:
            mode (str): Either 'encoder' or 'decoder'.

        Returns:
            tuple: (interpolation coefficients dictionary, point ID data dictionary)
        """
        if mode.lower() not in ['encoder', 'decoder']:
            raise ValueError('Mode parameter must be "encoder" or "decoder"')

        reduced_space1 = self.reduceSpace()
        reduced_space1 = np.array(reduced_space1)
        point_id_reduced = reduced_space1[:, 1].astype(int)

        if mode.lower() == 'encoder':
            # Encoder: from reduced space to full space
            encoded_points_search_space = self.calc_euc_dist(
                src_space_coords=reduced_space1[:, 2:5],
                src_point_id=point_id_reduced,
                dest_space_coords=self.coordinates[:, :3],
                dest_point_id=self.coordinates[:, -1].astype(int)
            )
            full_connectivity, coordinate_dict = self.calc_Mahal_dist(
                dest_point_id=point_id_reduced,
                points_search_space=encoded_points_search_space,
                num_neigh_connectivity=12
            )
        else:
            # Decoder: from full space to reduced space
            encoded_points_search_space = self.calc_euc_dist(
                src_space_coords=self.coordinates[:, :3],
                src_point_id=self.coordinates[:, -1].astype(int),
                dest_space_coords=reduced_space1[:, 2:5],
                dest_point_id=point_id_reduced
            )
            full_connectivity, coordinate_dict = self.calc_Mahal_dist(
                dest_point_id=self.coordinates[:, -1].astype(int),
                points_search_space=encoded_points_search_space,
                num_neigh_connectivity=12
            )

        # Normalize all distances between 0 and 1 and compute MLS interpolation coefficients
        intrp_coeffs = {}
        point_id_data = {}
        for key, values in full_connectivity.items():
            point_ids = [torch.tensor((value[0]), dtype=torch.int32) for value in values]

            # Moving Weighted Least Squares (MLS) interpolation
            x = [list(coordinate_dict[point_id[0]])[0] for point_id in values]
            y = [list(coordinate_dict[point_id[0]])[1] for point_id in values]
            z = [list(coordinate_dict[point_id[0]])[2] for point_id in values]

            MLS = MLS_intrp_function.MLS_interpolation(
                x_src=x,
                y_src=y,
                z_src=z,
                x_dst=list(coordinate_dict[key])[0],
                y_dst=list(coordinate_dict[key])[1],
                z_dst=list(coordinate_dict[key])[2]
            )

            intrp_coeffs[key] = MLS.interpolation()  # Update with the MLS results
            point_id_data[key] = point_ids

        return intrp_coeffs, point_id_data
