import torch
import numpy as np
import pandas as pd
import pickle

class CodecManager:
    """
    Manages encoding and decoding operations for unstructured grid data,
    including edge indices, distances, and interpolation matrices for pooling/unpooling.
    """

    def __init__(self, npoints, point_id_orig, edge_indices_orig, edge_distances):
        """
        Initialize the CodecManager with grid and connectivity data.

        Args:
            npoints (int): Number of points in the grid.
            point_id_orig (array-like): Original point IDs.
            edge_indices_orig (pd.DataFrame): DataFrame with columns ['point_i', 'point_j'] for edges.
            edge_distances (array-like): Distances for each edge.
        """
        self.edge_distances = edge_distances
        self.point_id_orig = point_id_orig
        path_dim_red = 'Dimensionality_reduction\\Pressure_gradient\\output\\'

        # Load reduced connectivity array
        with open(f'{path_dim_red}reduced_connectivity_array_{npoints}.pkl', 'rb') as file:
            reduced_connectivity_array = pickle.load(file)  # columns: point_i, point_j, distance

        # Load decoder (unpooling) and encoder (pooling) data
        self.unpool_point_id_data_list = torch.load(f'{path_dim_red}decoder_point_id_data_list_{npoints}.pt')
        self.unpool_interp_coeff_list = torch.load(f'{path_dim_red}decoder_interp_coeff_list_{npoints}.pt')
        self.pool_point_id_data_list = torch.load(f'{path_dim_red}encoder_point_id_data_list_{npoints}.pt')
        self.pool_interp_coeff_list = torch.load(f'{path_dim_red}encoder_interp_coeff_list_{npoints}.pt')

        # Extract reduced edge indices and distances
        self.edge_indices_reduced_orig = reduced_connectivity_array[['point_i', 'point_j']]
        self.edge_distances_reduced = reduced_connectivity_array[['distance']]
        self.point_id_reduced_orig = pd.unique(reduced_connectivity_array['point_i'])
        self.point_id_index_reduced_orig = np.where(np.isin(point_id_orig, self.point_id_reduced_orig))[0]

        # Remap edge indices to contiguous indices
        self.edge_indices = self.rename_edge_indices(point_id_orig, edge_indices_orig)
        self.edge_indices_reduced = self.rename_edge_indices(self.point_id_reduced_orig, self.edge_indices_reduced_orig)

        # --- Unpooling (Decoder) ---
        # Remap and convert unpooling data to tensors
        self.unpool_point_id_data_list_ = self.convert_dict_tensor(
            self.rename_data_values(self.unpool_point_id_data_list, self.point_id_reduced_orig)
        )
        self.unpool_interp_coeff_list_ = self.convert_dict_tensor(self.unpool_interp_coeff_list)

        # Sort unpooling dictionaries by original point order
        point_id_orig_temp = np.argsort(point_id_orig)
        self.unpool_point_id_data_list_sorted = {
            k: self.unpool_point_id_data_list_[k]
            for k in sorted(self.unpool_point_id_data_list_.keys(), key=lambda x: point_id_orig_temp[x])
        }
        self.unpool_interp_coeff_list_sorted = {
            k: self.unpool_interp_coeff_list_[k]
            for k in sorted(self.unpool_interp_coeff_list_.keys(), key=lambda x: point_id_orig_temp[x])
        }

        # --- Pooling (Encoder) ---
        # Remap and convert pooling data to tensors
        self.pool_point_id_data_list_ = self.convert_dict_tensor(
            self.rename_data_values(self.pool_point_id_data_list, point_id_orig)
        )
        self.pool_interp_coeff_list_ = self.convert_dict_tensor(self.pool_interp_coeff_list)

        # Sort pooling dictionaries by original point order
        self.pool_point_id_data_list_sorted = {
            k: self.pool_point_id_data_list_[k]
            for k in sorted(self.pool_point_id_data_list_.keys(), key=lambda x: point_id_orig_temp[x])
        }
        self.pool_interp_coeff_list_sorted = {
            k: self.pool_interp_coeff_list_[k]
            for k in sorted(self.pool_interp_coeff_list_.keys(), key=lambda x: point_id_orig_temp[x])
        }

    def get_edge_indices(self):
        """
        Returns the edge indices as a torch tensor (shape: [2, num_edges]).
        """
        return torch.tensor(np.array(self.edge_indices), dtype=torch.int64).t().contiguous()

    def get_edge_distances(self):
        """
        Returns the edge distances as a torch tensor.
        """
        return torch.tensor(np.array(self.edge_distances).squeeze(), dtype=torch.float32).t().contiguous()

    def get_indices_reduced(self):
        """
        Returns the reduced edge indices as a torch tensor.
        """
        return torch.tensor(np.array(self.edge_indices_reduced), dtype=torch.int64).t().contiguous()

    def get_edge_distances_reduced(self):
        """
        Returns the reduced edge distances as a torch tensor.
        """
        return torch.tensor(np.array(self.edge_distances_reduced).squeeze(), dtype=torch.float32).t().contiguous()

    def get_point_id_index_reduced(self):
        """
        Returns the indices of reduced points in the original point array as a torch tensor.
        """
        return torch.tensor(np.array(self.point_id_index_reduced_orig), dtype=torch.int64).t().contiguous()

    def get_interpolation_matrix(self):
        """
        Returns the encoder and decoder sparse interpolation matrices.

        Returns:
            tuple: (encoder_sparse_interpolation, decoder_sparse_interpolation)
        """
        encoder_interpolation, encoder_sparse_interpolation = self.create_interpolation_matrix(
            self.pool_interp_coeff_list_sorted,
            self.pool_point_id_data_list_sorted,
            self.point_id_orig,
            decoder=False
        )
        decoder_interpolation, decoder_sparse_interpolation = self.create_interpolation_matrix(
            self.unpool_interp_coeff_list_sorted,
            self.unpool_point_id_data_list_sorted,
            self.point_id_reduced_orig,
            decoder=True
        )
        return (encoder_sparse_interpolation, decoder_sparse_interpolation)

    def rename_edge_indices(self, point_id, edge_indices):
        """
        Remap edge indices to contiguous indices based on sorted point IDs.

        Args:
            point_id (array-like): Array of point IDs.
            edge_indices (pd.DataFrame): DataFrame with columns ['point_i', 'point_j'].

        Returns:
            pd.DataFrame: DataFrame with remapped indices.
        """
        if not set(edge_indices['point_i']).issubset(point_id) or not set(edge_indices['point_j']).issubset(point_id):
            raise ValueError("At least one point in edge_indices does not exist in point_id")
        sorted_points = np.sort(point_id)
        label_to_index = {label: idx for idx, label in enumerate(sorted_points)}
        edge_indices_copy = edge_indices.copy()
        edge_indices_copy['point_i'] = edge_indices_copy['point_i'].map(label_to_index)
        edge_indices_copy['point_j'] = edge_indices_copy['point_j'].map(label_to_index)
        return edge_indices_copy

    def convert_dict_tensor(self, data_dict):
        """
        Stack lists of tensors in a dictionary into single tensors.

        Args:
            data_dict (dict): Dictionary of lists of tensors.

        Returns:
            dict: Dictionary of stacked tensors.
        """
        for key in data_dict:
            data_dict[key] = torch.stack(data_dict[key])
        return data_dict

    def rename_data_values(self, point_id_data_list, point_id):
        """
        Remap point IDs in the data list to contiguous indices.

        Args:
            point_id_data_list (dict): Dictionary of lists of point IDs (as tensors).
            point_id (array-like): Array of point IDs.

        Returns:
            dict: Dictionary with remapped point IDs.
        """
        sorted_points = np.sort(point_id)
        label_to_index = {label: idx for idx, label in enumerate(sorted_points)}
        renamed_data_list = {}
        for key, data_list in point_id_data_list.items():
            renamed_data_list[key] = [
                torch.tensor(label_to_index.get(value.item(), value), dtype=torch.int64)
                for value in data_list
            ]
        return renamed_data_list

    def create_interpolation_matrix(self, data_list, point_id_data_list_, point_orig, decoder=False):
        """
        Create a dense and sparse interpolation matrix from data.

        Args:
            data_list (dict): Dictionary of interpolation coefficients.
            point_id_data_list_ (dict): Dictionary of point indices for interpolation.
            point_orig (array-like): Array of original point IDs.
            decoder (bool): If True, creates decoder matrix; otherwise, encoder.

        Returns:
            tuple: (dense_interpolation_matrix, sparse_interpolation_matrix)
        """
        interpolation_matrix = np.zeros((len(point_id_data_list_), len(point_orig)), dtype=float)

        for i, (row_key, indices) in enumerate(point_id_data_list_.items()):
            weights = data_list[row_key]
            interpolation_matrix[i, indices] = weights

        interpolation_matrix = torch.tensor(interpolation_matrix, dtype=torch.float32)
        sparse_interpolation_matrix = interpolation_matrix.to_sparse_coo()

        return interpolation_matrix, sparse_interpolation_matrix
