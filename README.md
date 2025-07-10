Graph-Convolutional Autoencoder for Steady Aerodynamic Vector Fields Prediction
====================================================================

This script supports the paper:  
**"Predicting transonic flowfields in non–homogeneous unstructured grids using autoencoder graph convolutional networks"**  
Available at: [https://www.sciencedirect.com/science/article/pii/S0021999124009562](https://www.sciencedirect.com/science/article/pii/S0021999124009562)


Purpose:
--------
This code implements a deep graph-convolutional autoencoder for predicting steady aerodynamic surface distributions of pressure (CP) and skin-friction (CFx, CFy, CFz) over a 3D wing. It includes:

- Multi-resolution graph encoding via gradient-informed point selection,
- Interpolation-based pooling/unpooling using Moving Least Squares (MLS),
- Physics-informed loss function incorporating moment-based regularisation,
- Hyperparameter optimisation via Bayesian search (Optuna).

Key Features:
-------------
- Dataset: Steady CFD simulations with surface quantities [x, y, z, CP, CFx, CFy, CFz, M, AoA].
- Architecture: Multi-resolution GCN autoencoder with two levels of spatial reduction.
- Loss Function: Combines MSE on vector field with error on integrated aerodynamic moments.
- Training Strategy: Bayesian hyperparameter tuning.
- Output: Surface fields CP and CF predictions over complex geometries.


Dependencies:
-------------
To ensure compatibility with PyTorch Geometric and other core libraries, use the following conda environment:

```bash
Conda_environment/pytorch_gnn.tar.gz
```

-------------
Ensure the following input files are available before running the scripts:
- `Dataset\dataset.npy`: steady-state simulation data for $n$ samples;
- `Dimensionality_reduction\Adjency_matrix\surface.csv`: contains PointIDs of surface mesh nodes;
- `Dimensionality_reduction\Adjency_matrix\mesh.su2`: unstructured mesh file;
- `Dataset\grid_data.dat`: volume and surface normal data from Tecplot (4 columns: `X_Grid_K_Unit_Normal`, `Y_Grid_K_Unit_Normal`, `Z_Grid_K_Unit_Normal`, `Cell_Volume`).
- `Dimensionality_reduction\Pressure_gradient\input\dataset_pressure_gradient.npy`: pressure gradient distribution (for the dimensionality reduction module).

Workflow:
---------
1. **Generate adjacency matrix** for the unstructured surface mesh:

   ```bash
   write_connectivity_matrix_3D_unstructured.ipynb
   ```

2. **Run dimensionality reduction** to select graph nodes based on gradient fields:

   ```bash
   dimensionality_reduction_two_levels.ipynb
   ```

3. **Train the GCN model** with architecture optimisation:

   ```bash
   main.ipynb
   ```

Python Files Descriptions:
•	CodecManager.py: Assembles the graph hierarchy and constructs sparse interpolation matrices.
•	Codec.py: Generates encoder/decoder connectivity via Mahalanobis-weighted MLS.
•	Connectivity.py: Computes distance-weighted adjacency matrices for unstructured surfaces.
•	MLS_intrp_function.py: Implements MLS interpolation using polynomial basis and Gaussian kernels.
•	Reduced_space_PDF_function.py: Selects nodes for reduced graphs based on probabilistic pressure-gradient sampling.


Author: Gabriele Immordino