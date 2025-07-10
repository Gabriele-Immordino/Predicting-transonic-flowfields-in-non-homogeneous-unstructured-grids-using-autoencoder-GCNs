import numpy as np

def calculate_probability_array(elements: np.ndarray, p1: float, pn: float) -> np.ndarray:
    """
    Calculate a probability distribution for an array of elements using an exponential formula.

    Parameters:
    - elements: numpy array of mesh node indices (assumed to be integers from 0 to n-1)
    - p1: probability assigned to the smallest element
    - pn: probability assigned to the largest element

    Returns:
    - probabilities: numpy array of calculated probabilities for each element
    """
    n = len(elements)
    # Compute the exponent terms for the probability formula
    exponent_terms = -2 * (elements / n)
    # Calculate probabilities using the provided formula
    probabilities = 1 + ((1 - np.exp(exponent_terms)) / (1 - np.exp(-2))) * (pn - p1) + p1
    return probabilities

def get_reduced_space(
    variable: np.ndarray,
    coordinates: np.ndarray,
    point_id: np.ndarray,
    p1: float,
    pn: float,
    num_selected: int
) -> tuple:
    """
    Reduce the dimensionality of the input variable array by probabilistic selection.

    Steps:
    1. Add small random noise to the variable to avoid duplicate values.
    2. Sort the noisy variable in descending order.
    3. Calculate selection probabilities for each element.
    4. Randomly select a subset of elements based on these probabilities.
    5. Retrieve the corresponding reduced variable and coordinates.

    Parameters:
    - variable: numpy array of values to be reduced
    - coordinates: numpy array of coordinates, last column should match point_id
    - point_id: numpy array of point identifiers corresponding to variable
    - p1: probability for the smallest element
    - pn: probability for the largest element
    - num_selected: number of elements to select for the reduced space

    Returns:
    - variable_reduced: numpy array of selected variable values (as column vector)
    - coordinates_reduced: numpy array of coordinates corresponding to selected variables
    """
    # Step 1: Add small random noise to avoid duplicate values
    noise = np.random.uniform(low=0, high=variable.min() * 0.00001, size=variable.shape)
    variable_noisy = variable + noise

    # Step 2: Sort the noisy variable in descending order
    variables_sorted = np.sort(variable_noisy)[::-1]
    indices_sorted = np.arange(len(variables_sorted))

    # Step 3: Calculate selection probabilities for the sorted array
    result_probabilities = calculate_probability_array(indices_sorted, p1, pn)
    normalized_probabilities = result_probabilities / np.sum(result_probabilities)

    # Step 4: Randomly select indices based on the calculated probabilities
    selected_indices = np.random.choice(
        np.arange(len(variables_sorted)),
        size=num_selected,
        replace=False,
        p=normalized_probabilities
    )
    variables_sorted_reduced = variables_sorted[selected_indices]

    # Step 5: Find the intersection between noisy variable and selected values
    # and retrieve the corresponding indices in the original variable
    _, indices_variable, _ = np.intersect1d(
        variable_noisy, variables_sorted_reduced, return_indices=True
    )
    indices_variable = np.sort(indices_variable)

    # Retrieve reduced variable and corresponding point IDs
    variable_reduced = variable[indices_variable]
    ptids = point_id[indices_variable]
    variable_reduced = variable_reduced.reshape(-1, 1)

    # Retrieve coordinates corresponding to the selected point IDs
    coordinates_reduced = coordinates[np.isin(coordinates[:, -1], ptids)]

    return variable_reduced, coordinates_reduced
