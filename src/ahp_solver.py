import numpy as np

def compute_ahp_weights(matrix):
    """
    Compute priority vector and consistency ratio from pairwise comparison matrix.
    """
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = np.argmax(eigvals.real)
    max_eigval = eigvals[max_index].real
    weights = eigvecs[:, max_index].real
    weights = weights / weights.sum()

    n = matrix.shape[0]
    ci = (max_eigval - n) / (n - 1)
    ri_values = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
    ri = ri_values.get(n, 1.49)
    cr = ci / ri if ri != 0 else 0

    return weights, max_eigval, ci, cr

def print_ahp_results(techs, weights, ci, cr):
    """
    Print AHP ranking results.
    """
    results = sorted(zip(techs, weights), key=lambda x: x[1], reverse=True)
    for tech, weight in results:
        print(f"{tech}: {weight:.4f}")
    print(f"\nCI: {ci:.4f}, CR: {cr:.4f}")
