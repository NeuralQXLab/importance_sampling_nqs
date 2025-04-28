import numpy as np


def vscore(
    E: float | np.ndarray, varE: float | np.ndarray, N: int | np.ndarray
) -> float | np.ndarray:
    """
    Compute the V-score V = NVarE/(E^2) (assuming E_inf = 0)
    """
    return N * varE / (E**2)


def to_array(data: list):
    """
    Convert data in the list to a numpy array of floats
    """
    return np.array(data, dtype=float)
