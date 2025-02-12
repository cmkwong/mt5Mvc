import numpy as np

def split_matrix(arr, percentage=0.8, axis=0):
    """
    Split a numpy array along a specified axis into two parts based on a percentage.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array to split. Must be at least 2-dimensional.
    percentage : float, optional
        The percentage of the array to include in the first split (default is 0.8).
        Must be between 0 and 1.
    axis : int, optional
        The axis along which to split the array (default is 0).

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays:
        - The first array contains the first `percentage` portion of the input array
        - The second array contains the remaining portion

    Raises
    ------
    ValueError
        If the input array is not 2D or if percentage is not between 0 and 1
    TypeError
        If the input is not a numpy array
    """
    
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array")
    if arr.ndim < 2:
        raise ValueError("Input array must be at least 2-dimensional")
    if not 0 <= percentage <= 1:
        raise ValueError("Percentage must be between 0 and 1")
    if not isinstance(axis, int):
        raise TypeError("Axis must be an integer")
    if axis >= arr.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for array of dimension {arr.ndim}")

    split_index = int(arr.shape[axis] * percentage)
    slices = [slice(None)] * arr.ndim
    
    # First portion
    slices[axis] = slice(0, split_index)
    first_part = arr[tuple(slices)]
    
    # Second portion
    slices[axis] = slice(split_index, arr.shape[axis])
    second_part = arr[tuple(slices)]
    
    return first_part, second_part
