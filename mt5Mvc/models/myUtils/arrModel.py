def split_matrix(arr, percentage=0.8, axis=0):
    """
    :param arr: np.array() 2D
    :param percentage: float
    :param axis: float
    :return: split array
    """
    cutOff = int(arr.shape[axis] * percentage)
    max = arr.shape[axis]
    I = [slice(None)] * arr.ndim
    I[axis] = slice(0, cutOff)
    upper_arr = arr[tuple(I)]
    I[axis] = slice(cutOff, max)
    lower_arr = arr[tuple(I)]
    return upper_arr, lower_arr