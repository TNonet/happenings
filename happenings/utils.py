from typing import Union, List, Sequence
from warnings import warn

import numpy as np
import pandas as pd


def is_increasing_time(x) -> bool:
    if isinstance(x, pd.offsets.BaseOffset):
        return "-" in x.freqstr  # I can't find a better way to handle this....
    else:
        raise NotImplementedError()


def increase_date_freq(dates: pd.DatetimeIndex, infill_num: Union[int, float]) -> pd.DatetimeIndex:
    if dates.freq is not None:
        return pd.date_range(start=dates[0], end=dates[-1], freq=dates.freq / infill_num)
    else:
        if not isinstance(infill_num, int):
            raise ValueError(f"expected `infill_num` to by an integer if `dates` has no set freq, but got {infill_num}")
        # noinspection PyTypeChecker
        return pd.DatetimeIndex(infill(dates.to_numpy(), infill_num=infill_num))


def decrease_date_freq(dates: pd.DatetimeIndex, infill_num: Union[int, float]) -> pd.DatetimeIndex:
    new_dates = pd.DatetimeIndex(data=extract(dates, extract_num=infill_num), freq="infer")
    if new_dates.freq is None:
        warn("decrease_date_freq was unable to infer frequency, this may create problems with downstream operations.")
    return new_dates


def infill(arr: Sequence, infill_num: int) -> np.ndarray:
    """ Expands x by filling in values between neighboring entries

    Parameters
    ----------
    arr : numpy array of shape (N, )
        Array to infill
    infill_num : integer
        Must be positive

    Returns
    -------
    infill_array : numpy array of shape (infill_num * (N - 1) + 1, )

    Examples
    -------
    >>>infill(np.array([1,1,3]), infill_num=2)
    array([1, 1, 1, 2, 3])
    >>>infill(np.array([1,1,3]), infill_num=3)
    array([1, 1, 1, 1, 1, 2, 3])
    >>>np.round(infill(np.array([1,1,3]).astype(float), infill_num=3), decimals=2)
    array([1.  , 1.  , 1.  , 1.  , 1.67, 2.33, 3.  ])

    See Also
    --------
    utils.extract :
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise NotImplementedError(f"expected x to be a 1D array, but got {arr.ndim}")

    if infill_num < 1 or not isinstance(infill_num, int):
        raise ValueError(f"expected infill_num to be a positive integer, but got {infill_num}")
    elif infill_num == 1:
        return arr

    n = arr.shape[0]

    infilled_x = np.zeros(infill_num * (n - 1) + 1, dtype=arr.dtype)

    x_diff = np.diff(arr)

    infilled_x[::infill_num] = arr

    for sub_sampling_index in range(2, infill_num + 1):
        infilling_values = arr[:-1] + (sub_sampling_index - 1) * x_diff / infill_num
        infilled_x[sub_sampling_index - 1 :: infill_num] = infilling_values

    return infilled_x


def extract(arr: Sequence, extract_num: int) -> np.ndarray:
    """

    Parameters
    ----------
    arr
    extract_num

    Returns
    -------

    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise NotImplementedError(f"expected x to be a 1D array, but got {arr.ndim}")

    if extract_num < 1 or not isinstance(extract_num, int):
        raise ValueError(f"expected infill_num to be a positive integer, but got {extract_num}")
    elif extract_num == 1:
        return arr

    return arr[::extract_num]


def support_sections(arr: Sequence) -> List[List[int]]:
    """ Returns a list of start and end indices of sections with non-zero support

    Parameters
    ----------
    arr : Sequence of numbers

    Returns
    -------
    support_sections: List of Lists of 2 integers
        support_sections[i] is the i-th non-zero support section
        support_sections[i][0] is the start index of the i-th non-zero support section
        support_sections[i][1] is the end index of the i-th non-zero support section

    Examples
    --------
    >>>support_sections([0,0,1,1,1,0,1,0,0,0,1,1,1])
    [[2, 5], [6, 7], [10, 13]]

    Notes
    -----
    Source:
        https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array

    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"expected x to be a 1D array, but got {arr.ndim}D")

    # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    isnonzero = np.concatenate(([0], (arr != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    return np.where(absdiff == 1)[0].reshape(-1, 2).tolist()


def center_value(arr: Sequence) -> float:
    """Returns the center value of an array.

    If arr is of even length, returns the average of the two "middle" values

    Parameters
    ----------
    arr : Sequence of values that support __add__ and __div__.
        Most likely a list or numpy array of numbers

    Examples
    --------
    >>>center_value([1,2,3])
    2
    >>>center_value([1,4,8,1])
    6
    Returns
    -------
    center_value : The center value of an array.
    """
    arr = np.asarray(arr)

    if arr.ndim != 1:
        raise ValueError

    n = len(arr)
    n_over_2 = n // 2

    if n % 2:
        mid_left, mid_right = arr[n_over_2 - 1 : n_over_2 + 1]
        return (mid_left + mid_right) / 2
    else:
        return arr[n_over_2]
