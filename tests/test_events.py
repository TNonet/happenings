import datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis.strategies import floats
from scipy.signal import windows, convolve as scipy_convolve

from happenings.core import function_event
from happenings.event import MONTHS, DAY_OF_MONTH, DAY_OF_YEAR
from tests.utils.utils import date_range_indexes

JAN_1_2019 = datetime.datetime.strptime("2019-01-01", "%Y-%M-%d")
JAN_1_2020 = datetime.datetime.strptime("2020-01-01", "%Y-%M-%d")
JAN_1_2021 = datetime.datetime.strptime("2021-01-01", "%Y-%M-%d") - datetime.timedelta(days=1)

DATES_2020 = pd.date_range(start=JAN_1_2020, end=JAN_1_2021, freq="D")
DATES_2019 = pd.date_range(start=JAN_1_2019, end=JAN_1_2020, freq="D")

JAN_EVENT = MONTHS["January"]
FEB_EVENT = MONTHS["February"]
MAR_EVENT = MONTHS["March"]
JAN_OR_FEB_EVENT = function_event(name="Jan_or_Feb",
                                  function=lambda x: x.month in [1, 2],
                                  vectorized=False)


def test_month_events():
    assert sum(JAN_EVENT.sample_from(dates=DATES_2020)) == 31
    assert sum(FEB_EVENT.sample_from(dates=DATES_2020)) == 29  # Leap Year
    assert sum(FEB_EVENT.sample_from(dates=DATES_2019)) == 28


def test_derived_events():
    assert sum((JAN_EVENT * FEB_EVENT).sample_from(DATES_2019)) == 0
    assert sum((JAN_EVENT + FEB_EVENT).sample_from(DATES_2020)) == 31 + 29


def test_union_events():
    np.testing.assert_array_equal((JAN_EVENT + FEB_EVENT).sample_from(DATES_2020),
                                  JAN_OR_FEB_EVENT.sample_from(dates=DATES_2020))
    np.testing.assert_array_equal((JAN_EVENT | FEB_EVENT).sample_from(DATES_2020),
                                  JAN_OR_FEB_EVENT.sample_from(dates=DATES_2020))


@given(t=date_range_indexes())
def test_renamed(t):
    jan_event_2 = JAN_EVENT.renamed("January_2")
    assert jan_event_2.name == "January_2"
    np.testing.assert_array_equal(jan_event_2.sample_from(t).to_numpy(),
                                  JAN_EVENT.sample_from(t).to_numpy())


def test_validate_dates():
    with pytest.raises(TypeError):
        JAN_EVENT.sample_from(0)
    with pytest.raises(ValueError):
        JAN_EVENT.sample_from(pd.DatetimeIndex([datetime.datetime.today(), datetime.datetime.today()]))


@pytest.mark.parametrize("event", [JAN_EVENT,
                                   JAN_EVENT + MAR_EVENT,
                                   np.sin(DAY_OF_MONTH),
                                   DAY_OF_YEAR % DAY_OF_MONTH])
@pytest.mark.parametrize("window_1", [windows.gaussian(100, 10), windows.gaussian(100, 1)])
@pytest.mark.parametrize("window_2", [windows.gaussian(100, 10), windows.gaussian(100, 1)])
@given(t=date_range_indexes())
def test_convolution_through_distributivity(event, window_1, window_2, t):
    """
    We know that f conv (g + h) == f conv g + f conv h.
    """
    np.testing.assert_array_almost_equal(event.convolve(window_1 + window_2).sample_from(t).to_numpy(),
                                         (event.convolve(window_1) + event.convolve(window_2)).sample_from(t).to_numpy())


@pytest.mark.parametrize("event", [JAN_EVENT,
                                   JAN_EVENT + MAR_EVENT,
                                   np.sin(DAY_OF_MONTH),
                                   DAY_OF_YEAR % DAY_OF_MONTH])
@pytest.mark.parametrize("window_1", [windows.gaussian(100, 10), windows.gaussian(100, 1)])
@given(t=date_range_indexes(), scalar=floats(allow_nan=False, max_value=10, min_value=-10))
def test_convolution_through_scalar_distributivity(event, scalar, window_1, t):
    """
    We know that f conv (g + h) == f conv g + f conv h.
    """
    np.testing.assert_array_almost_equal(event.convolve(scalar*window_1).sample_from(t).to_numpy(),
                                         (scalar*event.convolve(window_1)).sample_from(t).to_numpy(),)


# @pytest.mark.parametrize("event", [JAN_EVENT,
#                                    JAN_EVENT + MAR_EVENT,
#                                    np.sin(DAY_OF_MONTH),
#                                    DAY_OF_YEAR % DAY_OF_MONTH])
# @pytest.mark.parametrize("window_1", [windows.gaussian(100, 10), windows.gaussian(100, 1)])
# @pytest.mark.parametrize("window_2", [windows.gaussian(100, 10), windows.gaussian(100, 1)])
# @given(t=date_range_indexes())
# def test_convolution_through_associativity(event, window_1, window_2, t):
#     """
#     We know that f conv (g conv h) == (f conv g) conv  h
#     """
#     np.testing.assert_array_almost_equal(event.convolve(window_1).convolve(window_2).sample_from(t).to_numpy(),
#                                          (event.convolve(scipy_convolve(window_1, window_2)).sample_from(t).to_numpy()))
#